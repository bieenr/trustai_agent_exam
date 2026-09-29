from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from trusted_ai.taste import TasteConfig, TastePaths, load_dataset
from trusted_ai.taste.collaborative import compute_user_similarity
from trusted_ai.taste.config import EmbeddingSettings
from trusted_ai.taste.content import EmbeddingSpace, TfidfSpace, taste_weights
from trusted_ai.taste.dataset import assert_no_test_leak
from trusted_ai.taste.embedding import EmbeddingClient, MovieEmbeddings, VectorCache
from trusted_ai.taste.filters import candidate_mask
from trusted_ai.taste.genre_profile import avoid_genres, blind_spots, genre_preferences
from trusted_ai.taste.learned import STACK_FEATURES, MlpRatings, ScoreModel
from trusted_ai.taste.recommend import recommend_for_user, select_by_verdict
from trusted_ai.taste.scoring import CandidateScorer
from trusted_ai.taste.search import DescriptionSearch, QueryEncoder


MOVIES = [
    (1, "Heist One", "Crime|Thriller", "a crew plans a bank vault heist robbery"),
    (2, "Heist Two", "Crime|Thriller", "the heist crew cracks a vault in a robbery"),
    (3, "Scary House", "Horror", "a haunted house ghost brings terror"),
    (4, "Scary Woods", "Horror", "ghost terror in the haunted woods"),
    (5, "Space Trip", "Sci-Fi", "a spaceship crew lands on an alien planet"),
    (6, "Love Story", "Romance", "a romance ends in a wedding"),
    (7, "Heist Three", "Crime", "robbery of a vault, a heist gone wrong"),
    (8, "Ghost Three", "Horror", "a haunted ghost story of terror"),
    (9, "Scary Ship", "Horror|Sci-Fi", "ghost terror aboard a haunted spaceship"),
    (10, "Quiet Drama", "Drama", "a family argues over dinner"),
]
TARGET = {1: 5.0, 2: 4.5, 3: 2.0, 4: 2.0, 5: 4.0, 6: 3.5, 9: 2.0}
PEER = {1: 5.0, 2: 4.5, 3: 2.0, 4: 2.5, 5: 4.0, 6: 3.0, 7: 5.0, 8: 1.0}
OPPOSITE = {1: 1.0, 2: 1.5, 3: 5.0, 4: 4.5, 5: 2.0, 6: 4.0, 7: 1.0}
FEW_COMMON = {1: 5.0, 2: 4.5, 3: 2.0, 7: 1.0}
LAST_RATING_TS = 1_600_000_000  # 2020-09-13 UTC


def write_dataset(root: Path, test_rows: list[tuple[int, int, float]]) -> TastePaths:
    paths = TastePaths(root)
    paths.train_ratings.parent.mkdir(parents=True)
    pd.DataFrame(
        [(mid, title, 2000 + mid, genres, plot) for mid, title, genres, plot in MOVIES],
        columns=["movieId", "title", "year", "genres", "plot"],
    ).to_csv(paths.movies, index=False)
    users = {1: TARGET, 2: PEER, 3: PEER, 4: {**PEER, 6: 3.5}, 5: OPPOSITE, 6: FEW_COMMON}
    # Latest rating in 2020, after every movie's release year (2001–2010).
    rows = [(uid, mid, rating, LAST_RATING_TS) for uid, ratings in users.items() for mid, rating in ratings.items()]
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(paths.train_ratings, index=False)
    pd.DataFrame(test_rows, columns=["userId", "movieId", "rating"]).to_csv(paths.test_ratings, index=False)
    return paths



def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)

class TasteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = write_dataset(Path(self._tmp.name), [(1, 10, 4.0)])
        self.config = TasteConfig(paths=self.paths)
        self.dataset = load_dataset(self.paths)
        self.similarity = compute_user_similarity(self.dataset)
        self.tfidf = TfidfSpace(self.dataset, self.config)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class GenreAndPeersTest(TasteTestCase):
    def test_genre_deviation_uses_own_mean_and_confidence(self) -> None:
        preferences = {row["genre"]: row for row in genre_preferences(self.dataset, 1)}
        mean = np.mean(list(TARGET.values()))
        self.assertAlmostEqual(preferences["Horror"]["deviation"], round(2.0 - mean, 3))
        self.assertEqual(preferences["Horror"]["confidence"], "medium")
        self.assertEqual(preferences["Romance"]["confidence"], "low")
        avoided = [row["genre"] for row in avoid_genres(list(preferences.values()), -0.5)]
        self.assertEqual(avoided, ["Horror"])  # Sci-Fi dips too, but only on 2 ratings

    def test_blind_spots_include_never_rated_genres(self) -> None:
        self.assertIn("Drama", [row["genre"] for row in blind_spots(self.dataset, 1)])

    def test_similar_users_require_enough_common_ratings(self) -> None:
        peers = self.similarity.top_k(1, k=10, min_common=5)
        ids = [peer["user_id"] for peer in peers]
        self.assertEqual(set(ids), {2, 3, 4})
        self.assertNotIn(6, ids)  # identical taste on 3 movies is not enough evidence
        self.assertNotIn(5, ids)  # anti-correlated users are not similar users

    def test_taste_weights_favour_above_average_ratings(self) -> None:
        ratings = self.dataset.user_ratings(1)
        ids, weights = taste_weights(ratings, 4.0)
        self.assertEqual(list(ids), [1, 2, 5])
        self.assertGreater(weights[0], weights[1])
        uniform = taste_weights(pd.DataFrame({"movieId": [1, 2], "rating": [5.0, 5.0]}), 4.0)[1]
        self.assertEqual(list(uniform), [1.0, 1.0])

    def test_leak_check_rejects_test_pairs_in_train(self) -> None:
        assert_no_test_leak(self.dataset, self.paths.test_ratings)
        leaky = write_dataset(Path(self._tmp.name) / "leaky", [(1, 1, 5.0)])
        with self.assertRaises(ValueError):
            assert_no_test_leak(load_dataset(leaky), leaky.test_ratings)


class CandidateScorerTest(TasteTestCase):
    def build(self, llm_judge=None, **overrides: object) -> CandidateScorer:
        return CandidateScorer(self.dataset, self.similarity, self.tfidf,
                               replace(self.config, **overrides), llm_judge)

    def test_similar_user_prediction_comes_first(self) -> None:
        loved, hated = self.build().score(1, 7), self.build().score(1, 8)
        self.assertEqual((loved["verdict"], loved["source"]), ("pass", "similar_user_pseudo_label"))
        self.assertEqual(loved["evidence"]["n_similar_raters"], 3)
        self.assertGreaterEqual(loved["evidence"]["peer_predicted_rating"], 4.0)
        self.assertEqual((hated["verdict"], hated["evidence"]["weighted_avg"]), ("fail", 1.0))
        self.assertLessEqual(hated["evidence"]["peer_predicted_rating"], 2.5)

    def test_include_genres_requires_every_listed_genre(self) -> None:
        mask = candidate_mask(self.dataset, 6, include_genres=["Horror", "Sci-Fi"])
        self.assertEqual(self.dataset.movies.loc[mask, "movieId"].tolist(), [9])

    def test_no_movie_released_after_the_users_last_rating(self) -> None:
        self.dataset.user_last_year[1] = 2008  # pretend user 1 stopped rating in 2008
        picked = recommend_for_user(self.build(), 1, top_n=10)
        self.assertEqual(sorted(row["movie_id"] for row in picked["results"]), [7, 8])
        uncapped = recommend_for_user(self.build(cap_year_at_last_rating=False), 1, top_n=10)
        self.assertIn(10, [row["movie_id"] for row in uncapped["results"]])

    def test_new_user_gets_best_liked_movies_with_constraints(self) -> None:
        # Shrunk means (prior 10 at the catalog mean 3.24): movie 1 3.65, 2 3.52, 5 3.36, 6 and 7 3.29.
        # Movie 9 (one 2.0 rating) stays near the prior at 3.13 and beats movie 4 (five, mean 2.8).
        picked = recommend_for_user(self.build(), 999, top_n=3)
        self.assertTrue(picked["cold_start"])
        self.assertEqual([row["movie_id"] for row in picked["results"]], [1, 2, 5])
        horror = recommend_for_user(self.build(), 999, top_n=10, include_genres=["Horror"], min_year=2004)
        self.assertEqual([row["movie_id"] for row in horror["results"]], [9, 4, 8])

    def test_recommend_for_user_respects_years(self) -> None:
        # User 1 has not rated movies 7, 8 and 10 (released 2007, 2008, 2010).
        picked = recommend_for_user(self.build(), 1, top_n=10, min_year=2008, max_year=2009)
        self.assertEqual([row["movie_id"] for row in picked["results"]], [8])

    def test_peers_are_chosen_among_raters_of_the_movie(self) -> None:
        peers = self.similarity.top_k(1, k=10, min_common=5, among=[3, 5])
        self.assertEqual([peer["user_id"] for peer in peers], [3])

    def test_baseline_decides_without_peer_consensus(self) -> None:
        result = self.build(min_similar_raters=99, baseline_shrinkage=0.0).score(1, 8)
        self.assertEqual((result["verdict"], result["source"]), ("fail", "baseline_prediction"))
        self.assertLessEqual(result["evidence"]["baseline_predicted_rating"], 2.5)

    def test_content_is_evidence_only(self) -> None:
        result = self.build(min_similar_raters=99).score(1, 7)
        self.assertNotEqual(result["source"], "content_score")
        self.assertIn("content_percentile", result["evidence"])

    def test_genre_then_llm_judge_when_other_signals_are_silent(self) -> None:
        silent = {"min_similar_raters": 99, "baseline_shrinkage": 1e9}
        horror = self.build(**silent).score(1, 8)
        self.assertEqual((horror["verdict"], horror["source"]), ("fail", "genre_deviation"))

        drama = self.build(**silent).score(1, 10)
        self.assertEqual((drama["verdict"], drama["source"]), ("unknown", "insufficient_signal"))
        judged = self.build(lambda user, movie: {"verdict": "pass", "reason": "ok"}, **silent).score(1, 10)
        self.assertEqual((judged["verdict"], judged["source"]), ("pass", "llm_judge"))


class FixedClassifier:
    """Stand-in for a fitted GBM: the same probability for every row, and records what it saw."""

    def __init__(self, probability: float) -> None:
        self.probability, self.seen = probability, None

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.seen = x
        return np.tile([1 - self.probability, self.probability], (len(x), 1))


class PeerDrivenClassifier:
    """P(liked) rises with peer_offset, so turning the similar-users signal off moves it."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.clip(0.5 + x["peer_offset"].to_numpy(), 0.0, 1.0)
        return np.column_stack([1 - p, p])


class LearnedScorerTest(TasteTestCase):
    def model(self, p_like: float, p_dislike: float) -> ScoreModel:
        users, movies = np.arange(1, 7), np.array([mid for mid, *_ in MOVIES])
        mlp = MlpRatings(np.full((len(users), len(movies)), 3.5, dtype=np.float32), users, movies)
        return ScoreModel(FixedClassifier(p_like), FixedClassifier(p_dislike), 0.7, 0.7, STACK_FEATURES, mlp,
                          {"neutral": dict.fromkeys(STACK_FEATURES, 0.0)})

    def build(self, model: ScoreModel, llm_judge=None) -> CandidateScorer:
        return CandidateScorer(self.dataset, self.similarity, self.tfidf, self.config, llm_judge, model)

    def test_learned_model_decides_with_its_features_as_evidence(self) -> None:
        model = self.model(p_like=0.9, p_dislike=0.1)
        result = self.build(model).score(1, 7)
        self.assertEqual((result["verdict"], result["source"]), ("pass", "learned_model"))
        self.assertEqual(list(model.like_model.seen.columns), STACK_FEATURES)
        self.assertEqual((result["evidence"]["model"]["p_liked"], result["evidence"]["model"]["mlp_predicted_rating"]),
                         (0.9, 3.5))
        self.assertIn("peer_predicted_rating", result["evidence"])  # rule numbers stay quotable
        # A constant model depends on no signal, so there is nothing to give as a reason.
        self.assertEqual(result["evidence"]["reasons"], [])
        self.assertEqual(result["evidence"]["confidence"]["level"], "high")

        failed = self.build(self.model(p_like=0.1, p_dislike=0.8)).score(1, 8)
        self.assertEqual((failed["verdict"], failed["source"]), ("fail", "learned_model"))

    def test_reasons_are_the_signals_the_verdict_depends_on(self) -> None:
        model = self.model(p_like=0.9, p_dislike=0.1)
        model.like_model = PeerDrivenClassifier()
        result = self.build(model).score(1, 7)
        self.assertEqual(result["verdict"], "pass")
        reasons = result["evidence"]["reasons"]
        self.assertEqual([(r["signal"], r["direction"]) for r in reasons], [("similar_users", "for")])
        self.assertIn("3 people with taste like yours", reasons[0]["text"])
        self.assertIsNone(result["evidence"]["caveat"])

    def test_agent_payload_hides_model_internals(self) -> None:
        from trusted_ai.tools.factory import INTERNAL_KEYS, _evidence_lines, _trim
        result = self.build(self.model(p_like=0.9, p_dislike=0.1)).score(1, 7)
        trimmed = _trim({"results": [{"title": "X", "score": 0.9, "components": {}, "assessment": result}]})
        leaked = INTERNAL_KEYS & set(_all_keys(trimmed))
        self.assertEqual(leaked, set())
        self.assertIn("reasons", trimmed["results"][0]["assessment"]["evidence"])
        self.assertLessEqual(len(trimmed["results"][0]["assessment"]["evidence"].get("peer_ratings", [])), 5)
        lines = _evidence_lines(result)
        self.assertTrue(lines[0].startswith("Verdict: pass. About 9 in 10"))
        self.assertFalse(any("learned_model" in line or "0.9" in line for line in lines))

    def test_agent_sees_only_movie_facts_and_peer_ratings(self) -> None:
        from trusted_ai.tools.factory import _for_agent
        row = {"movie_id": 7, "title": "Heist Three", "year": 2007, "genres": ["Crime"], "score": 0.9,
               "components": {}, "verified": True, "assessment": self.build(self.model(0.9, 0.1)).score(1, 7)}
        movie = _for_agent(row, self.dataset)
        self.assertEqual(set(movie), {"movie_id", "title", "year", "genres", "plot", "mean_rating", "similar_users"})
        self.assertEqual((movie["plot"], movie["mean_rating"]), ("robbery of a vault, a heist gone wrong", 3.4))
        self.assertIn("people most like you who've seen it rated it", movie["similar_users"])

    def test_unsure_model_goes_to_llm_judge_not_to_the_rules(self) -> None:
        unsure = self.model(p_like=0.5, p_dislike=0.5)
        self.assertEqual(self.build(unsure).score(1, 7)["source"], "insufficient_signal")
        judged = self.build(unsure, lambda user, movie: {"verdict": "fail", "reason": "no"}).score(1, 7)
        self.assertEqual((judged["verdict"], judged["source"]), ("fail", "llm_judge"))

    def test_users_unknown_to_train_use_the_rule_cascade(self) -> None:
        result = self.build(self.model(p_like=0.9, p_dislike=0.1)).score(99, 7)
        self.assertNotEqual(result["source"], "learned_model")

    def test_artifacts_round_trip(self) -> None:
        directory = Path(self._tmp.name) / "score_model"
        self.assertIsNone(ScoreModel.load(directory))
        self.model(0.9, 0.1).save(directory)
        loaded = ScoreModel.load(directory)
        self.assertEqual((loaded.like_cut, loaded.features), (0.7, STACK_FEATURES))
        self.assertTrue(np.isnan(loaded.mlp.get(np.array([99]), np.array([7]))[0]))
        self.assertEqual(loaded.mlp.get(np.array([1]), np.array([7]))[0], 3.5)


class SelectByVerdictTest(unittest.TestCase):
    def rows(self) -> list[dict]:
        verdicts = ["unknown", "fail", "pass", "unknown", "pass"]
        return [{"movie_id": i, "assessment": {"verdict": v}} for i, v in enumerate(verdicts)]

    def test_pass_first_drops_fails_and_keeps_ranking_inside_groups(self) -> None:
        picked = select_by_verdict(self.rows(), 4, "pass_first")
        self.assertEqual([row["movie_id"] for row in picked], [2, 4, 0, 3])
        self.assertEqual([row["verified"] for row in picked], [True, True, False, False])

    def test_pass_only_and_off(self) -> None:
        self.assertEqual([row["movie_id"] for row in select_by_verdict(self.rows(), 4, "pass_only")], [2, 4])
        self.assertEqual([row["movie_id"] for row in select_by_verdict(self.rows(), 3, "off")], [0, 1, 2])
        with self.assertRaises(ValueError):
            select_by_verdict(self.rows(), 3, "strict")


class DescriptionSearchTest(TasteTestCase):
    def embedding_search(self, handler) -> tuple[DescriptionSearch, VectorCache]:
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(len(MOVIES), 4)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        movies = MovieEmbeddings(vectors, np.array([mid for mid, *_ in MOVIES]), {"model": "fake"})
        space = EmbeddingSpace(self.dataset, movies, None, self.config)
        client = EmbeddingClient(EmbeddingSettings(base_url="http://embed.test/v1", model="fake"),
                                 transport=httpx.MockTransport(handler), sleep=lambda _: None)
        cache = VectorCache(Path(self._tmp.name) / "queries", namespace="fake")
        return DescriptionSearch(self.tfidf, self.config, space, QueryEncoder(client, cache)), cache

    def test_falls_back_to_tfidf_when_server_is_down(self) -> None:
        search, _ = self.embedding_search(lambda request: httpx.Response(503))
        result = search.search(1, "haunted ghost terror", top_n=3)
        self.assertEqual((result["backend"], result["embedding_fallback"]), ("tfidf", True))
        self.assertEqual(result["results"][0]["movie_id"], 8)

    def test_embeds_query_once_and_filters_seen_and_excluded(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]})

        search, _ = self.embedding_search(handler)
        first = search.search(1, "anything", top_n=10, exclude_genres=["horror"])
        search.search(1, "anything", top_n=10)
        self.assertEqual(len(calls), 1)
        self.assertEqual((first["backend"], first["embedding_fallback"]), ("embedding", False))
        returned = {row["movie_id"] for row in first["results"]}
        self.assertEqual(returned, {7, 10})  # unseen and not Horror


if __name__ == "__main__":
    unittest.main()
