from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from trusted_ai.taste.baseline import item_biases
from trusted_ai.taste.collaborative import (
    UserSimilarity, peer_ratings, predict_from_peers, weighted_peer_average,
)
from trusted_ai.taste.config import TasteConfig
from trusted_ai.taste.content import ContentSpace
from trusted_ai.taste.dataset import TasteDataset
from trusted_ai.taste.genre_profile import avoid_genres, genre_preferences
from trusted_ai.taste.learned import FeatureBuilder, ScoreModel
from trusted_ai.taste.learned.explain import ablation_batch, explain


# (user_id, movie dict) -> {"verdict": "pass" | "fail", "reason": str}
LLMJudgeFn = Callable[[int, dict[str, Any]], dict[str, Any]]


class CandidateScorer:
    """Decides whether a user will like one movie, citing the numbers behind the verdict.

    With a trained ScoreModel and a user known from train, the learned model decides (GBM over
    the rule signals, item-CF and an MLP rating); when it is not confident enough, the LLM judge
    does. Otherwise signals are tried from most to least grounded: a mean-centred prediction
    from similar users who rated the movie, a user-mean + movie-bias baseline, genre deviation,
    then (optionally) an LLM judge. A prediction between the disliked and liked thresholds says
    little, so it falls through to the next signal. Content similarity never decides alone (on
    held-out ratings it predicts what a user watches, not whether they like it) but is kept
    in `evidence`, as is every other number computed, so the agent can quote it.
    """

    def __init__(self, dataset: TasteDataset, similarity: UserSimilarity, space: ContentSpace,
                 config: TasteConfig, llm_judge: LLMJudgeFn | None = None,
                 model: ScoreModel | None = None) -> None:
        self.dataset = dataset
        self.similarity = similarity
        self.space = space
        self.config = config
        self.llm_judge = llm_judge
        self.model = model
        self.features = FeatureBuilder(self)
        self._content_scores: dict[int, np.ndarray | None] = {}
        self._item_biases = item_biases(dataset, config.baseline_shrinkage)

    def similar_users(self, user_id: int, k: int | None = None) -> list[dict[str, Any]]:
        return self.similarity.top_k(user_id, k or self.config.similar_users_k,
                                     self.config.min_common_ratings)

    def content_scores(self, user_id: int) -> np.ndarray | None:
        """cos(taste, movie) for every catalog movie; the user's own distribution sets thresholds."""
        if user_id not in self._content_scores:
            taste = self.space.taste_vector(user_id)
            self._content_scores[user_id] = None if taste is None else self.space.scores(taste)
        return self._content_scores[user_id]

    def score(self, user_id: int, movie_id: int) -> dict[str, Any]:
        movie = self.dataset.movie(movie_id)
        if movie is None:
            return {"movie_id": movie_id, "verdict": "unknown", "source": "unknown_movie", "evidence": {}}
        if self.model is not None and self.dataset.has_user(user_id):
            return self._learned_score(user_id, movie)
        evidence: dict[str, Any] = {}
        self._content_evidence(user_id, movie, evidence)
        for check in (self._peer_signal, self._baseline_signal, self._genre_signal):
            verdict = check(user_id, movie, evidence)
            if verdict is not None:
                return {"movie_id": movie_id, "verdict": verdict[0], "source": verdict[1], "evidence": evidence}
        return self._judge(user_id, movie, evidence)

    def collect_evidence(self, user_id: int, movie: dict[str, Any]) -> dict[str, Any]:
        """Numbers of every rule signal, whatever their verdicts: the inputs of the learned model."""
        evidence: dict[str, Any] = {}
        self._content_evidence(user_id, movie, evidence)
        for signal in (self._peer_signal, self._baseline_signal, self._genre_signal):
            signal(user_id, movie, evidence)
        return evidence

    def _learned_score(self, user_id: int, movie: dict[str, Any]) -> dict[str, Any]:
        movie_id = movie["movie_id"]
        evidence = self.collect_evidence(user_id, movie)
        row = {**self.features.rule_features(user_id, movie, evidence),
               **self.features.item_cf_features(user_id, movie_id),
               "mlp_pred": float(self.model.mlp.get(np.array([user_id]), np.array([movie_id]))[0])}
        # One batch: the pair itself, then one copy per signal group set to neutral (for the reasons).
        p_like, p_dislike = self.model.probabilities(ablation_batch(row, self.model))
        verdict = str(self.model.verdicts(p_like[:1], p_dislike[:1])[0])
        explanation = explain(self.features, user_id, movie, row, evidence, verdict, p_like, p_dislike)
        rounded = lambda value: None if np.isnan(value) else round(float(value), 3)
        evidence.update({
            "reasons": explanation["reasons"], "confidence": explanation["confidence"],
            "caveat": explanation["caveat"],
            # Internals for debugging and evaluation; not meant to be shown to the user.
            "model": {"p_liked": rounded(p_like[0]), "p_disliked": rounded(p_dislike[0]),
                      "thresholds": {"pass_p_liked": self.model.like_cut, "fail_p_disliked": self.model.dislike_cut},
                      "mlp_predicted_rating": rounded(row["mlp_pred"]),
                      "item_knn_offset": rounded(row["item_knn_offset"]),
                      "item_knn_weight": rounded(row["item_knn_weight"]),
                      "signal_effects": explanation["signal_effects"]},
        })
        if verdict != "none":
            return {"movie_id": movie_id, "verdict": verdict, "source": "learned_model", "evidence": evidence}
        return self._judge(user_id, movie, evidence)

    def _judge(self, user_id: int, movie: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        movie_id = movie["movie_id"]
        if self.llm_judge is not None:
            decision = self.llm_judge(user_id, movie)
            evidence["llm_reason"] = decision.get("reason")
            return {"movie_id": movie_id, "verdict": decision["verdict"], "source": "llm_judge",
                    "evidence": evidence}
        return {"movie_id": movie_id, "verdict": "unknown", "source": "insufficient_signal",
                "evidence": evidence}

    def _peer_signal(self, user_id: int, movie: dict, evidence: dict) -> tuple[str, str] | None:
        movie_id = movie["movie_id"]
        raters = (self.dataset.user_item[movie_id].dropna().index
                  if movie_id in self.dataset.user_item.columns else [])
        # Neighbours are chosen among the users who rated this movie, not among the global
        # top-K, otherwise most movies end up with too few similar raters to say anything.
        peers = self.similarity.top_k(user_id, self.config.similar_users_k,
                                      self.config.min_common_ratings, among=raters)
        rows = peer_ratings(self.dataset, peers, movie_id)
        evidence["n_similar_raters"] = len(rows)
        average = weighted_peer_average(rows)
        # Similarities are rounded to 4 decimals, so a sliver-positive peer can weigh 0.
        if average is None:
            return None
        evidence["weighted_avg"] = round(average, 3)
        evidence["peer_ratings"] = [
            {"user_id": row["user_id"], "similarity": row["similarity"], "rating": row["rating"]}
            for row in rows
        ]
        # Peers' offsets from their own means, added to this user's mean: harsh and generous
        # raters become comparable, and the user's own scale is respected.
        predictions = predict_from_peers(self.dataset, user_id, peers, self.config.cf_shrinkage)
        predicted = float(predictions.loc[predictions["movieId"] == movie_id, "predicted_rating"].iloc[0])
        evidence["peer_predicted_rating"] = round(predicted, 3)
        # Fewer than three raters is anecdote, not consensus; defer to weaker signals.
        if len(rows) < self.config.min_similar_raters:
            return None
        label = self._label(predicted)
        return None if label is None else (label, "similar_user_pseudo_label")

    def _baseline_signal(self, user_id: int, movie: dict, evidence: dict) -> tuple[str, str] | None:
        user_mean = self.dataset.user_means.get(user_id)
        if user_mean is None:
            return None
        movie_id = movie["movie_id"]
        bias = float(self._item_biases["bias"].get(movie_id, 0.0))
        predicted = float(user_mean) + bias
        evidence.update({
            "user_mean": round(float(user_mean), 3),
            "movie_bias": round(bias, 3),
            "movie_n_ratings": int(self._item_biases["n_ratings"].get(movie_id, 0)),
            "baseline_predicted_rating": round(predicted, 3),
        })
        label = self._label(predicted)
        return None if label is None else (label, "baseline_prediction")

    def _content_evidence(self, user_id: int, movie: dict, evidence: dict) -> None:
        scores = self.content_scores(user_id)
        if scores is None:
            return
        score = float(scores[self.dataset.movie_position[movie["movie_id"]]])
        # Rank against the movies the user has not rated: already-liked movies would
        # otherwise occupy the top of the distribution and depress every candidate.
        unseen = scores[self._unseen_mask(user_id)]
        percentile = float(((unseen < score).mean() + (unseen <= score).mean()) * 50)
        evidence.update({
            "content_backend": self.space.backend,
            "content_similarity": round(score, 4),
            "content_percentile": round(percentile, 1),
        })

    def _genre_signal(self, user_id: int, movie: dict, evidence: dict) -> tuple[str, str] | None:
        avoided = avoid_genres(genre_preferences(self.dataset, user_id), self.config.avoid_deviation)
        hits = [row for row in avoided if row["genre"] in movie["genres"]]
        evidence["avoided_genres_matched"] = hits
        return ("fail", "genre_deviation") if hits else None

    def _unseen_mask(self, user_id: int) -> np.ndarray:
        seen = set(self.dataset.user_ratings(user_id)["movieId"])
        return ~self.dataset.movies["movieId"].isin(seen).to_numpy()

    def _label(self, rating: float) -> str | None:
        if rating >= self.config.liked_rating:
            return "pass"
        return "fail" if rating <= self.config.disliked_rating else None
