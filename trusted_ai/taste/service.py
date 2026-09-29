from __future__ import annotations

import json
import logging
from typing import Any

from trusted_ai.taste.collaborative import (
    UserSimilarity, compute_user_similarity, peer_ratings, weighted_peer_average,
)
from trusted_ai.taste.config import QUERY_INSTRUCTION, EmbeddingSettings, TasteConfig
from trusted_ai.taste.content import EmbeddingSpace, TfidfSpace
from trusted_ai.taste.dataset import TasteDataset, load_dataset
from trusted_ai.taste.embedding import (
    EmbeddingClient, VectorCache, load_movie_embeddings, load_user_embeddings,
)
from trusted_ai.taste.genre_profile import blind_spots, catalog_genre_share
from trusted_ai.taste.learned import ScoreModel
from trusted_ai.taste.profile import build_profile
from trusted_ai.taste.recommend import explain_match, recommend_for_user, select_by_verdict
from trusted_ai.taste.scoring import CandidateScorer, LLMJudgeFn
from trusted_ai.taste.search import DescriptionSearch, QueryEncoder
from trusted_ai.taste.summary import template_summary


logger = logging.getLogger(__name__)


class TasteService:
    """Runtime entry point: loads precomputed artifacts and answers tool calls.

    Nothing here calls the embedding API except `search_by_description`, which encodes
    the user's query (once per distinct query, thanks to the on-disk cache).
    """

    def __init__(self, config: TasteConfig | None = None, *, dataset: TasteDataset | None = None,
                 embedding_client: EmbeddingClient | None = None,
                 llm_judge: LLMJudgeFn | None = None, use_score_model: bool = True) -> None:
        self.config = config or TasteConfig()
        paths = self.config.paths
        self.dataset = dataset or load_dataset(paths)
        self.similarity = (UserSimilarity.load(paths.user_similarity, paths.user_similarity_ids)
                           or compute_user_similarity(self.dataset))
        self.tfidf = TfidfSpace(self.dataset, self.config)
        self.embedding = self._load_embedding_space()
        encoder = self._build_encoder(embedding_client)
        primary = self.embedding or self.tfidf
        model = ScoreModel.load(paths.score_model) if use_score_model else None
        if use_score_model and model is None:
            logger.info("No learned scorer in %s; score_candidate uses the rule cascade", paths.score_model)
        self.scorer = CandidateScorer(self.dataset, self.similarity, primary, self.config, llm_judge, model)
        self.search = DescriptionSearch(self.tfidf, self.config, self.embedding, encoder)
        self._catalog_share = catalog_genre_share(self.dataset.movies)
        self._profiles = self._load_profiles()

    def get_user_profile(self, user_id: int) -> dict[str, Any]:
        if user_id in self._profiles:
            return self._profiles[user_id]
        if not self.dataset.has_user(user_id):
            return {"user_id": user_id, "rating_count": 0, "cold_start": True,
                    "note": "New user with no ratings: there is no personal taste, similar users or history yet."}
        profile = build_profile(self.dataset, self.similarity, self.config, user_id, self._catalog_share)
        profile["taste_summary"] = template_summary(profile)
        return profile

    def get_similar_users(self, user_id: int, k: int = 10) -> list[dict[str, Any]]:
        return self.scorer.similar_users(user_id, k)

    def get_peer_opinion(self, user_id: int, movie_title: str, k: int | None = None) -> dict[str, Any]:
        matches = self.dataset.find_movies(movie_title)
        if not matches:
            return {"found": False, "query": movie_title}
        movie = matches[0]
        peers = self.scorer.similar_users(user_id, k)
        rows = peer_ratings(self.dataset, peers, movie["movie_id"])
        own = self.dataset.user_ratings(user_id)
        own_rating = own.loc[own["movieId"] == movie["movie_id"], "rating"]
        weighted = weighted_peer_average(rows)
        return {
            "found": True,
            "movie": {key: movie[key] for key in ("movie_id", "title", "year", "genres")},
            "other_title_matches": [match["title"] for match in matches[1:]],
            "n_similar_users_checked": len(peers),
            "n_similar_raters": len(rows),
            "weighted_avg": None if weighted is None else round(weighted, 3),
            "simple_avg": None if not rows else round(sum(r["rating"] for r in rows) / len(rows), 3),
            "ratings": rows,
            "your_rating": None if own_rating.empty else float(own_rating.iloc[0]),
            "assessment": self.scorer.score(user_id, movie["movie_id"]),
        }

    def search_by_description(self, user_id: int, query: str, top_n: int = 10, **filters: Any) -> dict[str, Any]:
        """Ranked matches, each with its score_candidate assessment, filtered by config.verdict_filter."""
        result = self.search.search(user_id, query, top_n=self._pool(top_n), **filters)
        for row in result["results"]:
            row["assessment"] = self.scorer.score(user_id, row["movie_id"])
        result["results"] = select_by_verdict(result["results"], top_n, self.config.verdict_filter)
        return result

    def recommend_for_user(self, user_id: int, top_n: int = 10, **options: Any) -> dict[str, Any]:
        result = recommend_for_user(self.scorer, user_id, top_n=self._pool(top_n), **options)
        result["results"] = select_by_verdict(result["results"], top_n, self.config.verdict_filter)
        return result

    def explain_match(self, user_id: int, movie_id: int) -> dict[str, Any]:
        return explain_match(self.scorer, user_id, movie_id)

    def get_blind_spots(self, user_id: int) -> list[dict[str, Any]]:
        return blind_spots(self.dataset, user_id, self._catalog_share)

    def score_candidate(self, user_id: int, movie_id: int) -> dict[str, Any]:
        return self.scorer.score(user_id, movie_id)

    def _pool(self, top_n: int) -> int:
        """Candidates to rank before the verdict filter; the plain top_n when filtering is off."""
        return top_n if self.config.verdict_filter == "off" else max(top_n, self.config.candidate_pool)

    def _load_embedding_space(self) -> EmbeddingSpace | None:
        directory = self.config.paths.embeddings_dir
        movies = load_movie_embeddings(directory)
        if movies is None:
            logger.info("No movie embeddings in %s; content signals use TF-IDF", directory)
            return None
        return EmbeddingSpace(self.dataset, movies, load_user_embeddings(directory), self.config)

    def _build_encoder(self, client: EmbeddingClient | None) -> QueryEncoder | None:
        if self.embedding is None:
            return None
        if client is None:
            settings = EmbeddingSettings.from_env()
            if settings is None:
                logger.info("EMBEDDING_BASE_URL is not set; description search uses TF-IDF")
                return None
            client = EmbeddingClient(settings)
        model = client.settings.model
        built_with = self.embedding.meta.get("model")
        if built_with and built_with.casefold() != model.casefold():
            # Query and movie vectors from different models are not comparable.
            logger.warning("Movie embeddings built with %s but server model is %s; using TF-IDF",
                           built_with, model)
            return None
        cache = VectorCache(self.config.paths.embedding_cache_dir / "queries",
                            namespace=f"{model}|{QUERY_INSTRUCTION}")
        return QueryEncoder(client, cache)

    def _load_profiles(self) -> dict[int, dict[str, Any]]:
        path = self.config.paths.profiles
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {int(profile["user_id"]): profile for profile in payload["profiles"]}
