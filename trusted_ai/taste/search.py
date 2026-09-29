from __future__ import annotations

import logging
from typing import Any

import numpy as np

from trusted_ai.taste.config import QUERY_INSTRUCTION, TasteConfig
from trusted_ai.taste.content import ContentSpace, EmbeddingSpace, TfidfSpace
from trusted_ai.taste.embedding import EmbeddingClient, EmbeddingError, VectorCache
from trusted_ai.taste.filters import candidate_mask


logger = logging.getLogger(__name__)


class QueryEncoder:
    """Encodes a query through the API once; repeated queries are served from disk."""

    def __init__(self, client: EmbeddingClient, cache: VectorCache | None,
                 instruction: str = QUERY_INSTRUCTION) -> None:
        self.client = client
        self.cache = cache
        self.instruction = instruction

    def encode(self, query: str) -> np.ndarray:
        cached = self.cache.get(query) if self.cache else None
        if cached is not None:
            return cached
        vector = self.client.embed_texts([query], instruction=self.instruction)[0]
        if self.cache:
            self.cache.put(query, vector)
        return vector


class DescriptionSearch:
    def __init__(self, tfidf: TfidfSpace, config: TasteConfig,
                 embedding: EmbeddingSpace | None = None, encoder: QueryEncoder | None = None) -> None:
        self.tfidf = tfidf
        self.embedding = embedding
        self.encoder = encoder
        self.config = config

    def search(
        self,
        user_id: int,
        query: str,
        *,
        top_n: int = 10,
        exclude_genres: list[str] | None = None,
        include_genres: list[str] | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        backend: str = "auto",
    ) -> dict[str, Any]:
        """Rank unseen movies by 0.6·cos(query, movie) + 0.4·cos(taste, movie).

        backend="auto" uses embeddings when available and falls back to TF-IDF on any
        API failure; "tfidf" forces the baseline (used for the report comparison).
        """
        space, query_vector, fallback_reason = self._encode(query, backend)
        query_scores = space.scores(query_vector)
        taste = space.taste_vector(user_id)
        taste_scores = space.scores(taste) if taste is not None else np.zeros_like(query_scores)
        if taste is None:  # no liked movies in train: pure query match
            final = query_scores.copy()
        else:
            final = self.config.query_weight * query_scores + self.config.taste_weight * taste_scores
        negative = space.negative_vector(user_id)
        if negative is not None and self.config.negative_weight:
            final -= self.config.negative_weight * space.scores(negative)

        mask = candidate_mask(space.dataset, user_id, exclude_genres=exclude_genres,
                              include_genres=include_genres, min_year=min_year, max_year=max_year,
                              until_last_rating=self.config.cap_year_at_last_rating)
        final = np.where(mask, final, -np.inf)
        order = [pos for pos in np.argsort(-final)[:top_n] if np.isfinite(final[pos])]
        movies = space.dataset.movies
        return {
            "query": query,
            "backend": space.backend,
            "embedding_fallback": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "used_taste_vector": taste is not None,
            "results": [
                {
                    "movie_id": int(movies.at[pos, "movieId"]),
                    "title": str(movies.at[pos, "title"]),
                    "year": int(movies.at[pos, "year"]),
                    "genres": list(movies.at[pos, "genre_list"]),
                    "score": round(float(final[pos]), 4),
                    "query_similarity": round(float(query_scores[pos]), 4),
                    "taste_similarity": round(float(taste_scores[pos]), 4),
                }
                for pos in order
            ],
        }

    def _encode(self, query: str, backend: str) -> tuple[ContentSpace, Any, str | None]:
        if backend not in {"auto", "embedding", "tfidf"}:
            raise ValueError(f"Unknown search backend: {backend}")
        if backend == "tfidf":
            return self.tfidf, self.tfidf.encode_query(query), None
        if self.embedding is None or self.encoder is None:
            reason = "embedding artifacts or server not configured"
        else:
            try:
                return self.embedding, self.encoder.encode(query), None
            except EmbeddingError as exc:
                reason = str(exc)
                logger.warning("embedding_fallback=True: %s", reason)
        if backend == "embedding":
            raise EmbeddingError(reason)
        return self.tfidf, self.tfidf.encode_query(query), reason
