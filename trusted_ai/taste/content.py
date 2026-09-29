from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from trusted_ai.taste.config import TasteConfig
from trusted_ai.taste.dataset import TasteDataset
from trusted_ai.taste.embedding import MovieEmbeddings, UserEmbeddings


def embedding_text(movie: pd.Series | dict[str, Any], max_chars: int | None = None) -> str:
    """Title and genres ahead of the plot anchor the embedding to the right kind of film.

    Truncation only ever drops the end of the plot, never the title or genres.
    """
    genres = ", ".join(movie["genre_list"]) or "Unknown"
    text = f"{movie['title']} ({movie['year']}). Genres: {genres}. {movie['plot']}".strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def tfidf_text(movie: pd.Series | dict[str, Any]) -> str:
    return f"{movie['title']}. {' '.join(movie['genre_list'])}. {movie['plot']}"


def taste_weights(ratings: pd.DataFrame, liked_rating: float) -> tuple[np.ndarray, np.ndarray]:
    """Movies rated >= liked_rating, weighted by how far they beat the user's own mean."""
    liked = ratings[ratings["rating"] >= liked_rating]
    weights = (liked["rating"] - ratings["rating"].mean()).clip(lower=0).to_numpy(dtype=float)
    if len(liked) and weights.sum() == 0:
        weights = np.ones(len(liked))  # a user who rates everything 5.0 still has a taste
    return liked["movieId"].to_numpy(), weights


def negative_weights(ratings: pd.DataFrame, disliked_rating: float) -> tuple[np.ndarray, np.ndarray]:
    disliked = ratings[ratings["rating"] <= disliked_rating]
    weights = (ratings["rating"].mean() - disliked["rating"]).clip(lower=0).to_numpy(dtype=float)
    if len(disliked) and weights.sum() == 0:
        weights = np.ones(len(disliked))
    return disliked["movieId"].to_numpy(), weights


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class ContentSpace:
    """Movie vectors aligned with `dataset.movies` rows; dot product equals cosine similarity."""

    backend = "base"

    def __init__(self, dataset: TasteDataset, matrix: np.ndarray | sparse.csr_matrix,
                 config: TasteConfig) -> None:
        self.dataset = dataset
        self.matrix = matrix
        self.config = config

    def scores(self, vector: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Cosine similarity between one unit vector and every movie."""
        if sparse.issparse(vector):
            vector = vector.toarray().ravel()
        # macOS Accelerate BLAS raises spurious FP warnings on large matmuls; results are exact.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return np.asarray(self.matrix @ vector).ravel()

    def movie_vectors(self, movie_ids: np.ndarray) -> np.ndarray | sparse.csr_matrix:
        return self.matrix[[self.dataset.movie_position[int(mid)] for mid in movie_ids]]

    def weighted_profile(self, movie_ids: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
        if len(movie_ids) == 0:
            return None
        rows = self.movie_vectors(movie_ids)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # see scores()
            profile = np.asarray(rows.T @ weights).ravel() / weights.sum()
        return _unit(profile)

    def taste_vector(self, user_id: int) -> np.ndarray | None:
        ratings = self.dataset.user_ratings(user_id)
        return self.weighted_profile(*taste_weights(ratings, self.config.liked_rating))

    def negative_vector(self, user_id: int) -> np.ndarray | None:
        ratings = self.dataset.user_ratings(user_id)
        return self.weighted_profile(*negative_weights(ratings, self.config.disliked_rating))


class TfidfSpace(ContentSpace):
    """Lexical baseline, and the fallback whenever the embedding server is unreachable."""

    backend = "tfidf"

    def __init__(self, dataset: TasteDataset, config: TasteConfig) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20_000, ngram_range=(1, 2))
        texts = [tfidf_text(row) for _, row in dataset.movies.iterrows()]
        super().__init__(dataset, self.vectorizer.fit_transform(texts).tocsr(), config)

    def encode_query(self, query: str) -> sparse.csr_matrix:
        return self.vectorizer.transform([query])


class EmbeddingSpace(ContentSpace):
    """Precomputed Qwen3 movie embeddings; user taste vectors are loaded or derived locally."""

    backend = "embedding"

    def __init__(self, dataset: TasteDataset, movies: MovieEmbeddings,
                 users: UserEmbeddings | None, config: TasteConfig) -> None:
        dim = movies.vectors.shape[1]
        matrix = np.zeros((len(dataset.movies), dim), dtype=np.float32)
        source = {int(mid): row for row, mid in enumerate(movies.movie_ids)}
        rows = [(pos, source[int(mid)]) for pos, mid in enumerate(dataset.movies["movieId"]) if int(mid) in source]
        matrix[[pos for pos, _ in rows]] = movies.vectors[[row for _, row in rows]]
        self.missing_movies = len(dataset.movies) - len(rows)
        self.meta = movies.meta
        self._users = users
        self._user_position = {} if users is None else {int(uid): i for i, uid in enumerate(users.user_ids)}
        super().__init__(dataset, matrix, config)

    def taste_vector(self, user_id: int) -> np.ndarray | None:
        position = self._user_position.get(int(user_id))
        if position is not None and np.any(self._users.taste[position]):
            return self._users.taste[position]
        return super().taste_vector(user_id)

    def negative_vector(self, user_id: int) -> np.ndarray | None:
        position = self._user_position.get(int(user_id))
        if position is not None and np.any(self._users.negative[position]):
            return self._users.negative[position]
        return super().negative_vector(user_id)


def compute_user_embeddings(space: ContentSpace) -> UserEmbeddings:
    """Taste and negative-taste vectors for every train user (zeros when undefined)."""
    user_ids = np.array(space.dataset.user_ids)
    dim = space.matrix.shape[1]
    taste = np.zeros((len(user_ids), dim), dtype=np.float32)
    negative = np.zeros_like(taste)
    for row, user_id in enumerate(user_ids):
        ratings = space.dataset.user_ratings(int(user_id))
        for target, (ids, weights) in (
            (taste, taste_weights(ratings, space.config.liked_rating)),
            (negative, negative_weights(ratings, space.config.disliked_rating)),
        ):
            vector = space.weighted_profile(ids, weights)
            if vector is not None:
                target[row] = vector
    return UserEmbeddings(taste, negative, user_ids)
