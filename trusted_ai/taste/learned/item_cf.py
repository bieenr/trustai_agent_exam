"""Item-item collaborative filtering on train ratings: adjusted-cosine similarity and the
"how did this user rate the K most similar movies" offset, used by the learned scorer."""

from __future__ import annotations

import numpy as np

from trusted_ai.taste.dataset import TasteDataset


K = 20
ITEM_SHRINKAGE = 10.0


def adjusted_cosine(users: np.ndarray, movies: np.ndarray, ratings: np.ndarray, n_users: int,
                    n_movies: int) -> np.ndarray:
    """(n_movies, n_movies) cosine of user-mean-centred ratings, shrunk by n_common / (n_common + ITEM_SHRINKAGE).

    `users` / `movies` are 0-based row indices; movies nobody rated get similarity 0; the diagonal is 0.
    """
    means = np.bincount(users, ratings, n_users) / np.maximum(np.bincount(users, minlength=n_users), 1)
    values = np.zeros((n_users, n_movies), dtype=np.float32)
    rated = np.zeros((n_users, n_movies), dtype=np.float32)
    values[users, movies] = ratings - means[users]
    rated[users, movies] = 1.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious Accelerate BLAS warnings
        dot = values.T @ values
        common = rated.T @ rated
    norms = np.sqrt(np.diag(dot)).copy()
    norms[norms == 0] = 1.0
    sim = dot / np.outer(norms, norms) * (common / (common + ITEM_SHRINKAGE))
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def weighted_offset(sims: np.ndarray, offsets: np.ndarray) -> tuple[float, float]:
    """Similarity-weighted mean offset over the top-K positive neighbours, and their weight sum."""
    top = np.argsort(-sims)[:K]
    top = top[sims[top] > 0]
    if not len(top):
        return float("nan"), 0.0
    weights = sims[top]
    return float(np.dot(weights, offsets[top]) / weights.sum()), float(weights.sum())


def item_similarity(dataset: TasteDataset) -> np.ndarray:
    """Adjusted cosine between movies on the dataset's ratings, indexed by catalog position."""
    ratings = dataset.ratings
    users = {int(uid): index for index, uid in enumerate(dataset.user_means.index)}
    return adjusted_cosine(ratings["userId"].map(users).to_numpy(),
                           ratings["movieId"].map(dataset.movie_position).to_numpy(),
                           ratings["rating"].to_numpy(np.float32), len(users), len(dataset.movies))
