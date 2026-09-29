from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trusted_ai.taste.dataset import TasteDataset


@dataclass
class UserSimilarity:
    """Centered-cosine similarity for every user pair, computed once for the whole train split."""

    matrix: np.ndarray  # (n_users, n_users)
    common: np.ndarray  # number of co-rated movies per pair
    user_ids: np.ndarray

    def __post_init__(self) -> None:
        self.position = {int(uid): pos for pos, uid in enumerate(self.user_ids)}

    def top_k(self, user_id: int, k: int, min_common: int,
              among: Iterable[int] | None = None) -> list[dict[str, Any]]:
        """Most similar users; `among` restricts the pool (e.g. to users who rated one movie)."""
        position = self.position.get(int(user_id))
        if position is None:
            return []
        scores = self.matrix[position].copy()
        if among is not None:
            allowed = np.zeros(len(scores), dtype=bool)
            allowed[[self.position[int(uid)] for uid in among if int(uid) in self.position]] = True
            scores[~allowed] = -np.inf
        scores[position] = -np.inf
        # Similarity from two or three shared movies is noise, whatever its value.
        scores[self.common[position] < min_common] = -np.inf
        # Anti-correlated users are not "similar"; they would dilute peer averages.
        order = [index for index in np.argsort(-scores) if np.isfinite(scores[index]) and scores[index] > 0][:k]
        return [
            {"user_id": int(self.user_ids[index]), "similarity": round(float(scores[index]), 4),
             "n_common_ratings": int(self.common[position, index])}
            for index in order
        ]

    def save(self, matrix_path: Path, ids_path: Path) -> None:
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, np.stack([self.matrix, self.common]).astype(np.float32))
        np.save(ids_path, self.user_ids.astype(np.int64))

    @classmethod
    def load(cls, matrix_path: Path, ids_path: Path) -> "UserSimilarity | None":
        if not matrix_path.exists() or not ids_path.exists():
            return None
        stacked = np.load(matrix_path)
        return cls(stacked[0], stacked[1].astype(np.int32), np.load(ids_path))


def compute_user_similarity(dataset: TasteDataset) -> UserSimilarity:
    user_item = dataset.user_item
    rated = user_item.notna().to_numpy(dtype=np.float32)
    # Centering by each user's mean lets harsh and generous raters be compared.
    centered = user_item.sub(user_item.mean(axis=1), axis=0).fillna(0.0).to_numpy(dtype=np.float64)
    norms = np.linalg.norm(centered, axis=1)
    norms[norms == 0] = 1.0
    normalised = centered / norms[:, None]
    # macOS Accelerate BLAS raises spurious FP warnings on large matmuls; results are exact.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        matrix = normalised @ normalised.T
        common = (rated @ rated.T).astype(np.int32)
    return UserSimilarity(matrix.astype(np.float32), common, user_item.index.to_numpy())


def peer_ratings(dataset: TasteDataset, peers: list[dict[str, Any]], movie_id: int) -> list[dict[str, Any]]:
    """Train-split ratings of one movie by the given similar users."""
    if not peers or movie_id not in dataset.user_item.columns:
        return []
    column = dataset.user_item[movie_id]
    rows = []
    for peer in peers:
        rating = column.get(peer["user_id"])
        if rating is not None and not pd.isna(rating):
            rows.append({**peer, "rating": float(rating)})
    return rows


def weighted_peer_average(rows: list[dict[str, Any]]) -> float | None:
    weights = np.array([max(row["similarity"], 0.0) for row in rows])
    if not rows or weights.sum() == 0:
        return None
    return float(np.dot(weights, [row["rating"] for row in rows]) / weights.sum())


def predict_from_peers(dataset: TasteDataset, user_id: int, peers: list[dict[str, Any]],
                       shrinkage: float) -> pd.DataFrame:
    """Mean-centered neighbourhood prediction for every movie any peer rated.

    Returns movieId, predicted_rating, n_peer_raters. The peer offset is shrunk by
    n / (n + shrinkage) so one enthusiastic neighbour cannot dominate the ranking.
    """
    columns = ["movieId", "predicted_rating", "n_peer_raters"]
    positive = [peer for peer in peers if peer["similarity"] > 0]
    if not positive:
        return pd.DataFrame(columns=columns)
    ids = [peer["user_id"] for peer in positive]
    weights = pd.Series({peer["user_id"]: peer["similarity"] for peer in positive})
    ratings = dataset.user_item.loc[ids]
    offsets = ratings.sub(dataset.user_means.loc[ids], axis=0)
    mask = offsets.notna()
    weight_sum = mask.mul(weights, axis=0).sum()
    weighted = offsets.fillna(0.0).mul(weights, axis=0).sum()
    counts = mask.sum()
    valid = counts > 0
    offset = (weighted[valid] / weight_sum[valid]) * (counts[valid] / (counts[valid] + shrinkage))
    user_mean = float(dataset.user_means.get(user_id, dataset.ratings["rating"].mean()))
    return pd.DataFrame({
        "movieId": offset.index.astype(int),
        "predicted_rating": (user_mean + offset).clip(0.5, 5.0).to_numpy(),
        "n_peer_raters": counts[valid].astype(int).to_numpy(),
    })
