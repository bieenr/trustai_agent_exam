"""Features of the learned scorer, all computed from the train split.

The rule signals of CandidateScorer fill `evidence`; FeatureBuilder turns it into the rule
features (the 13 of the original experiment) and adds item-CF: how the user rated the K train
movies most similar to the target by co-rating. `mlp_pred` comes from ScoreModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trusted_ai.taste.genre_profile import genre_preferences
from trusted_ai.taste.learned.item_cf import item_similarity, weighted_offset

if TYPE_CHECKING:
    from trusted_ai.taste.scoring import CandidateScorer


RULE_FEATURES = {
    "peer": ["peer_offset", "log_n_raters", "has_peer"],
    "baseline": ["user_mean", "movie_bias", "log_movie_n"],
    "user": ["user_std", "log_user_n"],
    "genre": ["genre_dev_mean", "genre_dev_min", "avoided_hit"],
    "content": ["content_percentile", "content_pos_minus_neg"],
}
ALL_RULE = [name for group in RULE_FEATURES.values() for name in group]
UNUSED = ["has_peer", "avoided_hit", "content_percentile"]  # zero permutation importance in the GBM
ITEM_CF = ["item_knn_offset", "item_knn_weight"]
STACK_FEATURES = [name for name in ALL_RULE if name not in UNUSED] + ITEM_CF + ["mlp_pred"]


class FeatureBuilder:
    """Per-pair features; caches what is per user (genre deviations, history) or global (item similarity)."""

    def __init__(self, scorer: CandidateScorer) -> None:
        self.scorer = scorer
        self.dataset = scorer.dataset
        self._user_stats = self.dataset.ratings.groupby("userId")["rating"].agg(["std", "size"])
        self._genres: dict[int, dict[str, dict]] = {}
        self._history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._item_sim: np.ndarray | None = None

    @property
    def item_sim(self) -> np.ndarray:
        if self._item_sim is None:
            self._item_sim = item_similarity(self.dataset)
        return self._item_sim

    def rule_features(self, user_id: int, movie: dict[str, Any], evidence: dict[str, Any]) -> dict[str, float]:
        """The 13 features of scripts.experiment_score_model, from the evidence of the rule signals."""
        devs = list(self.genre_deviations(user_id, movie).values())
        user_mean = evidence["user_mean"]
        peer = evidence.get("peer_predicted_rating")
        return {
            "peer_offset": 0.0 if peer is None else peer - user_mean,
            "log_n_raters": np.log1p(evidence.get("n_similar_raters", 0)),
            "has_peer": float(peer is not None),
            "user_mean": user_mean,
            "movie_bias": evidence["movie_bias"],
            "log_movie_n": np.log1p(evidence["movie_n_ratings"]),
            "user_std": float(self._user_stats.at[user_id, "std"]),
            "log_user_n": np.log1p(self._user_stats.at[user_id, "size"]),
            "genre_dev_mean": float(np.mean(devs)) if devs else 0.0,
            "genre_dev_min": float(np.min(devs)) if devs else 0.0,
            "avoided_hit": float(bool(evidence.get("avoided_genres_matched"))),
            "content_percentile": evidence.get("content_percentile", 50.0),
            "content_pos_minus_neg": self.content_contrast(user_id, movie["movie_id"]),
        }

    def genre_deviations(self, user_id: int, movie: dict[str, Any]) -> dict[str, float]:
        """The user's rating deviation (vs their own mean) on each of the movie's genres they have rated."""
        if user_id not in self._genres:
            self._genres[user_id] = {row["genre"]: row for row in genre_preferences(self.dataset, user_id)}
        return {g: self._genres[user_id][g]["deviation"] for g in movie["genres"] if g in self._genres[user_id]}

    def content_contrast(self, user_id: int, movie_id: int) -> float:
        """cos(taste, movie) − cos(negative taste, movie); 0 when either vector is missing."""
        position = self.dataset.movie_position[movie_id]
        positive = self.scorer.content_scores(user_id)
        negative = self.scorer.space.negative_vector(user_id)
        if positive is None or negative is None:
            return 0.0
        return float(positive[position] - self.scorer.space.scores(negative)[position])

    def item_cf_features(self, user_id: int, movie_id: int) -> dict[str, float]:
        """Similarity-weighted (rating − user mean) over the user's K train movies most similar to the target."""
        _, offsets, sims = self._neighbours(user_id, movie_id)
        offset, weight = weighted_offset(sims, offsets)
        return {"item_knn_offset": offset, "item_knn_weight": weight}

    def similar_rated(self, user_id: int, movie_id: int, n: int = 3) -> list[dict[str, Any]]:
        """The n movies the user rated that are most similar to the target by co-rating, with their ratings."""
        positions, offsets, sims = self._neighbours(user_id, movie_id)
        top = [i for i in np.argsort(-sims)[:n] if sims[i] > 0]
        mean = float(self.dataset.user_means[user_id])
        return [{"title": str(self.dataset.movies.at[positions[i], "title"]), "rating": float(offsets[i] + mean)}
                for i in top]

    def _neighbours(self, user_id: int, movie_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Catalog positions, (rating − user mean) and similarity to the target of the user's other rated movies."""
        if user_id not in self._history:
            rated = self.dataset.user_ratings(user_id)
            self._history[user_id] = (rated["movieId"].map(self.dataset.movie_position).to_numpy(),
                                      (rated["rating"] - self.dataset.user_means[user_id]).to_numpy())
        positions, offsets = self._history[user_id]
        target = self.dataset.movie_position[movie_id]
        keep = positions != target
        return positions[keep], offsets[keep], self.item_sim[target, positions[keep]]

    def frame(self, pairs: pd.DataFrame, item_cf: bool = True) -> pd.DataFrame:
        """Rule features (+ item-CF) for every (userId, movieId) row of `pairs`, same index."""
        rows = []
        for user_id, movie_id in zip(pairs["userId"].astype(int), pairs["movieId"].astype(int)):
            movie = self.dataset.movie(movie_id)
            row = self.rule_features(user_id, movie, self.scorer.collect_evidence(user_id, movie))
            if item_cf:
                row.update(self.item_cf_features(user_id, movie_id))
            rows.append(row)
        return pd.DataFrame(rows, index=pairs.index)
