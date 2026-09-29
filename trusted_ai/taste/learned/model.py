"""The learned scorer (gbm_stack): two GBM classifiers, P(liked) and P(disliked), over STACK_FEATURES.

Trained offline by scripts.train_score_model. `mlp_pred` is read from a precomputed
(train user × catalog movie) matrix of MLP ratings, so serving needs neither torch nor the MLP.
The artifacts are pickled scikit-learn models: load them only from a directory you wrote.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GBM_FILE, MLP_FILE, META_FILE = "gbm.pkl", "mlp_pred.npz", "meta.json"


@dataclass
class MlpRatings:
    """MLP rating for every (train user, movie) pair; NaN outside the matrix."""

    pred: np.ndarray        # (n_users, n_movies) float32
    user_ids: np.ndarray
    movie_ids: np.ndarray

    def __post_init__(self) -> None:
        self._users = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        self._movies = pd.Series(np.arange(len(self.movie_ids)), index=self.movie_ids)

    def get(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        rows = self._users.reindex(user_ids).to_numpy()
        cols = self._movies.reindex(movie_ids).to_numpy()
        known = ~(np.isnan(rows) | np.isnan(cols))
        out = np.full(len(rows), np.nan)
        out[known] = self.pred[rows[known].astype(int), cols[known].astype(int)]
        return out

    def save(self, path: Path) -> None:
        np.savez_compressed(path, pred=self.pred, user_ids=self.user_ids, movie_ids=self.movie_ids)

    @classmethod
    def load(cls, path: Path) -> "MlpRatings":
        with np.load(path) as arrays:
            return cls(arrays["pred"], arrays["user_ids"], arrays["movie_ids"])


@dataclass
class ScoreModel:
    like_model: Any
    dislike_model: Any
    like_cut: float         # pass when P(liked) >= like_cut
    dislike_cut: float      # fail when not pass and P(disliked) >= dislike_cut
    features: list[str]
    mlp: MlpRatings
    meta: dict[str, Any] = field(default_factory=dict)

    def probabilities(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = rows[self.features]
        return self.like_model.predict_proba(x)[:, 1], self.dislike_model.predict_proba(x)[:, 1]

    def verdicts(self, p_like: np.ndarray, p_dislike: np.ndarray) -> np.ndarray:
        passed = p_like >= self.like_cut
        failed = ~passed & (p_dislike >= self.dislike_cut)
        return np.where(passed, "pass", np.where(failed, "fail", "none"))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / GBM_FILE).write_bytes(pickle.dumps({"like": self.like_model, "dislike": self.dislike_model}))
        self.mlp.save(directory / MLP_FILE)
        meta = {**self.meta, "features": self.features, "like_cut": self.like_cut, "dislike_cut": self.dislike_cut}
        (directory / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "ScoreModel | None":
        if not all((directory / name).exists() for name in (GBM_FILE, MLP_FILE, META_FILE)):
            return None
        models = pickle.loads((directory / GBM_FILE).read_bytes())
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return cls(models["like"], models["dislike"], float(meta.pop("like_cut")), float(meta.pop("dislike_cut")),
                   list(meta.pop("features")), MlpRatings.load(directory / MLP_FILE), meta)
