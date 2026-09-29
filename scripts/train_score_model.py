#!/usr/bin/env python3
"""Train the learned scorer of `score_candidate` (gbm_stack) and write its artifacts.

Usage: python -m scripts.train_score_model [--seed 0] [--epochs 11]
1. MLP of scripts/mlp_score (PCA 128, user 64, 256 → 64) trained on ratings_train.csv for a fixed number of epochs,
   never looking at validation, so its validation ratings are out of sample for the GBM. Its
   rating is precomputed for every (train user, catalog movie) pair.
2. Two GBM classifiers, P(liked = rating ≥ liked_rating) and P(disliked = rating ≤ disliked_rating),
   on STACK_FEATURES of the ratings_validation.csv pairs (features from train only). Thresholds
   are picked out-of-fold (GroupKFold by user) for precision pass ≥ 80%, fail ≥ 70%.
Writes TastePaths.score_model (data/score_model/), which TasteService loads when present.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import GroupKFold, cross_val_predict

from scripts.experiment_score_model import TARGET_FAIL_PRECISION, TARGET_PASS_PRECISION, models, threshold_for
from scripts.mlp_score.data import ExperimentData, load
from scripts.mlp_score.model import ScoreMLP
from scripts.mlp_score.train import TrainSettings, fit, predict
from trusted_ai.taste import TasteService
from trusted_ai.taste.learned import STACK_FEATURES, MlpRatings, ScoreModel


PCA_DIM, CF_DIM, HIDDEN, DROPOUT = 128, 64, (256, 64), 0.3


@torch.no_grad()
def full_predictions(model: ScoreMLP, data: ExperimentData) -> np.ndarray:
    """MLP rating for every (train user, movie): user vectors once, then one forward pass per user."""
    model.eval()
    history, biases = data.histories, data.biases
    center = (biases.mu + biases.b_u).astype(np.float32)
    b_i = biases.b_i.astype(np.float32)
    weights = [(ratings - center[u]) / np.sqrt(max(len(ratings), 1)) for u, ratings in enumerate(history.ratings)]
    bag = (torch.from_numpy(np.concatenate(history.movies)).long(),
           torch.tensor(np.cumsum([0] + [len(m) for m in history.movies[:-1]])),
           torch.from_numpy(np.concatenate(weights).astype(np.float32)))
    user_cf = model.cf_bag(*bag[:2], per_sample_weights=bag[2])
    user_content = model.content_bag(*bag[:2], per_sample_weights=bag[2])
    movies = model.movie.weight
    out = np.empty((len(history.movies), len(movies)), dtype=np.float32)
    for u in range(len(history.movies)):
        x = torch.cat([movies, user_cf[u].expand(len(movies), -1), user_content[u].expand(len(movies), -1)], dim=1)
        out[u] = np.clip(center[u] + b_i + model.mlp(x).squeeze(1).numpy(), 0.5, 5.0)
    return out


def train_mlp(service: TasteService, seed: int, epochs: int) -> MlpRatings:
    torch.manual_seed(seed)
    data = load(service.config, PCA_DIM, with_genres=False)
    model = ScoreMLP(data.movie_features, CF_DIM, HIDDEN, DROPOUT)
    fit(model, data, TrainSettings(seed=seed, epochs=epochs, early_stop=False))
    matrix = full_predictions(model, data)
    # The matrix must give the same ratings as the per-pair predict() used in the experiments.
    gap = np.abs(matrix[data.test.user, data.test.movie] - predict(model, data, data.test)).max()
    if gap > 1e-4:
        raise SystemExit(f"MLP matrix differs from predict() by {gap:.2e}")
    return MlpRatings(matrix, data.user_ids, data.movie_ids)


def stack_frame(service: TasteService, pairs: pd.DataFrame, mlp: MlpRatings) -> pd.DataFrame:
    """STACK_FEATURES for every (userId, movieId) row of `pairs`."""
    x = service.scorer.features.frame(pairs)
    x["mlp_pred"] = mlp.get(pairs["userId"].to_numpy(), pairs["movieId"].to_numpy())
    return x[STACK_FEATURES]


def fit_stack(x: pd.DataFrame, liked: np.ndarray, disliked: np.ndarray, groups: np.ndarray) -> dict:
    """Both classifiers on all of `x`, their out-of-fold probabilities and the thresholds picked on those."""
    out = {}
    for name, y, goal in (("like", liked, TARGET_PASS_PRECISION), ("dislike", disliked, TARGET_FAIL_PRECISION)):
        oof = cross_val_predict(models()["gbm"], x, y, groups=groups, cv=GroupKFold(n_splits=5),
                                method="predict_proba")[:, 1]
        out[name] = {"model": models()["gbm"].fit(x, y), "oof": oof, "cut": threshold_for(oof, y, goal)}
    return out


def validation_pairs(service: TasteService) -> pd.DataFrame:
    frame = pd.read_csv(service.config.paths.train_ratings.parent / "ratings_validation.csv")
    return frame[frame["userId"].isin(service.dataset.user_means.index)].reset_index(drop=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="MLP seed")
    parser.add_argument("--epochs", type=int, default=11, help="fixed MLP epochs (no early stopping)")
    args = parser.parse_args()

    service = TasteService(use_score_model=False)
    config = service.config
    mlp = train_mlp(service, args.seed, args.epochs)
    validation = validation_pairs(service)
    x = stack_frame(service, validation, mlp)
    liked = (validation["rating"] >= config.liked_rating).to_numpy()
    disliked = (validation["rating"] <= config.disliked_rating).to_numpy()
    fitted = fit_stack(x, liked, disliked, validation["userId"].to_numpy())

    model = ScoreModel(fitted["like"]["model"], fitted["dislike"]["model"], fitted["like"]["cut"],
                       fitted["dislike"]["cut"], STACK_FEATURES, mlp, meta={
                           "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "sklearn_version": sklearn.__version__,
                           "fit_pairs": len(validation), "liked_rating": config.liked_rating,
                           "disliked_rating": config.disliked_rating,
                           # medians of the fitting pairs: the "signal off" value used to explain verdicts
                           "neutral": {name: float(x[name].median()) for name in STACK_FEATURES},
                           "mlp": {"seed": args.seed, "epochs": args.epochs, "pca_dim": PCA_DIM, "cf_dim": CF_DIM,
                                   "hidden": list(HIDDEN), "dropout": DROPOUT},
                       })
    model.save(config.paths.score_model)
    oof_verdicts = model.verdicts(fitted["like"]["oof"], fitted["dislike"]["oof"])
    passed, failed = oof_verdicts == "pass", oof_verdicts == "fail"
    print(f"Saved {config.paths.score_model}: pass if P(liked) ≥ {model.like_cut:.3f}, "
          f"fail if P(disliked) ≥ {model.dislike_cut:.3f}.")
    print(f"Out-of-fold on validation: {passed.sum()} pass ({liked[passed].mean():.1%} liked), "
          f"{failed.sum()} fail ({disliked[failed].mean():.1%} disliked), coverage {(passed | failed).mean():.1%}.")


if __name__ == "__main__":
    main()
