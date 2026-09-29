"""Training loop with masked targets, denoising and early stopping on validation RMSE."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .data import ExperimentData, Pairs, user_bags
from .model import ScoreMLP


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    drop_prob: float = 0.2
    seed: int = 0
    early_stop: bool = True  # False: keep the last epoch, validation is never looked at


def fit(model: ScoreMLP, data: ExperimentData, settings: TrainSettings) -> list[dict[str, float]]:
    """Train on masked train pairs; keep the weights of the best validation epoch (or the last one)."""
    rng = np.random.default_rng(settings.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    history, best, best_state, stale = [], np.inf, None, 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses = []
        for rows in np.array_split(rng.permutation(len(data.train)),
                                   max(1, len(data.train) // settings.batch_size)):
            residual = torch.from_numpy(data.train.rating[rows] - data.train.baseline[rows])
            output = model(torch.from_numpy(data.train.movie[rows]).long(),
                           *user_bags(data.train, rows, data.histories, settings.drop_prob, rng))
            loss = torch.mean((output - residual) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if not settings.early_stop:
            history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_rmse": float("nan")})
            print(f"epoch {epoch:3d}  train mse {history[-1]['train_mse']:.4f}")
            continue
        val_rmse = rmse(predict(model, data, data.validation), data.validation.rating)
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_rmse": val_rmse})
        print(f"epoch {epoch:3d}  train mse {history[-1]['train_mse']:.4f}  val rmse {val_rmse:.4f}")
        if val_rmse < best:
            best, best_state, stale = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model: ScoreMLP, data: ExperimentData, pairs: Pairs, batch_size: int = 1024) -> np.ndarray:
    """Predicted rating (baseline + residual), clipped to the MovieLens scale."""
    model.eval()
    rng = np.random.default_rng(0)
    out = []
    for start in range(0, len(pairs), batch_size):
        rows = np.arange(start, min(start + batch_size, len(pairs)))
        out.append(model(torch.from_numpy(pairs.movie[rows]).long(),
                         *user_bags(pairs, rows, data.histories, 0.0, rng)).numpy())
    return np.clip(pairs.baseline + np.concatenate(out), 0.5, 5.0)


def rmse(pred: np.ndarray, rating: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - rating) ** 2)))
