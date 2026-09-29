"""MLP that predicts the residual rating − (mu + b_u + b_i) of one (user, movie) pair."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ScoreMLP(nn.Module):
    """Input = [movie PCA vector, learned user vector, content profile of the user].

    `cf_bag` is the plan's Linear layer on the sparse user vector, stored as an EmbeddingBag:
    sum_j w_j · E[j] equals the one-hot-weighted matrix product without materialising 5k slots.
    `content_bag` reuses the frozen movie vectors, i.e. a rating-weighted average of the
    movies the user has seen.
    """

    def __init__(self, movie_features: np.ndarray, cf_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        features = torch.from_numpy(movie_features)
        n_movies, content_dim = features.shape
        self.movie = nn.Embedding.from_pretrained(features, freeze=True)
        self.content_bag = nn.EmbeddingBag.from_pretrained(features, freeze=True, mode="sum")
        self.cf_bag = nn.EmbeddingBag(n_movies, cf_dim, mode="sum")
        nn.init.normal_(self.cf_bag.weight, std=0.01)

        layers, width = [], 2 * content_dim + cf_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, movie: torch.Tensor, bag_movies: torch.Tensor, bag_offsets: torch.Tensor,
                bag_weights: torch.Tensor) -> torch.Tensor:
        user_cf = self.cf_bag(bag_movies, bag_offsets, per_sample_weights=bag_weights)
        user_content = self.content_bag(bag_movies, bag_offsets, per_sample_weights=bag_weights)
        return self.mlp(torch.cat([self.movie(movie), user_cf, user_content], dim=1)).squeeze(1)
