"""Inputs of the MLP scorer: rating splits, reduced movie vectors, bias baseline and user bags.

The user vector of the plan (one slot per movie, 0 = not rated) is kept sparse: each user is a
list of (movie position, rating). Values fed to the model are ratings centred on the user's
baseline, so "not rated" (0) and "disliked" (negative) no longer collide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from trusted_ai.taste.config import TasteConfig
from trusted_ai.taste.dataset import split_genres
from trusted_ai.taste.embedding.store import load_movie_embeddings


USER_SHRINKAGE = 5.0


@dataclass(frozen=True)
class Pairs:
    """(user, movie) pairs with the baseline the model predicts a residual on."""

    user: np.ndarray        # row in Histories
    movie: np.ndarray       # row in the movie matrix
    rating: np.ndarray
    baseline: np.ndarray    # mu + b_u + b_i
    center: np.ndarray      # mu + b_u; subtracted from the user's other ratings
    leave_out: bool         # True: the pair's own rating is in the history and must be masked

    def __len__(self) -> int:
        return len(self.rating)


@dataclass(frozen=True)
class Histories:
    """Train ratings per user, sorted by movie position."""

    movies: list[np.ndarray]
    ratings: list[np.ndarray]


@dataclass(frozen=True)
class ExperimentData:
    movie_features: np.ndarray  # (n_movies, pca_dim [+ n_genres]): whitened PCA [+ genre one-hot]
    explained_variance: float
    genres: list[str]
    histories: Histories
    train: Pairs
    validation: Pairs
    test: Pairs
    validation_frame: pd.DataFrame  # userId / movieId / rating of the validation pairs, same order
    test_frame: pd.DataFrame    # userId / movieId / rating of the test pairs, same order
    user_ids: np.ndarray        # userId of each Histories row
    movie_ids: np.ndarray       # movieId of each movie-matrix row
    biases: Biases


def load(config: TasteConfig, pca_dim: int, with_genres: bool) -> ExperimentData:
    paths = config.paths
    embeddings = load_movie_embeddings(paths.embeddings_dir)
    if embeddings is None:
        raise SystemExit(f"No movie embeddings in {paths.embeddings_dir}")
    # PCA is unsupervised and sees no rating, so fitting it on every movie leaks nothing.
    pca = PCA(n_components=pca_dim, whiten=True, random_state=0)
    features = pca.fit_transform(embeddings.vectors).astype(np.float32)
    position = pd.Series(np.arange(len(embeddings.movie_ids)), index=embeddings.movie_ids)
    genres: list[str] = []
    if with_genres:
        one_hot = genre_one_hot(paths.movies, embeddings.movie_ids)
        genres = list(one_hot.columns)
        features = np.hstack([features, one_hot.to_numpy(np.float32)])

    split_dir = paths.train_ratings.parent
    train = pd.read_csv(paths.train_ratings)
    validation = pd.read_csv(split_dir / "ratings_validation.csv")
    test = pd.read_csv(paths.test_ratings)
    train = train[train["movieId"].isin(position.index)]
    users = pd.Series(np.arange(train["userId"].nunique()), index=np.sort(train["userId"].unique()))
    known = lambda frame: frame[frame["userId"].isin(users.index) & frame["movieId"].isin(position.index)]
    validation, test = known(validation).reset_index(drop=True), known(test).reset_index(drop=True)

    u = users.loc[train["userId"]].to_numpy()
    m = position.loc[train["movieId"]].to_numpy()
    r = train["rating"].to_numpy(np.float32)
    bias = Biases.fit(u, m, r, len(users), len(position), config.baseline_shrinkage)

    order = np.lexsort((m, u))
    bounds = np.searchsorted(u[order], np.arange(len(users) + 1))
    histories = Histories([m[order][a:b] for a, b in zip(bounds[:-1], bounds[1:])],
                          [r[order][a:b] for a, b in zip(bounds[:-1], bounds[1:])])

    return ExperimentData(
        movie_features=features,
        explained_variance=float(pca.explained_variance_ratio_.sum()),
        genres=genres,
        histories=histories,
        train=bias.leave_one_out(u, m, r),
        validation=bias.pairs(validation, users, position),
        test=bias.pairs(test, users, position),
        validation_frame=validation[["userId", "movieId", "rating"]],
        test_frame=test[["userId", "movieId", "rating"]],
        user_ids=users.index.to_numpy(),
        movie_ids=np.asarray(embeddings.movie_ids),
        biases=bias,
    )


def genre_one_hot(movies_csv, movie_ids: np.ndarray) -> pd.DataFrame:
    """0/1 genre columns in embedding order; the genre of the movie is explicit, not left to the text."""
    genres = pd.read_csv(movies_csv, usecols=["movieId", "genres"]).set_index("movieId")["genres"]
    lists = genres.reindex(movie_ids).map(split_genres)
    return pd.DataFrame([dict.fromkeys(g, 1) for g in lists]).fillna(0).sort_index(axis=1)


@dataclass(frozen=True)
class Biases:
    """Damped baseline r ≈ mu + b_u + b_i fitted on train, kept as sums for leave-one-out."""

    mu: float
    user_sum: np.ndarray    # Σ (r − mu) per user
    user_n: np.ndarray
    item_sum: np.ndarray    # Σ (r − mu − b_u) per movie
    item_n: np.ndarray
    item_shrinkage: float

    @classmethod
    def fit(cls, u, m, r, n_users: int, n_movies: int, item_shrinkage: float) -> "Biases":
        mu = float(r.mean())
        user_sum = np.bincount(u, r - mu, n_users)
        user_n = np.bincount(u, minlength=n_users).astype(float)
        b_u = user_sum / (user_n + USER_SHRINKAGE)
        item_sum = np.bincount(m, r - mu - b_u[u], n_movies)
        item_n = np.bincount(m, minlength=n_movies).astype(float)
        return cls(mu, user_sum, user_n, item_sum, item_n, item_shrinkage)

    @property
    def b_u(self) -> np.ndarray:
        return self.user_sum / (self.user_n + USER_SHRINKAGE)

    @property
    def b_i(self) -> np.ndarray:
        return self.item_sum / (self.item_n + self.item_shrinkage)

    def leave_one_out(self, u, m, r) -> Pairs:
        """Train pairs whose baseline and centring exclude the pair's own rating.

        With the full user mean, the centred history of the other movies would sum to exactly
        −(r − mean) and hand the target to the model. b_i still sees the target through b_u,
        a second-order effect of weight 1 / (n_u + shrinkage).
        """
        b_u = (self.user_sum[u] - (r - self.mu)) / (self.user_n[u] - 1 + USER_SHRINKAGE)
        b_i = (self.item_sum[m] - (r - self.mu - self.b_u[u])) / (self.item_n[m] - 1 + self.item_shrinkage)
        center = self.mu + b_u
        return Pairs(u, m, r, (center + b_i).astype(np.float32), center.astype(np.float32), leave_out=True)

    def pairs(self, frame: pd.DataFrame, users: pd.Series, position: pd.Series) -> Pairs:
        u = users.loc[frame["userId"]].to_numpy()
        m = position.loc[frame["movieId"]].to_numpy()
        center = self.mu + self.b_u[u]
        return Pairs(u, m, frame["rating"].to_numpy(np.float32), (center + self.b_i[m]).astype(np.float32),
                     center.astype(np.float32), leave_out=False)


def user_bags(pairs: Pairs, rows: np.ndarray, histories: Histories, drop_prob: float,
              rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EmbeddingBag inputs (movies, offsets, weights) for the users of `rows`.

    Weight = (rating − center) / sqrt(n kept). The target movie is masked for train pairs;
    `drop_prob` also hides that share of the other ratings (denoising, train only).
    """
    movies, weights, offsets, start = [], [], [], 0
    for row in rows:
        user_movies = histories.movies[pairs.user[row]]
        user_ratings = histories.ratings[pairs.user[row]]
        keep = np.ones(len(user_movies), dtype=bool)
        if pairs.leave_out:
            keep[np.searchsorted(user_movies, pairs.movie[row])] = False
        if drop_prob:
            keep &= rng.random(len(user_movies)) >= drop_prob
        kept = user_movies[keep]
        offsets.append(start)
        start += len(kept)
        movies.append(kept)
        weights.append((user_ratings[keep] - pairs.center[row]) / np.sqrt(max(len(kept), 1)))
    return (torch.from_numpy(np.concatenate(movies)).long(), torch.tensor(offsets),
            torch.from_numpy(np.concatenate(weights).astype(np.float32)))
