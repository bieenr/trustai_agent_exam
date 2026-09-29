from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "ml-latest-small-filtered"


@dataclass(frozen=True)
class Catalog:
    movies: dict[int, dict[str, Any]]
    train_rated: set[tuple[int, int]]
    train_max_timestamps: dict[int, int]
    test_ratings: dict[tuple[int, int], float]

    def train_max_year(self, user_id: int) -> int:
        try:
            timestamp = self.train_max_timestamps[user_id]
        except KeyError as exc:
            raise ValueError(f"No train timestamp found for user_id={user_id}") from exc
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).year


def load_catalog(data_dir: str | Path = DATA_DIR) -> Catalog:
    data_dir = Path(data_dir)
    split_dir = data_dir / "ranking_split"
    movies = pd.read_csv(data_dir / "movies_with_plots.csv",
                         usecols=["movieId", "title", "year", "genres", "plot"])
    train = pd.read_csv(split_dir / "ratings_train.csv", usecols=["userId", "movieId", "timestamp"])
    test = pd.read_csv(split_dir / "ratings_test.csv", usecols=["userId", "movieId", "rating"])
    return Catalog(
        movies={
            int(row.movieId): {
                "movie_id": int(row.movieId), "title": str(row.title),
                "year": None if pd.isna(row.year) else int(row.year),
                "genres": [] if pd.isna(row.genres) else str(row.genres).split("|"),
                "plot": "" if pd.isna(row.plot) else str(row.plot),
            }
            for row in movies.itertuples(index=False)
        },
        train_rated={(int(row.userId), int(row.movieId)) for row in train.itertuples(index=False)},
        train_max_timestamps={
            int(user_id): int(timestamp)
            for user_id, timestamp in train.groupby("userId")["timestamp"].max().items()
        },
        test_ratings={
            (int(row.userId), int(row.movieId)): float(row.rating)
            for row in test.itertuples(index=False)
        },
    )
