from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trusted_ai.taste.config import TastePaths


NO_GENRES = "(no genres listed)"


def split_genres(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [genre for genre in value.split("|") if genre and genre != NO_GENRES]


@dataclass
class TasteDataset:
    """Train-split ratings plus the movie catalog. Test ratings are never loaded here."""

    movies: pd.DataFrame
    ratings: pd.DataFrame
    user_item: pd.DataFrame
    ratings_path: Path

    def __post_init__(self) -> None:
        self.movies = self.movies.reset_index(drop=True)
        self.movie_position = {int(mid): pos for pos, mid in enumerate(self.movies["movieId"])}
        self.user_means = self.ratings.groupby("userId")["rating"].mean()
        # UTC year of each user's latest train rating: the "now" of that user in this offline dataset.
        last = self.ratings.groupby("userId")["timestamp"].max() if "timestamp" in self.ratings else None
        self.user_last_year = (pd.Series(dtype=int) if last is None
                               else pd.to_datetime(last, unit="s", utc=True).dt.year.astype(int))
        self._by_user = {int(uid): frame for uid, frame in self.ratings.groupby("userId")}

    @property
    def user_ids(self) -> list[int]:
        return [int(uid) for uid in self.user_item.index]

    def has_user(self, user_id: int) -> bool:
        return user_id in self._by_user

    def user_ratings(self, user_id: int) -> pd.DataFrame:
        return self._by_user.get(user_id, self.ratings.iloc[0:0])

    def movie(self, movie_id: int) -> dict | None:
        position = self.movie_position.get(int(movie_id))
        if position is None:
            return None
        row = self.movies.iloc[position]
        return {
            "movie_id": int(row["movieId"]), "title": str(row["title"]),
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "genres": list(row["genre_list"]), "plot": str(row["plot"]),
        }

    def find_movies(self, title: str, limit: int = 5) -> list[dict]:
        """Exact title match first, then substring, then fuzzy (difflib) match."""
        from difflib import get_close_matches

        needle = title.strip().casefold()
        titles = self.movies["title"].str.casefold()
        exact = self.movies.index[titles == needle].tolist()
        partial = self.movies.index[titles.str.contains(needle, regex=False)].tolist()
        ranked = list(dict.fromkeys(exact + sorted(partial, key=lambda pos: len(titles[pos]))))
        if not ranked:
            lookup = dict(zip(titles, self.movies.index))
            ranked = [lookup[match] for match in get_close_matches(needle, list(lookup), n=limit, cutoff=0.6)]
        return [self.movie(int(self.movies.iloc[pos]["movieId"])) for pos in ranked[:limit]]


def load_dataset(paths: TastePaths | None = None, ratings_path: Path | None = None) -> TasteDataset:
    paths = paths or TastePaths()
    ratings_path = ratings_path or paths.train_ratings
    movies = pd.read_csv(paths.movies, usecols=["movieId", "title", "year", "genres", "plot"])
    movies["plot"] = movies["plot"].fillna("")
    movies["genre_list"] = movies["genres"].map(split_genres)
    ratings = pd.read_csv(ratings_path, usecols=lambda column: column in ("userId", "movieId", "rating", "timestamp"))
    unknown = set(ratings["movieId"]) - set(movies["movieId"])
    if unknown:
        raise ValueError(f"Ratings reference movies missing from the catalog: {sorted(unknown)[:10]}")
    user_item = ratings.pivot(index="userId", columns="movieId", values="rating").sort_index()
    return TasteDataset(movies=movies, ratings=ratings, user_item=user_item, ratings_path=ratings_path)


def assert_no_test_leak(dataset: TasteDataset, test_path: Path) -> None:
    """Fail loudly if any held-out (user, movie) pair made it into the training ratings."""
    test = pd.read_csv(test_path, usecols=["userId", "movieId"])
    train_pairs = set(zip(dataset.ratings["userId"], dataset.ratings["movieId"]))
    leaked = [pair for pair in zip(test["userId"], test["movieId"]) if pair in train_pairs]
    if leaked:
        raise ValueError(f"{len(leaked)} test ratings are present in the training data, e.g. {leaked[:3]}")
