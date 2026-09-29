from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from trusted_ai.taste.dataset import TasteDataset


# A genre counts as a blind spot when the user's share of it is below this fraction
# of the genre's share in the catalog (or the user never rated it at all).
BLIND_SPOT_RATIO = 0.25
MIN_CATALOG_SHARE = 0.02


def confidence(n_ratings: int) -> str:
    if n_ratings < 3:
        return "low"
    return "medium" if n_ratings < 10 else "high"


def genre_preferences(dataset: TasteDataset, user_id: int) -> list[dict[str, Any]]:
    """Per-genre average relative to the user's own mean, so a 4.0 from a 4.3-average
    user reads as mildly negative instead of being ranked on raw position."""
    ratings = dataset.user_ratings(user_id)
    if ratings.empty:
        return []
    user_mean = float(ratings["rating"].mean())
    by_genre: dict[str, list[float]] = defaultdict(list)
    for movie_id, rating in zip(ratings["movieId"], ratings["rating"]):
        for genre in dataset.movies.at[dataset.movie_position[int(movie_id)], "genre_list"]:
            by_genre[genre].append(float(rating))
    rows = [
        {
            "genre": genre,
            "avg_rating": round(sum(values) / len(values), 3),
            "deviation": round(sum(values) / len(values) - user_mean, 3),
            "n_ratings": len(values),
            "confidence": confidence(len(values)),
        }
        for genre, values in by_genre.items()
    ]
    return sorted(rows, key=lambda row: (-row["deviation"], -row["n_ratings"], row["genre"]))


def avoid_genres(preferences: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in ("genre", "deviation", "n_ratings", "confidence")}
        for row in preferences
        if row["deviation"] <= threshold and row["confidence"] != "low"
    ]


def catalog_genre_share(movies: pd.DataFrame) -> dict[str, float]:
    counts = Counter(genre for genres in movies["genre_list"] for genre in genres)
    return {genre: count / len(movies) for genre, count in counts.items()}


def blind_spots(dataset: TasteDataset, user_id: int,
                catalog_share: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Genres the catalog offers in volume but the user has (almost) never rated."""
    catalog_share = catalog_share or catalog_genre_share(dataset.movies)
    ratings = dataset.user_ratings(user_id)
    counts = Counter(
        genre for movie_id in ratings["movieId"]
        for genre in dataset.movies.at[dataset.movie_position[int(movie_id)], "genre_list"]
    )
    total = max(len(ratings), 1)
    result = []
    for genre, share in catalog_share.items():
        user_share = counts.get(genre, 0) / total
        never_rated = counts.get(genre, 0) == 0
        if never_rated or (share >= MIN_CATALOG_SHARE and user_share < BLIND_SPOT_RATIO * share):
            result.append({
                "genre": genre, "n_ratings": counts.get(genre, 0),
                "user_share": round(user_share, 4), "catalog_share": round(share, 4),
            })
    return sorted(result, key=lambda row: (row["user_share"] / row["catalog_share"], -row["catalog_share"]))
