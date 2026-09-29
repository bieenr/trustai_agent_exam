from __future__ import annotations

from typing import Any

from trusted_ai.taste.collaborative import UserSimilarity
from trusted_ai.taste.config import TasteConfig
from trusted_ai.taste.dataset import TasteDataset
from trusted_ai.taste.genre_profile import avoid_genres, blind_spots, genre_preferences


MOVIE_LIST_LIMIT = 10


def density_bucket(rating_count: int) -> str:
    if rating_count < 30:
        return "sparse"
    return "medium" if rating_count < 100 else "dense"


def _movie_list(dataset: TasteDataset, ratings, ascending: bool) -> list[dict[str, Any]]:
    ordered = ratings.sort_values(["rating", "movieId"], ascending=[ascending, True]).head(MOVIE_LIST_LIMIT)
    return [
        {"movie_id": int(movie_id), "title": dataset.movie(int(movie_id))["title"], "rating": float(rating)}
        for movie_id, rating in zip(ordered["movieId"], ordered["rating"])
    ]


def build_profile(dataset: TasteDataset, similarity: UserSimilarity, config: TasteConfig,
                  user_id: int, catalog_share: dict[str, float],
                  embedding_row: int | None = None) -> dict[str, Any]:
    """Readable numbers only: no plots and no vectors (vectors live in .npy files)."""
    ratings = dataset.user_ratings(user_id)
    preferences = genre_preferences(dataset, user_id)
    profile = {
        "user_id": user_id,
        "rating_count": len(ratings),
        "average_rating": round(float(ratings["rating"].mean()), 3),
        "density_bucket": density_bucket(len(ratings)),
        "genre_preferences": preferences,
        "avoid_genres": avoid_genres(preferences, config.avoid_deviation),
        "blind_spot_genres": [row["genre"] for row in blind_spots(dataset, user_id, catalog_share)],
        "similar_users": similarity.top_k(user_id, config.similar_users_k, config.min_common_ratings),
        "top_rated_movies": _movie_list(dataset, ratings, ascending=False),
        "lowest_rated_movies": _movie_list(dataset, ratings, ascending=True),
        "taste_summary": None,
        "embedding_ref": None,
    }
    if embedding_row is not None:
        profile["embedding_ref"] = {"file": "user_taste_embeddings.npy", "row": embedding_row}
    return profile
