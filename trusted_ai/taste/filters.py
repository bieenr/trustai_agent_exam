from __future__ import annotations

import numpy as np

from trusted_ai.taste.dataset import TasteDataset


def candidate_mask(
    dataset: TasteDataset,
    user_id: int,
    *,
    exclude_genres: list[str] | None = None,
    include_genres: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    exclude_seen: bool = True,
    until_last_rating: bool = False,
) -> np.ndarray:
    """Boolean mask over `dataset.movies` rows that satisfy the request's hard constraints.

    `include_genres`: a movie must have every listed genre. `until_last_rating`: no movie released
    after the year of the user's latest rating (the dataset's "now" for that user).
    """
    movies = dataset.movies
    mask = np.ones(len(movies), dtype=bool)
    if exclude_seen:
        seen = set(dataset.user_ratings(user_id)["movieId"])
        mask &= ~movies["movieId"].isin(seen).to_numpy()
    genre_sets = movies["genre_list"].map(lambda genres: {genre.casefold() for genre in genres})
    if exclude_genres:
        blocked = {genre.casefold() for genre in exclude_genres}
        mask &= ~genre_sets.map(lambda genres: bool(genres & blocked)).to_numpy()
    if include_genres:
        wanted = {genre.casefold() for genre in include_genres}
        mask &= genre_sets.map(lambda genres: wanted <= genres).to_numpy()
    if until_last_rating and user_id in dataset.user_last_year.index:
        last = int(dataset.user_last_year[user_id])
        max_year = last if max_year is None else min(max_year, last)
    years = movies["year"].to_numpy()
    if min_year is not None:
        mask &= years >= min_year
    if max_year is not None:
        mask &= years <= max_year
    return mask
