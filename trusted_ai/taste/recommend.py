from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from trusted_ai.taste.collaborative import predict_from_peers
from trusted_ai.taste.genre_profile import genre_preferences
from trusted_ai.taste.filters import candidate_mask
from trusted_ai.taste.scoring import CandidateScorer


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True).to_numpy()


def genre_affinity(scorer: CandidateScorer, user_id: int) -> np.ndarray:
    """Mean confidence-weighted deviation of each movie's genres, in rating points."""
    weight = {"low": 0.25, "medium": 0.75, "high": 1.0}
    deviation = {row["genre"]: row["deviation"] * weight[row["confidence"]]
                 for row in genre_preferences(scorer.dataset, user_id)}
    return np.array([
        np.mean([deviation.get(genre, 0.0) for genre in genres]) if genres else 0.0
        for genres in scorer.dataset.movies["genre_list"]
    ])


def popular_movies(scorer: CandidateScorer, user_id: int, *, top_n: int = 10,
                   **filters: Any) -> dict[str, Any]:
    """Cold start: the best-liked movies overall, each mean rating shrunk toward the catalog mean
    so a movie with two 5★ ratings does not beat one loved by a hundred people."""
    dataset = scorer.dataset
    prior = scorer.config.cold_start_prior
    stats = dataset.ratings.groupby("movieId")["rating"].agg(["sum", "size", "mean"])
    shrunk = (stats["sum"] + prior * dataset.ratings["rating"].mean()) / (stats["size"] + prior)
    positions = stats.index.map(dataset.movie_position).to_numpy()
    final = np.full(len(dataset.movies), -np.inf)
    final[positions] = shrunk.to_numpy()
    mask = candidate_mask(dataset, user_id, until_last_rating=False, **filters)
    final = np.where(mask, final, -np.inf)
    results = []
    for pos in [pos for pos in np.argsort(-final)[:top_n] if np.isfinite(final[pos])]:
        movie_id = int(dataset.movies.at[pos, "movieId"])
        results.append({
            "movie_id": movie_id,
            "title": str(dataset.movies.at[pos, "title"]),
            "year": int(dataset.movies.at[pos, "year"]),
            "genres": list(dataset.movies.at[pos, "genre_list"]),
            "score": round(float(final[pos]), 4),
            "n_ratings": int(stats.at[movie_id, "size"]),
            "assessment": scorer.score(user_id, movie_id),
        })
    return {"user_id": user_id, "cold_start": True,
            "basis": "No ratings from this user yet: these are the best-liked movies across all users.",
            "results": results}


def recommend_for_user(scorer: CandidateScorer, user_id: int, *, top_n: int = 10,
                       exclude_genres: list[str] | None = None,
                       include_genres: list[str] | None = None,
                       min_year: int | None = None, max_year: int | None = None) -> dict[str, Any]:
    """Blend peer prediction, content similarity and genre affinity over unseen movies.

    Each signal is converted to a within-user percentile before blending so the three
    scales are comparable. Users with no ratings get `popular_movies` instead.
    """
    config = scorer.config
    dataset = scorer.dataset
    if not dataset.has_user(user_id):
        return popular_movies(scorer, user_id, top_n=top_n, exclude_genres=exclude_genres,
                              include_genres=include_genres, min_year=min_year, max_year=max_year)
    peers = scorer.similar_users(user_id)
    predicted = predict_from_peers(dataset, user_id, peers, config.cf_shrinkage)
    cf = np.full(len(dataset.movies), np.nan)
    n_raters = np.zeros(len(dataset.movies), dtype=int)
    if not predicted.empty:
        positions = predicted["movieId"].map(dataset.movie_position).to_numpy()
        cf[positions] = predicted["predicted_rating"].to_numpy()
        n_raters[positions] = predicted["n_peer_raters"].to_numpy()
    content = scorer.content_scores(user_id)
    genre = genre_affinity(scorer, user_id)

    # A movie no peer rated gets the neutral CF percentile (0.5) rather than dropping the
    # term; otherwise obscure films would be ranked on content + genre alone and win.
    has_cf = ~np.isnan(cf)
    cf_rank = np.full(len(cf), 0.5)
    if has_cf.any():
        cf_rank[has_cf] = _percentile_rank(cf[has_cf])
    components = [(cf_rank, config.cf_weight), (_percentile_rank(genre), config.genre_weight)]
    if content is not None:
        components.append((_percentile_rank(content), config.content_weight))
    final = sum(weight * values for values, weight in components) / sum(w for _, w in components)

    mask = candidate_mask(dataset, user_id, exclude_genres=exclude_genres, include_genres=include_genres,
                          min_year=min_year, max_year=max_year, until_last_rating=config.cap_year_at_last_rating)
    final = np.where(mask, final, -np.inf)
    order = [pos for pos in np.argsort(-final)[:top_n] if np.isfinite(final[pos])]
    results = []
    for pos in order:
        movie_id = int(dataset.movies.at[pos, "movieId"])
        results.append({
            "movie_id": movie_id,
            "title": str(dataset.movies.at[pos, "title"]),
            "year": int(dataset.movies.at[pos, "year"]),
            "genres": list(dataset.movies.at[pos, "genre_list"]),
            "score": round(float(final[pos]), 4),
            "components": {
                "cf_predicted_rating": None if np.isnan(cf[pos]) else round(float(cf[pos]), 3),
                "cf_peer_raters": int(n_raters[pos]),
                "content_similarity": None if content is None else round(float(content[pos]), 4),
                "genre_affinity": round(float(genre[pos]), 3),
            },
            "assessment": scorer.score(user_id, movie_id),
        })
    return {"user_id": user_id, "content_backend": scorer.space.backend, "results": results}


def explain_match(scorer: CandidateScorer, user_id: int, movie_id: int, n_neighbors: int = 3) -> dict[str, Any]:
    """Verdict evidence plus the user's highly rated movies closest to the candidate."""
    dataset = scorer.dataset
    movie = dataset.movie(movie_id)
    if movie is None:
        return {"found": False, "movie_id": movie_id}
    ratings = dataset.user_ratings(user_id)
    liked = ratings[ratings["rating"] >= scorer.config.liked_rating]
    neighbors: list[dict[str, Any]] = []
    if not liked.empty:
        space = scorer.space
        candidate = space.movie_vectors(np.array([movie_id]))
        liked_vectors = space.movie_vectors(liked["movieId"].to_numpy())
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # Accelerate BLAS noise
            product = liked_vectors @ candidate.T
        similarity = (product.toarray() if sparse.issparse(product) else np.asarray(product)).ravel()
        for index in np.argsort(-similarity)[:n_neighbors]:
            neighbor = dataset.movie(int(liked["movieId"].iloc[index]))
            neighbors.append({
                "movie_id": neighbor["movie_id"], "title": neighbor["title"],
                "user_rating": float(liked["rating"].iloc[index]),
                "similarity": round(float(similarity[index]), 4),
            })
    shared = [row for row in genre_preferences(dataset, user_id) if row["genre"] in movie["genres"]]
    return {
        "found": True,
        "movie": {key: movie[key] for key in ("movie_id", "title", "year", "genres")},
        "already_rated": bool((ratings["movieId"] == movie_id).any()),
        "assessment": scorer.score(user_id, movie_id),
        "most_similar_liked_movies": neighbors,
        "genre_preferences_for_movie_genres": shared,
        "content_backend": scorer.space.backend,
    }


def select_by_verdict(rows: list[dict[str, Any]], top_n: int, mode: str) -> list[dict[str, Any]]:
    """Keep top_n ranked rows given their `assessment` verdicts (see TasteConfig.verdict_filter).

    Order inside each group is the original ranking, so relevance to the request still decides
    among movies with the same verdict. Rows get `verified` = the verdict was pass.
    """
    if mode not in ("off", "pass_first", "pass_only"):
        raise ValueError(f"Unknown verdict_filter {mode!r}")
    for row in rows:
        row["verified"] = row["assessment"]["verdict"] == "pass"
    if mode == "off":
        return rows[:top_n]
    passed = [row for row in rows if row["verified"]]
    if mode == "pass_only":
        return passed[:top_n]
    unsure = [row for row in rows if not row["verified"] and row["assessment"]["verdict"] != "fail"]
    return (passed + unsure)[:top_n]

