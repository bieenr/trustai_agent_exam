from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trusted_ai.schemas import Recommendation, ScoreComponents, ToolEvent
from trusted_ai.taste import TasteService
from trusted_ai.taste.dataset import TasteDataset


MAX_PEER_ROWS = 5
# Ranking and scorer internals: useful for the UI and debugging, but the agent must not quote them
# (the system prompt forbids models, probabilities and scores; a field it can read, it will quote).
INTERNAL_KEYS = frozenset({
    "model", "source", "score", "components", "backend", "content_backend", "content_similarity",
    "content_percentile", "query_similarity", "taste_similarity", "similarity", "used_taste_vector",
    "fallback_reason", "peer_predicted_rating", "baseline_predicted_rating", "movie_bias",
})


@dataclass
class ToolBundle:
    tools: list[Any]
    events: list[ToolEvent] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)


def _trim(value: Any) -> Any:
    """The copy of a tool result the agent sees: internals removed at every depth, and only the
    strongest peer ratings kept (enough to cite, small payload)."""
    if isinstance(value, dict):
        return {key: _trim(item)[:MAX_PEER_ROWS] if key == "peer_ratings" else _trim(item)
                for key, item in value.items() if key not in INTERNAL_KEYS}
    if isinstance(value, list):
        return [_trim(item) for item in value]
    return value


def _for_agent(row: dict[str, Any], dataset: TasteDataset) -> dict[str, Any]:
    """What the agent sees of one recommended movie: what it is, and how it is rated overall and
    by people with the user's taste. Verdicts and ranking signals stay with the UI."""
    movie_id = row["movie_id"]
    ratings = dataset.user_item.get(movie_id)
    mean_rating = None if ratings is None or not ratings.count() else round(float(ratings.mean()), 1)
    movie = dataset.movie(movie_id) or {}
    evidence = row.get("assessment", {}).get("evidence", {})
    n_peers, peer_avg = evidence.get("n_similar_raters", 0), evidence.get("weighted_avg")
    if not n_peers or peer_avg is None:
        similar_users = "No one with taste like yours has rated it yet."
    elif n_peers == 1:
        similar_users = f"The one person most like you who's seen it rated it {peer_avg:.1f}★."
    else:
        similar_users = f"The {n_peers} people most like you who've seen it rated it {peer_avg:.1f}★ on average."
    # movie_id lets the agent follow up with explain_match; the prompt keeps it out of answers.
    return {"movie_id": movie_id, "title": row["title"], "year": row["year"], "genres": row["genres"],
            "plot": movie.get("plot", ""), "mean_rating": mean_rating, "similar_users": similar_users}


def _evidence_lines(assessment: dict[str, Any]) -> list[str]:
    evidence = assessment.get("evidence", {})
    if "reasons" in evidence:  # learned scorer: plain-language reasons, strongest first
        lines = [f"Verdict: {assessment['verdict']}. {evidence['confidence']['text']}"]
        lines += [reason["text"] if reason["direction"] == "for" else f"On the other hand: {reason['text']}"
                  for reason in evidence["reasons"]]
        return lines + ([evidence["caveat"]] if evidence.get("caveat") else [])
    lines = [f"Verdict: {assessment['verdict']} (source: {assessment['source']})."]
    if evidence.get("weighted_avg") is not None:
        lines.append(f"{evidence['n_similar_raters']} similar users rated it, "
                     f"similarity-weighted average {evidence['weighted_avg']:.2f}.")
    if "content_percentile" in evidence:
        lines.append(f"Content similarity to your liked movies: {evidence['content_similarity']:.3f} "
                     f"({evidence['content_percentile']:.0f}th percentile, {evidence['content_backend']}).")
    for row in evidence.get("avoided_genres_matched", []):
        lines.append(f"Contains {row['genre']}, which you rate {row['deviation']:+.2f} vs your average.")
    return lines


def _to_recommendation(row: dict[str, Any]) -> Recommendation:
    components = row.get("components", {})
    cf = components.get("cf_predicted_rating")
    return Recommendation(
        movie_id=row["movie_id"], title=row["title"], year=row["year"], genres=row["genres"],
        scores=ScoreComponents(
            content=components.get("content_similarity") or row.get("query_similarity") or 0.0,
            collaborative=0.0 if cf is None else cf / 5.0,
            genre=components.get("genre_affinity", 0.0),
            final=row["score"],
        ),
        verified=row.get("verified"),
        evidence=_evidence_lines(row["assessment"]) if "assessment" in row else [],
    )


def build_tools(user_id: int, taste: TasteService, max_results: int) -> ToolBundle:
    """Create request-scoped tools so user_id never needs to enter chat text."""
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before creating the agent") from exc

    bundle = ToolBundle(tools=[])

    def record(name: str, arguments: dict[str, Any], count: int | None, result: Any) -> Any:
        """Log the call and return what the agent sees (the trimmed result)."""
        output = _trim(result)
        bundle.events.append(ToolEvent(name=name, arguments=arguments, result_count=count, output=output))
        return output

    def clamp(value: int) -> int:
        return min(max(value, 1), max_results)

    def publish(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Recommendation tools: expose the full list to the UI, give the agent the trimmed one."""
        bundle.recommendations = [_to_recommendation(row) for row in result["results"]]
        movies = [_for_agent(row, taste.dataset) for row in result["results"]]
        return record(name, arguments, len(movies), {**result, "results": movies})

    @tool
    def get_user_profile() -> dict[str, Any]:
        """Return the user's taste profile: genre deviations vs their own average (with
        confidence), avoided genres, blind spots, top/lowest rated movies and a summary."""
        profile = taste.get_user_profile(user_id)
        output = {key: value for key, value in profile.items() if key != "embedding_ref"}
        return record("get_user_profile", {}, profile.get("rating_count"), output)

    @tool
    def get_similar_users(k: int = 10) -> list[dict[str, Any]]:
        """Users whose mean-centered ratings best match this user's (>= 5 movies in common)."""
        rows = taste.get_similar_users(user_id, min(max(k, 1), 20))
        return record("get_similar_users", {"k": k}, len(rows), rows)

    @tool
    def get_peer_opinion(movie_title: str) -> dict[str, Any]:
        """How this user's similar users rated a movie (fuzzy title match), with the
        weighted average, each rating, and whether the user already rated it."""
        result = taste.get_peer_opinion(user_id, movie_title)
        return record("get_peer_opinion", {"movie_title": movie_title}, result.get("n_similar_raters", 0), result)

    @tool
    def search_by_description(query: str, exclude_genres: list[str] | None = None,
                              include_genres: list[str] | None = None, min_year: int | None = None,
                              max_year: int | None = None, top_n: int = 5) -> dict[str, Any]:
        """Find unseen movies matching a free-text description (plot, tone, genre), blended
        with the user's taste. Use exclude_genres/include_genres/years for constraints;
        include_genres keeps movies that have ALL listed genres, so list only the ones required."""
        arguments = {"query": query, "exclude_genres": exclude_genres, "include_genres": include_genres,
                     "min_year": min_year, "max_year": max_year, "top_n": top_n}
        result = taste.search_by_description(
            user_id, query, top_n=clamp(top_n), exclude_genres=exclude_genres,
            include_genres=include_genres, min_year=min_year, max_year=max_year,
        )
        return publish("search_by_description", arguments, result)

    @tool
    def recommend_for_user(top_n: int = 5, exclude_genres: list[str] | None = None,
                           include_genres: list[str] | None = None, min_year: int | None = None,
                           max_year: int | None = None) -> dict[str, Any]:
        """Personalised picks combining similar users, content similarity and genre
        affinity; every movie carries the evidence behind its verdict. Use min_year/max_year
        (release year, inclusive) for any year or decade the user asks for; include_genres keeps
        movies that have ALL listed genres, so list only the ones required."""
        arguments = {"exclude_genres": exclude_genres, "include_genres": include_genres,
                     "min_year": min_year, "max_year": max_year, "top_n": top_n}
        result = taste.recommend_for_user(user_id, top_n=clamp(top_n), exclude_genres=exclude_genres,
                                          include_genres=include_genres, min_year=min_year, max_year=max_year)
        return publish("recommend_for_user", arguments, result)

    @tool
    def explain_match(movie_id: int) -> dict[str, Any]:
        """Explain why the user would (not) like a movie: verdict evidence, the most similar
        movies they rated highly, and their deviation on the movie's genres."""
        result = taste.explain_match(user_id, movie_id)
        return record("explain_match", {"movie_id": movie_id}, len(result.get("most_similar_liked_movies", [])),
                      result)

    @tool
    def get_blind_spots() -> list[dict[str, Any]]:
        """Genres the catalog offers but the user has rarely or never rated."""
        rows = taste.get_blind_spots(user_id)
        return record("get_blind_spots", {}, len(rows), rows)

    bundle.tools = [get_user_profile, get_similar_users, get_peer_opinion, search_by_description,
                    recommend_for_user, explain_match, get_blind_spots]
    return bundle
