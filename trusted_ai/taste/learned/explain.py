"""Plain-language reasons for a verdict of the learned scorer.

A signal group's effect is how much the verdict's probability drops when that group is set to a
neutral value (its median on the pairs the model was fit on). The base row and one row per
disabled group go through the model in a single batch, so explaining costs about as much as
predicting. Only groups whose effect reaches MIN_EFFECT and that rest on actual data become
reasons, strongest first (missing data goes to the caveat instead); each is written from the
numbers the user can check (their ratings, similar users, genres), never from model internals.
Texts are in English; the agent rephrases them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from trusted_ai.taste.learned.features import FeatureBuilder
    from trusted_ai.taste.learned.model import ScoreModel


SIGNAL_GROUPS = {
    "similar_movies": ["item_knn_offset", "item_knn_weight"],
    "similar_users": ["peer_offset", "log_n_raters"],
    "rating_habits": ["user_mean", "user_std", "log_user_n"],
    "movie_reception": ["movie_bias", "log_movie_n"],
    "genres": ["genre_dev_mean", "genre_dev_min"],
    "plot": ["content_pos_minus_neg"],
    "overall_history": ["mlp_pred"],
}
MIN_EFFECT = 0.05       # probability change a signal needs to be shown
STRONG, MODERATE = 0.15, 0.08
MAX_REASONS = 4


def ablation_batch(row: dict[str, float], model: ScoreModel) -> pd.DataFrame:
    """The pair's features, then one copy per signal group with that group set to neutral."""
    neutral = model.meta.get("neutral", {})
    rows = [row] + [{**row, **{name: neutral.get(name, 0.0) for name in names}} for names in SIGNAL_GROUPS.values()]
    return pd.DataFrame(rows)


def explain(builder: FeatureBuilder, user_id: int, movie: dict[str, Any], row: dict[str, float],
            evidence: dict[str, Any], verdict: str, p_like: np.ndarray, p_dislike: np.ndarray) -> dict[str, Any]:
    """reasons / confidence / caveat for the batch of `ablation_batch` (first row = the pair itself).

    For "none" verdicts the reasons explain the lean towards or away from liking.
    """
    p = p_dislike if verdict == "fail" else p_like
    effects = dict(zip(SIGNAL_GROUPS, p[0] - p[1:]))
    texts = _texts(builder, user_id, movie, row, evidence)
    reasons = [{"signal": name, "direction": "for" if effect > 0 else "against",
                "strength": "strong" if abs(effect) >= STRONG else "moderate" if abs(effect) >= MODERATE else "weak",
                "text": texts[name]}
               for name, effect in sorted(effects.items(), key=lambda item: -abs(item[1]))
               if abs(effect) >= MIN_EFFECT and texts[name] is not None][:MAX_REASONS]
    movie_specific = evidence.get("n_similar_raters", 0) > 0 or not np.isnan(row["item_knn_offset"])
    caveat = None if movie_specific else (
        "Nobody with similar taste has rated this movie and you have not rated anything closely related, "
        "so this call rests on your general rating habits.")
    return {"reasons": reasons, "confidence": _confidence(verdict, float(p[0])), "caveat": caveat,
            "signal_effects": {name: round(float(effect), 3) for name, effect in effects.items()}}


def _confidence(verdict: str, probability: float) -> dict[str, Any]:
    if verdict == "none":
        return {"level": "low", "text": "The signals are mixed; this is not a confident call either way."}
    level = "high" if probability >= 0.85 else "medium"
    outcome = "liked" if verdict == "pass" else "disliked"
    return {"level": level,
            "text": f"About {round(probability * 10)} in 10 movies judged like this one were {outcome}."}


def readable_title(title: str) -> str:
    """MovieLens puts articles last ("Usual Suspects, The"); move them back to the front."""
    match = re.match(r"^(.*), (The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El)$", title)
    return f"{match.group(2)} {match.group(1)}" if match else title


def _texts(builder: FeatureBuilder, user_id: int, movie: dict[str, Any], row: dict[str, float],
           evidence: dict[str, Any]) -> dict[str, str | None]:
    """One sentence per signal group; None when the group has no data to point to."""
    similar = builder.similar_rated(user_id, movie["movie_id"])
    if similar:
        titles = ", ".join(f"{readable_title(m['title'])} ({m['rating']:g}★)" for m in similar)
        similar_text = (f"The movies you rated that are most like this one ({titles}) you rated "
                        f"{row['item_knn_offset']:+.1f}★ vs your average.")
    else:
        similar_text = None
    n_peers = evidence.get("n_similar_raters", 0)
    peers_text = (f"{n_peers} people with taste like yours rated it {evidence['weighted_avg']:.1f}★ on average."
                  if n_peers and evidence.get("weighted_avg") is not None
                  else None)
    mean = evidence["user_mean"]
    habit = "generous" if mean >= 3.8 else "demanding" if mean <= 3.2 else "middle-of-the-road"
    n_ratings = evidence.get("movie_n_ratings", 0)
    reception_text = (f"Other viewers rate it {evidence['movie_bias']:+.1f}★ vs their usual ({n_ratings} ratings)."
                      if n_ratings else None)
    deviations = builder.genre_deviations(user_id, movie)
    if deviations:
        genre, deviation = max(deviations.items(), key=lambda item: abs(item[1]))
        genre_text = f"You rate {genre} {deviation:+.1f}★ vs your average."
    else:
        genre_text = None
    plot_text = ("Its plot is closer to movies you liked than to ones you disliked."
                 if row["content_pos_minus_neg"] > 0 else
                 "Its plot is closer to movies you disliked than to ones you liked.")
    history_text = (f"Across your whole rating history, you would likely rate it about {row['mlp_pred']:.1f}★."
                    if not np.isnan(row["mlp_pred"]) else None)
    return {
        "similar_movies": similar_text, "similar_users": peers_text,
        "rating_habits": f"You are a {habit} rater ({mean:.1f}★ on average).",
        "movie_reception": reception_text, "genres": genre_text, "plot": plot_text,
        "overall_history": history_text,
    }
