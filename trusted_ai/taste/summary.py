from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SUMMARY_PROMPT_VERSION = "taste-summary-v2"
SUMMARY_PROMPT = """You summarise the movie taste of one MovieLens user.
Use only the figures provided (genre deviation from the user's own average rating,
confidence, highest/lowest rated movies). Do not use outside knowledge about the user.
Ignore genres with confidence "low". Write 1-2 specific sentences in English; do not start with "This user".
Return only the summary, no Markdown."""


def summary_facts(profile: dict[str, Any]) -> dict[str, Any]:
    """The compact, numbers-only input shared by the template and the LLM summariser."""
    confident = [row for row in profile["genre_preferences"] if row["confidence"] != "low"]
    return {
        "rating_count": profile["rating_count"],
        "average_rating": profile["average_rating"],
        "preferred_genres": [
            {"genre": row["genre"], "deviation": row["deviation"], "n_ratings": row["n_ratings"]}
            for row in confident if row["deviation"] > 0
        ][:5],
        "avoid_genres": profile["avoid_genres"],
        "top_rated_titles": [movie["title"] for movie in profile["top_rated_movies"][:5]],
        "lowest_rated_titles": [movie["title"] for movie in profile["lowest_rated_movies"][:3]],
    }


def template_summary(profile: dict[str, Any]) -> str:
    """Deterministic fallback used when no LLM is requested or available."""
    facts = summary_facts(profile)
    liked = ", ".join(row["genre"] for row in facts["preferred_genres"][:3])
    parts = []
    if liked:
        parts.append(f"Rates {liked} above their own {facts['average_rating']:.2f} average")
    else:
        parts.append(f"Rates genres fairly evenly (average {facts['average_rating']:.2f})")
    if facts["top_rated_titles"]:
        parts[-1] += f"; favourites include {', '.join(facts['top_rated_titles'][:3])}"
    avoided = ", ".join(row["genre"] for row in facts["avoid_genres"][:3])
    if avoided:
        parts.append(f"tends to avoid {avoided}")
    sentence = "; ".join(parts) + "."
    if profile["density_bucket"] == "sparse":
        sentence += f" Short history ({profile['rating_count']} ratings), so the signal is still weak."
    return sentence


class LLMSummariser:
    """Writes `taste_summary` from `summary_facts` only, cached by input hash + prompt version."""

    def __init__(self, model: str, cache_dir: Path, temperature: float = 0.0) -> None:
        self.model_name = model
        self.cache_dir = cache_dir
        self.temperature = temperature
        self._model: Any | None = None

    def __call__(self, profile: dict[str, Any]) -> str:
        payload = json.dumps(summary_facts(profile), ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(f"{SUMMARY_PROMPT_VERSION}:{self.model_name}:{payload}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        response = self._get_model().invoke([("system", SUMMARY_PROMPT), ("user", payload)])
        text = str(getattr(response, "content", response)).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    def _get_model(self) -> Any:
        if self._model is None:
            from langchain.chat_models import init_chat_model

            self._model = init_chat_model(self.model_name, temperature=self.temperature)
        return self._model
