"""The two LLM steps (claim extraction, judge): prompts and an on-disk cached caller."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


EXTRACT_PROMPT = """You extract factual claims from a movie assistant's answer so they can be checked
against the data the assistant had.

Extract every statement that asserts a fact about: a movie (title, year, genres, plot, tone),
the user (their ratings, rating count, average, genre preferences, blind spots), similar users
(how many rated a movie, their average or individual ratings), or a prediction/confidence for the
user. Split compound sentences into atomic claims, one fact each, restated so each claim stands
alone (name the movie; do not write "it"). Keep numbers exactly as written.

Do not extract: suggestions, questions, offers ("Want me to..."), greetings, or statements about
what the assistant is going to do. Every recommended movie yields a movie_ref claim with its title
(and year/genres if the answer states them).

Types: movie_ref (title/year/genres of a movie), user_rating (the user's rating of a specific
movie, or whether they rated/watched it), user_stat (rating count, average, rating habits),
genre_pref (genre liking, deviation, confidence, blind spots), peer_stat (similar users' counts,
averages, ratings), prediction (expected rating or confidence for this user), qualitative (plot,
tone, "similar to X", any other descriptive statement).
For each claim give movie_title (as written) when it is about one movie, and value when it states
a single number (4.3 for "4.3★", 20 for "20 raters", 0.6 for "0.6 below average").
"""

JUDGE_PROMPT = """You check whether each claim in a movie assistant's answer is supported by the
evidence the assistant had: tool outputs (JSON), the earlier conversation and, when earlier
turns were compacted, a summary of them (written from their tool outputs, so it counts as data).

Label each claim:
- supported: the evidence states it, or it follows directly (rounding, simple counting, "high
  confidence" for a confidence text of "about 9 in 10"). A number is supported if it matches any
  field that plausibly means the same thing, even if another field disagrees.
- contradicted: the evidence states something incompatible (a different rating, count, year,
  genre, or the opposite direction).
- unsupported: the evidence does not contain it and it is about the dataset, the user or similar
  users (e.g. an invented number, a movie the tools never returned, a rating that is not there).
- external: general movie knowledge not present in the evidence (plot details, director, actors,
  tone, awards) — true or not, it did not come from the data.
Things the user said earlier count as evidence for restating what the user said, not as facts
about the dataset. Give the evidence item id you relied on (null if none) and a one-line reason.

Also answer, only when not_found_queries in the input is non-empty (otherwise null):
- states_not_found: for those lookups, does the answer say the movie was not found (instead of
  answering as if it had data)?
"""


class CachedStructuredLLM:
    """One prompt with structured output, cached on disk by prompt version, model, step and input,
    with every uncached call appended to an audit file."""

    def __init__(self, *, step: str, system_prompt: str, schema: type[BaseModel], model: str,
                 temperature: float, prompt_version: str, cache_dir: Path, audit_path: Path) -> None:
        self.step, self.system_prompt, self.schema = step, system_prompt, schema
        self.model_name, self.temperature, self.prompt_version = model, temperature, prompt_version
        self.cache_dir, self.audit_path = cache_dir, audit_path
        self._model: Any | None = None

    async def ainvoke(self, case_id: str, payload: str) -> BaseModel:
        raw_key = f"{self.prompt_version}:{self.model_name}:{self.step}:{case_id}:{payload}"
        cache_path = self.cache_dir / f"{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()}.json"
        if cache_path.exists():
            return self.schema.model_validate_json(cache_path.read_text(encoding="utf-8"))
        result = await self._get_model().ainvoke([("system", self.system_prompt), ("user", payload)])
        result = result if isinstance(result, self.schema) else self.schema.model_validate(result)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "step": self.step,
                  "prompt_version": self.prompt_version, "model": self.model_name,
                  "temperature": self.temperature, "case_id": case_id,
                  "input": {"system_prompt": self.system_prompt, "user": payload}, "output": result.model_dump()}
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result

    def _get_model(self) -> Any:
        if self._model is None:
            from langchain.chat_models import init_chat_model

            self._model = init_chat_model(self.model_name, temperature=self.temperature
                                          ).with_structured_output(self.schema)
        return self._model
