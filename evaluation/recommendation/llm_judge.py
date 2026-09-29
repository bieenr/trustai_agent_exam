from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from evaluation.recommendation.schemas import JudgeDecision


SEMANTIC_PROMPT = """You are a movie recommendation semantic evaluator.
Decide whether the candidate movie fits the user's request in the conversation.
Use only the supplied conversation and movie information. Do not infer user taste.
Return pass=true only when the movie is semantically appropriate.
Give a short, evidence-based reason.
Return exactly one JSON object with this schema:
{"pass": true, "reason": "Short evidence-based explanation"}
The pass field must be a JSON boolean (true or false), not a string.
Do not include Markdown or any text outside the JSON object.
"""


class SemanticJudge(Protocol):
    def semantic(self, *, case_id: str, conversation: list[dict[str, str]],
                 movie: dict[str, Any]) -> JudgeDecision: ...


class CachedLLMJudge:
    """Semantic judge with an on-disk cache keyed by prompt version, judge model, case and movie."""

    def __init__(
        self,
        *,
        model: str = "openai:deepseek/deepseek-v4.1-flash",
        temperature: float = 0.0,
        prompt_version: str = "conversation-eval-v2",
        cache_dir: str | Path = ".eval_cache",
        audit_path: str | Path = "evaluation_audit.jsonl",
    ) -> None:
        self.model_name = model
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.cache_dir = Path(cache_dir)
        self.audit_path = Path(audit_path)
        self._model: Any | None = None

    def semantic(self, *, case_id: str, conversation: list[dict[str, str]],
                 movie: dict[str, Any]) -> JudgeDecision:
        movie_id = int(movie["movie_id"])
        cache_path = self._cache_path(case_id, movie_id)
        if cache_path.exists():
            return JudgeDecision.model_validate_json(cache_path.read_text(encoding="utf-8"))

        payload = {"conversation": conversation, "semantic_requirement": True, "movie": movie}
        response = self._get_model().invoke(
            [("system", SEMANTIC_PROMPT), ("user", json.dumps(payload, ensure_ascii=False))]
        )
        decision = response if isinstance(response, JudgeDecision) else JudgeDecision.model_validate(response)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(decision.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
        self._audit(case_id, movie_id, payload, decision)
        return decision

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from langchain.chat_models import init_chat_model
            except ImportError as exc:
                raise RuntimeError("Install dependencies with `pip install -r requirements.txt`") from exc
            self._model = init_chat_model(
                self.model_name, temperature=self.temperature
            ).with_structured_output(JudgeDecision)
        return self._model

    def _cache_path(self, case_id: str, movie_id: int) -> Path:
        # The model is part of the key: another judge model must not reuse these decisions.
        raw_key = f"{self.prompt_version}:{self.model_name}:semantic:{case_id}:{movie_id}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _audit(self, case_id: str, movie_id: int, payload: dict[str, Any],
               decision: JudgeDecision) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_version": self.prompt_version,
            "judge_type": "semantic",
            "case_id": case_id,
            "movie_id": movie_id,
            "model": self.model_name,
            "temperature": self.temperature,
            "input": {"system_prompt": SEMANTIC_PROMPT, **payload},
            "output": decision.model_dump(by_alias=True),
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
