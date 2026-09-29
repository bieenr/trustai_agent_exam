from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.explanation_groundedness.evidence import not_found_queries
from evaluation.explanation_groundedness.llm import EXTRACT_PROMPT, JUDGE_PROMPT, CachedStructuredLLM
from evaluation.explanation_groundedness.schemas import (
    AnswerEvaluation,
    ClaimResult,
    EvidenceItem,
    ExtractedClaims,
    JudgeOutput,
)

# Internals the system prompt forbids quoting (models, probabilities, scores, source names, raw
# field names, movie IDs). Plain words like "score" (a rating) or "baseline" (the user's average)
# are fine on their own; they count only next to a raw number.
LEAK_PATTERN = re.compile(
    r"\b(models?|probabilit\w*|percentile|learned_model|embeddings?|cosine|"
    r"(?:scores?|similarity)(?: of)? \d?\.\d{2,}|movie ?id:? ?\d+|[a-z]+_[a-z]+(?:_[a-z]+)*)\b", re.IGNORECASE)


def leaked_terms(answer: str) -> list[str]:
    return sorted({match.group(0).lower() for match in LEAK_PATTERN.finditer(answer)})


class GroundednessEvaluator:
    """An LLM splits the answer into claims; an LLM judge labels each against the evidence."""

    def __init__(self, *, model: str, temperature: float = 0.0, prompt_version: str = "groundedness-v2",
                 cache_dir: Path = Path(".eval_cache/groundedness"),
                 audit_path: Path = Path("evaluation_audit.jsonl")) -> None:
        common = dict(model=model, temperature=temperature, prompt_version=prompt_version,
                      cache_dir=cache_dir, audit_path=audit_path)
        self.extractor = CachedStructuredLLM(step="extract", system_prompt=EXTRACT_PROMPT,
                                             schema=ExtractedClaims, **common)
        self.judge = CachedStructuredLLM(step="judge", system_prompt=JUDGE_PROMPT, schema=JudgeOutput, **common)

    async def evaluate(self, *, case_id: str, turn: int, intent: str | None, user_id: int, answer: str,
                       items: list[EvidenceItem]) -> AnswerEvaluation:
        """`items`: what the agent could see for this answer (evidence.evidence_for_turn)."""
        result = AnswerEvaluation(case_id=case_id, turn=turn, intent=intent, user_id=user_id,
                                  answer=answer, leaked_terms=leaked_terms(answer))
        result.not_found_queries = not_found_queries(items)

        claims = (await self.extractor.ainvoke(case_id, answer)).claims
        if not (claims or result.not_found_queries):
            return result
        payload = json.dumps({
            "evidence": [item.model_dump() for item in items],
            "answer": answer,
            "claims": [{"index": index, "text": claim.text, "type": claim.type} for index, claim in enumerate(claims)],
            "not_found_queries": result.not_found_queries,
        }, ensure_ascii=False)
        output = await self.judge.ainvoke(case_id, payload)
        judged = {judgement.index: judgement for judgement in output.judgements if 0 <= judgement.index < len(claims)}
        result.claims = [ClaimResult(**claims[index].model_dump(), label=judgement.label,
                                     evidence_id=judgement.evidence_id, reason=judgement.reason)
                         for index, judgement in sorted(judged.items())]
        if len(judged) < len(claims):
            result.error = f"judge returned no label for {len(claims) - len(judged)} claim(s)"
        if result.not_found_queries:
            result.states_not_found = output.states_not_found
        return result
