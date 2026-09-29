from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from evaluation.explanation_groundedness.schemas import AnswerEvaluation

LABELS = ("supported", "contradicted", "unsupported", "external")
# Claims about the dataset that an explanation should cite with a number.
DATA_TYPES = ("user_rating", "user_stat", "genre_pref", "peer_stat", "prediction")


def _share(part: int, whole: int) -> float | None:
    return None if whole == 0 else round(part / whole, 4)


def _rate(values: list[bool | None]) -> dict[str, Any]:
    applicable = [value for value in values if value is not None]
    return {"applicable": len(applicable), "rate": _share(sum(applicable), len(applicable))}


def summarize_answers(all_evaluations: list[AnswerEvaluation]) -> dict[str, Any]:
    """Metrics over answers that were fully evaluated; failed ones are only counted."""
    evaluations = [evaluation for evaluation in all_evaluations if evaluation.error is None]
    claims = [claim for evaluation in evaluations for claim in evaluation.claims]
    labels = Counter(claim.label for claim in claims)
    return {
        "answers": len(evaluations),
        "claims": len(claims),
        "claims_per_answer": _share(len(claims), len(evaluations)),
        "claim_labels": {label: _share(labels[label], len(claims)) for label in LABELS},
        "fully_grounded_answers": _share(
            sum(all(claim.label == "supported" for claim in evaluation.claims) for evaluation in evaluations),
            len(evaluations)),
        "answers_citing_data": _share(
            sum(any(claim.type in DATA_TYPES and claim.value is not None and claim.label == "supported"
                    for claim in evaluation.claims) for evaluation in evaluations),
            len(evaluations)),
        "answers_leaking_internals": _share(sum(bool(evaluation.leaked_terms) for evaluation in evaluations),
                                            len(evaluations)),
        "states_not_found": _rate([evaluation.states_not_found for evaluation in evaluations]),
        "answers_with_errors": len(all_evaluations) - len(evaluations),
    }


def by_claim_type(evaluations: list[AnswerEvaluation]) -> dict[str, dict[str, Any]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for evaluation in (evaluation for evaluation in evaluations if evaluation.error is None):
        for claim in evaluation.claims:
            groups[claim.type][claim.label] += 1
    return {kind: {"claims": sum(counts.values()),
                   **{label: _share(counts[label], sum(counts.values())) for label in LABELS}}
            for kind, counts in sorted(groups.items())}


def summarize(evaluations: list[AnswerEvaluation]) -> dict[str, Any]:
    turns: dict[str, list[AnswerEvaluation]] = defaultdict(list)
    intents: dict[str, list[AnswerEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        turns[f"turn_{evaluation.turn}"].append(evaluation)
        intents[evaluation.intent or "no intent"].append(evaluation)
    return {
        "overall": summarize_answers(evaluations),
        "by_turn": {name: summarize_answers(group) for name, group in sorted(turns.items())},
        "by_intent": {name: summarize_answers(group) for name, group in sorted(intents.items())},
        "by_claim_type": by_claim_type(evaluations),
        "leaked_terms": dict(Counter(term for evaluation in evaluations for term in evaluation.leaked_terms)),
    }
