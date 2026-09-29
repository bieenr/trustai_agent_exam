#!/usr/bin/env python3
"""Check the agent's answers against the tool outputs and conversation it had (explanation groundedness).

Usage:
  python -m scripts.run_groundedness_eval --transcripts logs/conversations/ \
      --output evaluation/explanation_groundedness/results/groundedness_logs.json
  python -m scripts.run_groundedness_eval --transcripts evaluation/explanation_groundedness/results/agent_pass_first_v3/
--transcripts takes transcript files or directories (Streamlit chat logs, scripts.run_agent_cases
runs). Every turn the agent produced is checked (replayed context turns and failed turns are
skipped) against what it could see: the earlier turns still in its context, or the summary that
replaced them, plus that turn's tool outputs. Writes per-answer results (JSON) and a Markdown
summary next to it. LLM steps (claim extraction, judge) are cached in .eval_cache/groundedness/
and audited in evaluation_audit.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluation.explanation_groundedness import (
    AnswerEvaluation,
    GroundednessEvaluator,
    evidence_for_turn,
    summarize,
)
from trusted_ai import transcripts
from trusted_ai.config import AgentConfig
from trusted_ai.taste.config import ROOT_DIR


BASE = ROOT_DIR / "evaluation" / "explanation_groundedness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", type=Path, nargs="+", required=True, help="transcript files or directories")
    parser.add_argument("--output", type=Path, default=BASE / "results" / "groundedness.json")
    parser.add_argument("--model", default=AgentConfig().model, help="extractor and judge model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-version", default="groundedness-v2")
    parser.add_argument("--concurrency", type=int, default=6)
    return parser.parse_args()


def jobs(paths: list[Path]) -> list[dict]:
    """One job per answer the agent produced; case_id is `<conversation>#t<turn number>`."""
    result = []
    for path, transcript in transcripts.load(paths):
        conversation = transcript.get("case_id") or path.stem
        for index, turn in enumerate(transcript["turns"]):
            if turn.get("replayed") or turn.get("error") or "answer" not in turn:
                continue
            result.append(dict(case_id=f"{conversation}#t{index + 1}", turn=index + 1,
                               intent=transcript.get("intent"), user_id=transcript["user_id"],
                               answer=turn["answer"], items=evidence_for_turn(transcript, index)))
    return result


def pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def write_summary(summary: dict, path: Path) -> None:
    lines = ["# Explanation groundedness — summary", "",
             "| group | answers (failed) | claims/answer | supported | contradicted | unsupported | external | "
             "fully grounded | cites data | leaks internals | states not found |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = {"overall": summary["overall"], **summary["by_turn"],
              **{f"intent: {name}": value for name, value in summary["by_intent"].items()}}
    for name, row in groups.items():
        labels = row["claim_labels"]
        lines.append(
            f"| {name} | {row['answers']} ({row['answers_with_errors']}) | {row['claims_per_answer'] or 0:.1f} | "
            f"{pct(labels['supported'])} | "
            f"{pct(labels['contradicted'])} | {pct(labels['unsupported'])} | {pct(labels['external'])} | "
            f"{pct(row['fully_grounded_answers'])} | {pct(row['answers_citing_data'])} | "
            f"{pct(row['answers_leaking_internals'])} | "
            f"{pct(row['states_not_found']['rate'])} (n={row['states_not_found']['applicable']}) |")
    lines += ["", "| claim type | claims | supported | contradicted | unsupported | external |", "|---|---|---|---|---|---|"]
    for kind, row in summary["by_claim_type"].items():
        lines.append(f"| {kind} | {row['claims']} | {pct(row['supported'])} | {pct(row['contradicted'])} | "
                     f"{pct(row['unsupported'])} | {pct(row['external'])} |")
    lines += ["", f"Leaked terms: {summary['leaked_terms']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    args = parse_args()
    evaluator = GroundednessEvaluator(model=args.model, temperature=args.temperature,
                                      prompt_version=args.prompt_version)
    gate = asyncio.Semaphore(args.concurrency)

    async def run_one(job: dict) -> AnswerEvaluation:
        async with gate:
            try:
                evaluation = await evaluator.evaluate(**job)
            except Exception as exc:  # keep going; cached steps make a rerun cheap
                evaluation = AnswerEvaluation(case_id=job["case_id"], turn=job["turn"], intent=job["intent"],
                                              user_id=job["user_id"], answer=job["answer"],
                                              error=f"{type(exc).__name__}: {exc}")
            print(f"{job['case_id']}: {len(evaluation.claims)} claims, "
                  f"{sum(claim.label != 'supported' for claim in evaluation.claims)} not supported"
                  + (f" [{evaluation.error}]" if evaluation.error else ""))
            return evaluation

    evaluations = await asyncio.gather(*(run_one(job) for job in jobs(args.transcripts)))
    summary = summarize(list(evaluations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "answers": [e.model_dump() for e in evaluations]},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary, args.output.with_suffix(".md"))
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
