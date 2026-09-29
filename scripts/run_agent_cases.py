#!/usr/bin/env python3
"""Run the movie agent on the conversation test cases and save each conversation as a transcript.

Usage: python -m scripts.run_agent_cases --verdict-filter pass_first [--model openai:...] [--tag name] [--concurrency 4]
       [--results-dir DIR] [--cases FILE] [--no-recs]
For each case the earlier turns are given as context (replayed turns in the transcript) and the
last user turn is sent as the message. Explanation cases (evaluation/explanation_groundedness/data/
explain_cases.json) carry the transcript of an earlier first turn instead (tool calls included), so
every agent version answers the follow-up after the same context. The recommended movie IDs are
the list shown to the user (the last recommend_for_user / search_by_description call of the turn).
Writes <results-dir>/agent_<filter>[_<tag>]/<case_id>.json (trusted_ai.transcripts format, the input
of scripts.run_groundedness_eval) and agent_<filter>[_<tag>]_recs.json ({case_id: [movie_id, ...]},
the input of scripts.run_conversation_eval; skipped with --no-recs). Runs of both case files with
the same tag share one directory. Cases already saved without an error are skipped, so a rerun
only calls the LLM for missing cases.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from evaluation.recommendation import load_cases
from trusted_ai import transcripts
from trusted_ai.agent import SYSTEM_PROMPT, MovieAgent
from trusted_ai.config import AgentConfig
from trusted_ai.taste import TasteConfig, TasteService
from trusted_ai.taste.config import ROOT_DIR


RESULTS_DIR = ROOT_DIR / "evaluation" / "recommendation" / "results"
CASES = ROOT_DIR / "evaluation" / "recommendation" / "data" / "recommend_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verdict-filter", choices=["off", "pass_first", "pass_only"], default="pass_first")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--model", default=AgentConfig().model, help="agent chat model (langchain id)")
    parser.add_argument("--tag", default="", help="suffix of the output names, e.g. the model name")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-recs", action="store_true", help="do not write the _recs.json file")
    return parser.parse_args()


def load_turns(cases: Path) -> list[dict]:
    """Per case: its ids, the context turns to replay and the message to send."""
    rows = json.loads(cases.read_text(encoding="utf-8"))
    if rows and "transcript" in rows[0]:  # explanation cases: an earlier run's first turn
        return [{"case": {"case_id": row["case_id"], "base_case_id": row["base_case_id"], "intent": row["intent"],
                          "target_title": row["target_title"]},
                 "user_id": row["user_id"], "context": row["transcript"]["turns"], "message": row["message"]}
                for row in rows]
    result = []
    for case in load_cases(cases):
        earlier = case.conversation[:-1]
        context = [{"user": earlier[index].content,
                    "answer": earlier[index + 1].content if index + 1 < len(earlier) else ""}
                   for index in range(0, len(earlier), 2)]
        result.append({"case": {"case_id": case.case_id}, "user_id": case.user_id, "context": context,
                       "message": case.conversation[-1].content})
    return result


async def run_case(agent: MovieAgent, row: dict, directory: Path, gate: asyncio.Semaphore) -> None:
    async with gate:
        case_id = row["case"]["case_id"]
        session_id = f"eval-{case_id}"
        transcript = transcripts.new_transcript(row["user_id"], session_id, str(agent.model), asdict(agent.config),
                                                SYSTEM_PROMPT, **row["case"])
        transcripts.add_replayed_turns(transcript, row["context"])
        await agent.sessions.set(session_id, transcripts.to_messages(transcript["turns"]))
        started = time.perf_counter()
        try:
            response = await agent.invoke(user_id=row["user_id"], message=row["message"], session_id=session_id)
        except Exception as exc:  # keep going; the case is retried on the next run
            print(f"{case_id}: {type(exc).__name__}: {exc}")
            transcripts.add_turn(transcript, row["message"], error=f"{type(exc).__name__}: {exc}")
        else:
            print(f"{case_id}: {len(response.recommendations)} movies")
            transcripts.add_turn(transcript, row["message"], response,
                                 latency_s=round(time.perf_counter() - started, 2))
        transcripts.save(transcript, directory)


def finished(path: Path) -> bool:
    return path.exists() and "error" not in json.loads(path.read_text(encoding="utf-8"))["turns"][-1]


async def main() -> None:
    args = parse_args()
    name = f"agent_{args.verdict_filter}" + (f"_{args.tag}" if args.tag else "")
    directory = args.results_dir / name
    rows = [row for row in load_turns(args.cases) if not finished(directory / f"{row['case']['case_id']}.json")]
    print(f"{len(rows)} cases to run with verdict_filter={args.verdict_filter}, model={args.model} -> {directory}")

    taste = TasteService(replace(TasteConfig(), verdict_filter=args.verdict_filter))
    taste.scorer.features.item_sim  # build once, before tools run concurrently
    agent = MovieAgent(model=args.model, taste=taste)
    gate = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(run_case(agent, row, directory, gate) for row in rows))

    saved = [transcript for _, transcript in transcripts.load([directory])] if directory.exists() else []
    recs = {transcript["case_id"]: [movie["movie_id"] for movie in transcript["turns"][-1].get("recommendations", [])]
            for transcript in saved}
    if not args.no_recs:
        (args.results_dir / f"{name}_recs.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    failed = [transcript["case_id"] for transcript in saved if "error" in transcript["turns"][-1]]
    print(f"{directory}: {len(saved)} cases, {sum(bool(ids) for ids in recs.values())} with recommendations, "
          f"{len(failed)} errors: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
