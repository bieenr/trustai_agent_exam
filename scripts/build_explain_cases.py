#!/usr/bin/env python3
"""Build the explanation-groundedness cases: one follow-up question per conversation, after a frozen turn 1.

Usage: python -m scripts.build_explain_cases --base-run <transcript directory of a recommend_test.json run>
An LLM plays the user: it reads the conversation and the agent's answer and asks a follow-up of
an assigned intent (rotated over the cases so each intent gets the same share). Each case stores
that run's transcript (tool calls and outputs included) so every agent version answers the
follow-up after the same context. Cases already in the output are kept; only missing ones are generated. Review
and edit the questions before running them with scripts.run_agent_cases --cases <output>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from evaluation.recommendation import load_cases
from trusted_ai import transcripts
from trusted_ai.config import AgentConfig
from trusted_ai.taste.config import ROOT_DIR


CASES = ROOT_DIR / "evaluation" / "recommendation" / "data" / "recommend_test.json"
OUTPUT = ROOT_DIR / "evaluation" / "explanation_groundedness" / "data" / "explain_cases.json"

INTENTS = {
    "why_movie": "Ask why they would like one specific movie from the answer (refer to it naturally, "
                 "e.g. by title or 'the second one').",
    "peer_opinion": "Ask what people with similar taste think of one movie mentioned in the answer.",
    "own_history": "Ask about your own past: whether you already watched or rated a movie mentioned in "
                   "the answer, or how you rated it.",
    "challenge_claim": "Question one concrete fact or number the assistant stated (a rating, a count, "
                       "an average, a genre preference) and ask where it comes from or to double-check it.",
    "profile": "Ask about your taste in general: which genres you like or avoid, or what you are missing.",
}

PROMPT = """You are role-playing a person chatting with a movie recommendation assistant.
Read the conversation and write the person's next message.
Intent of the message: {intent}
Style: casual and short (at most 25 words), in English, like a real chat message. Do not mention
that you are testing anything. Refer only to movies and facts that appear in the conversation.
Return JSON: {{"message": "...", "target_title": "movie title the message is about, or null"}}"""


class FollowUp(BaseModel):
    message: str = Field(min_length=1)
    target_title: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--model", default=AgentConfig().model)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from langchain.chat_models import init_chat_model

    model = init_chat_model(args.model, temperature=args.temperature).with_structured_output(FollowUp)
    base = {transcript["case_id"]: transcript for _, transcript in transcripts.load([args.base_run])}
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else []
    done = {row["base_case_id"] for row in existing}
    intents = list(INTENTS)
    rows = list(existing)
    try:
        for index, case in enumerate(load_cases(args.cases)):
            run = base.get(case.case_id)
            if case.case_id in done or run is None or "error" in run["turns"][-1]:
                continue
            intent = intents[index % len(intents)]
            conversation = [{"role": turn.role, "content": turn.content} for turn in case.conversation]
            conversation.append({"role": "assistant", "content": run["turns"][-1]["answer"]})
            followup = model.invoke([("system", PROMPT.format(intent=INTENTS[intent])),
                                     ("user", json.dumps(conversation, ensure_ascii=False))])
            rows.append({"case_id": f"{case.case_id}__{intent}", "base_case_id": case.case_id,
                         "user_id": case.user_id, "intent": intent, "target_title": followup.target_title,
                         "message": followup.message, "transcript": run})
            print(f"{case.case_id} [{intent}]: {followup.message}")
    finally:  # keep what was generated; a rerun only asks for the missing cases
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
