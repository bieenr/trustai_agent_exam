"""Conversation transcripts: the one format for chat logs (Streamlit) and evaluation runs.

A transcript holds the setup a conversation ran with (user, model, agent config, system prompt)
and one entry per turn: the user message, every tool call with the exact output the LLM received,
the answer, the recommendations shown in the UI, token usage and any history compaction. Turns
marked `replayed` were given as context (earlier conversation, a frozen first turn), not produced
in this run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from trusted_ai.schemas import AgentResponse


LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "conversations"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_transcript(user_id: int, session_id: str, model: str, agent_config: dict[str, Any],
                   system_prompt: str, **case: Any) -> dict[str, Any]:
    """`case` holds evaluation fields such as case_id and intent."""
    return {**case, "user_id": user_id, "session_id": session_id, "started_at": _now(), "model": model,
            "agent_config": agent_config, "system_prompt": system_prompt, "turns": []}


def add_turn(transcript: dict[str, Any], message: str, response: AgentResponse | None = None,
             error: str | None = None, latency_s: float | None = None) -> None:
    turn: dict[str, Any] = {"time": _now(), "user": message}
    if response is not None:
        turn.update({
            # `output` is exactly what the LLM received from the tool.
            "tool_calls": [event.model_dump(mode="json") for event in response.tool_events],
            "answer": response.answer,
            "recommendations": [movie.model_dump(mode="json") for movie in response.recommendations],
            "usage": response.metadata.get("usage"),
        })
        if response.metadata.get("history_compaction"):
            # Earlier turns were replaced by this summary before the turn ran.
            turn["history_compaction"] = response.metadata["history_compaction"]
    if latency_s is not None:
        turn["latency_s"] = latency_s
    if error is not None:
        turn["error"] = error
    transcript["turns"].append(turn)


def add_replayed_turns(transcript: dict[str, Any], turns: list[dict[str, Any]]) -> None:
    """Context turns: a text-only earlier exchange ({"user", "answer"}) or turns of an earlier run."""
    transcript["turns"] += [{"tool_calls": [], **turn, "replayed": True} for turn in turns]


def to_messages(turns: list[dict[str, Any]]) -> list[BaseMessage]:
    """Rebuild the agent's session history from turns (failed turns never reached the session).
    A turn's tool calls become one assistant tool-call message and its tool results. History
    compaction is not replayed: the agent compacts again if the context is still too large."""
    messages: list[BaseMessage] = []
    for index, turn in enumerate(turns):
        if turn.get("error"):
            continue
        messages.append(HumanMessage(content=turn["user"]))
        calls = turn.get("tool_calls") or []
        if calls:
            ids = [f"replay_{index}_{position}" for position in range(len(calls))]
            messages.append(AIMessage(content="", tool_calls=[
                {"name": call["name"], "args": call["arguments"], "id": call_id}
                for call, call_id in zip(calls, ids)]))
            messages += [ToolMessage(content=json.dumps(call.get("output"), ensure_ascii=False, default=str),
                                     tool_call_id=call_id, name=call["name"])
                         for call, call_id in zip(calls, ids)]
        messages.append(AIMessage(content=turn["answer"]))
    return messages


def file_name(transcript: dict[str, Any]) -> str:
    if transcript.get("case_id"):
        return f"{transcript['case_id']}.json"
    started = datetime.fromisoformat(transcript["started_at"]).strftime("%Y%m%d-%H%M%S")
    return f"{started}_user{transcript['user_id']}_{transcript['session_id'][:8]}.json"


def to_json(transcript: dict[str, Any]) -> str:
    # numpy scalars in tool outputs become plain numbers.
    return json.dumps(transcript, ensure_ascii=False, indent=2,
                      default=lambda value: value.item() if hasattr(value, "item") else str(value))


def save(transcript: dict[str, Any], directory: Path = LOG_DIR) -> Path:
    """Rewrite the conversation's file with every turn so far."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / file_name(transcript)
    path.write_text(to_json(transcript), encoding="utf-8")
    return path


def load(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    """Transcripts from files and directories (every *.json inside, sorted)."""
    files = [file for path in paths for file in (sorted(path.glob("*.json")) if path.is_dir() else [path])]
    return [(file, json.loads(file.read_text(encoding="utf-8"))) for file in files]
