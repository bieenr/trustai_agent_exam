"""What the agent could see when it wrote an answer, rebuilt from its transcript: the earlier turns
still in its context (user messages, tool outputs, answers, or the summary that replaced them)
and the current turn's user message and tool outputs."""

from __future__ import annotations

from typing import Any

from evaluation.explanation_groundedness.schemas import EvidenceItem


def from_tools(tools: list[dict[str, Any]], prefix: str) -> list[EvidenceItem]:
    return [EvidenceItem(id=f"{prefix}.{event['name']}#{index}", source="tool", name=event["name"],
                         content={"arguments": event["arguments"], "output": event.get("output")})
            for index, event in enumerate(tools)]


def _turn_items(turn: dict[str, Any], prefix: str) -> list[EvidenceItem]:
    return [EvidenceItem(id=f"{prefix}.user", source="conversation", name="user", content=turn["user"]),
            *from_tools(turn.get("tool_calls") or [], prefix),
            EvidenceItem(id=f"{prefix}.answer", source="conversation", name="assistant", content=turn["answer"])]


def context_turns(turns: list[dict[str, Any]], index: int) -> tuple[list[int], str | None]:
    """Indices of the earlier turns the agent still had verbatim at turn `index`, and the summary
    standing in for the others. Mirrors trusted_ai.history.compact: each compaction keeps the first
    turn and folds the oldest `summarized_turns` turns after it into the summary."""
    kept: list[int] = []
    summary = None
    for position in range(index + 1):
        compaction = turns[position].get("history_compaction")
        if compaction:
            del kept[1:1 + compaction["summarized_turns"]]
            summary = compaction["summary"]
        if position < index and not turns[position].get("error"):
            kept.append(position)
    return kept, summary


def evidence_for_turn(transcript: dict[str, Any], index: int) -> list[EvidenceItem]:
    """Evidence for the answer of turn `index` (0-based): earlier context, then this turn's
    user message and tool calls. Items of turn n are prefixed H<n+1>, the summary S, this turn T."""
    turns = transcript["turns"]
    kept, summary = context_turns(turns, index)
    items: list[EvidenceItem] = []
    for position in kept:
        items += _turn_items(turns[position], f"H{position + 1}")
        if position == kept[0] and summary is not None:
            items.append(EvidenceItem(id="S.summary", source="conversation", name="summary of earlier turns",
                                      content=summary))
    current = turns[index]
    return items + [EvidenceItem(id="T.user", source="conversation", name="user", content=current["user"]),
                    *from_tools(current.get("tool_calls") or [], "T")]


def not_found_queries(items: list[EvidenceItem]) -> list[str]:
    queries = []
    for item in items:
        if item.source == "tool" and item.id.startswith("T.") and isinstance(item.content, dict):
            output = item.content.get("output")
            if isinstance(output, dict) and output.get("found") is False:
                queries.append(str(output.get("query") or item.content.get("arguments")))
    return queries
