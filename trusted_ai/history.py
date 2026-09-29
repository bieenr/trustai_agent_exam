from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


SUMMARY_ID = "history-summary"
SUMMARY_HEADER = "(Context note, not written by the user) Summary of the earlier part of this conversation:\n"
SUMMARY_PROMPT = """Summarise the earlier part of a conversation between a user and a movie
recommendation assistant, so the assistant can carry on without it.

Keep, as short bullet points:
- what the user asked for and anything they said about their taste or mood;
- every movie recommended or discussed, with its year and the numbers the tools gave for it
  (average rating, how people with similar taste rated it), and how the user reacted to it;
- facts about the user's taste that the tools reported.
Give each movie's story in one short phrase at most. Use only information in the transcript,
and keep numbers exactly as written. At most 250 words.
"""

# (previous summary or None, messages to fold in) -> new summary text
Summarizer = Callable[[str | None, list[BaseMessage]], Awaitable[str]]


def is_summary(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and message.id == SUMMARY_ID


def split_turns(messages: list[BaseMessage]) -> tuple[list[list[BaseMessage]], str | None]:
    """Group messages into turns, each starting at a user message and running through the
    assistant's tool calls, tool results and final answer. Returns the turns and the text of an
    existing summary message, if any."""
    turns: list[list[BaseMessage]] = []
    summary = None
    for message in messages:
        if is_summary(message):
            summary = str(message.content).removeprefix(SUMMARY_HEADER)
        elif isinstance(message, HumanMessage) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns, summary


def last_input_tokens(messages: list[BaseMessage]) -> int | None:
    """Prompt size of the latest LLM call, as reported by the API: the size of the context."""
    for message in reversed(messages):
        usage = getattr(message, "usage_metadata", None)
        if isinstance(message, AIMessage) and usage:
            return usage.get("input_tokens")
    return None


async def compact(messages: list[BaseMessage], summarize: Summarizer,
                  keep_recent_turns: int) -> tuple[list[BaseMessage], dict[str, Any] | None]:
    """Keep the first turn and the last `keep_recent_turns` turns; fold the turns in between
    (and any earlier summary) into one summary message placed after the first turn."""
    turns, previous = split_turns(messages)
    middle = turns[1:len(turns) - keep_recent_turns]
    if not middle:
        return messages, None
    folded = [message for turn in middle for message in turn]
    summary = await summarize(previous, folded)
    compacted = [*turns[0], HumanMessage(content=SUMMARY_HEADER + summary, id=SUMMARY_ID),
                 *(message for turn in turns[len(turns) - keep_recent_turns:] for message in turn)]
    return compacted, {"summary": summary, "summarized_turns": len(middle),
                       "summarized_messages": len(folded)}


def render(messages: list[BaseMessage]) -> str:
    """Plain-text transcript for the summariser."""
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, ToolMessage):
            lines.append(f"Tool result ({message.name}): {message.content}")
        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                lines.append(f"Assistant called {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
            if message.content:
                lines.append(f"Assistant: {message.content}")
    return "\n".join(lines)


def llm_summarizer(model: Any) -> Summarizer:
    async def summarize(previous: str | None, messages: list[BaseMessage]) -> str:
        earlier = f"Summary of the turns before these (fold it in):\n{previous}\n\n" if previous else ""
        reply = await model.ainvoke([
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"{earlier}Transcript:\n{render(messages)}"},
        ])
        return str(reply.content).strip()
    return summarize
