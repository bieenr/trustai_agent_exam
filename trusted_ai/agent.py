from __future__ import annotations

from typing import Any

from trusted_ai.config import AgentConfig
from trusted_ai.history import compact, last_input_tokens, llm_summarizer
from trusted_ai.schemas import AgentRequest, AgentResponse
from trusted_ai.session import InMemorySessionStore, SessionStore
from trusted_ai.taste import TasteService
from trusted_ai.tools import build_tools

SYSTEM_PROMPT = """You are a grounded movie discovery assistant.

Grounding
- Every fact about a movie (story, characters, cast, crew, reputation, awards, trivia) must come
  from this conversation's tool results, mostly its plot, or from the summary of earlier turns
  (which was written from them). Never add anything you remember or
  believe about a movie, and never call it famous, acclaimed or infamous unless a tool says so.
  If the user asks for something the tools don't give, say you don't have that detail.
- Don't reveal the ending or late twists unless the user asks for spoilers.
- The current user identity is already bound to the tools. Never ask for or invent a user ID.

Choosing tools
- Use recommend_for_user for open-ended recommendations and search_by_description when the
  user describes what they want; pass genre/year constraints as tool arguments.
- Use get_peer_opinion for "what do people like me think of X" and explain_match for "why".
- If a result has cold_start, the user has no ratings yet: present picks as well-liked by
  viewers overall (not "for you"), and ask what movies or kinds of stories they enjoy so you can
  search_by_description with that.

Voice
- Talk like a friend who knows movies well: warm, casual, straight to the picks. No preamble,
  no headings, no methodology, no disclaimers about how the picks were made.
- Never mention tools, data, the user's rating history or how many movies they rated, unless
  they ask about it.
- Keep it short: a line or two per movie. End with at most one short, natural follow-up question.

Explaining
- Only use numbers the tools return; never invent them. Weave them into the sentence the way a
  person would ("folks with taste like yours gave it 4.3★"), never as field names or raw output,
  and never show movie IDs, scores, models, probabilities or source names.
- Recommendation results are ranked best fit first. Each movie gives its plot (sum it up in one
  sentence, without spoilers), its average rating across all users (0.5-5★) and how people with
  taste like the user's rated it; that is the evidence for each pick.
- Only say the user loves a genre when its confidence is "medium" or "high"; genre deviations
  are relative to their own average rating.
- When an assessment has `reasons`, explain with those (strongest first; an "against" reason is
  a counterpoint), plus its `confidence` text and any `caveat`.
- Mention uncertainty in a few words only when it matters for a pick, not as a general caveat.
"""


class MovieAgent:
    """Framework-neutral service object intended to be called by an API handler."""

    def __init__(
        self,
        model: Any | None = None,
        config: AgentConfig | None = None,
        taste: TasteService | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.model = model or self.config.model
        self.taste = taste or TasteService()
        self.sessions = session_store or InMemorySessionStore()

    async def invoke(
        self,
        request: AgentRequest | None = None,
        *,
        user_id: int | None = None,
        message: str | None = None,
        session_id: str | None = None,
    ) -> AgentResponse:
        if request is None:
            request = AgentRequest(user_id=user_id, message=message, session_id=session_id)  # type: ignore[arg-type]
        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -r requirements.txt`") from exc

        bundle = build_tools(request.user_id, self.taste, self.config.max_results)
        graph = create_agent(model=self.model, tools=bundle.tools, system_prompt=SYSTEM_PROMPT)
        history = await self.sessions.get(request.session_id) if request.session_id else []
        compaction = None
        context_tokens = last_input_tokens(history)
        if context_tokens is not None and context_tokens > self.config.history_token_limit:
            history, compaction = await compact(history, llm_summarizer(self._chat_model()),
                                                self.config.keep_recent_turns)
            if compaction:
                compaction["trigger_input_tokens"] = context_tokens
        messages = history + [{"role": "user", "content": request.message}]
        state = await graph.ainvoke(
            {"messages": messages},
            config={"recursion_limit": self.config.max_agent_steps},
        )
        output_messages = state["messages"]
        usage = self._usage(output_messages[len(messages):])
        answer = self._message_text(output_messages[-1])
        if request.session_id:
            await self.sessions.set(request.session_id, output_messages)
        return AgentResponse(
            answer=answer,
            recommendations=bundle.recommendations,
            tool_events=bundle.events,
            metadata={"user_id": request.user_id, "session_id": request.session_id,
                      "model": str(self.config.model),
                      "cold_start": not self.taste.dataset.has_user(request.user_id),
                      "content_backend": self.taste.scorer.space.backend, "usage": usage,
                      "history_compaction": compaction},
        )

    def _chat_model(self) -> Any:
        if not isinstance(self.model, str):
            return self.model
        from langchain.chat_models import init_chat_model
        return init_chat_model(self.model)

    @staticmethod
    def _usage(new_messages: list[Any]) -> dict[str, int]:
        """Token usage and LLM calls of this turn (history messages are excluded)."""
        totals = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
        for message in new_messages:
            usage = getattr(message, "usage_metadata", None)
            if usage:
                totals["llm_calls"] += 1
                totals["input_tokens"] += usage.get("input_tokens", 0)
                totals["output_tokens"] += usage.get("output_tokens", 0)
        return totals

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
        return str(content)
