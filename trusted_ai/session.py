from __future__ import annotations

from typing import Any, Protocol


class SessionStore(Protocol):
    async def get(self, session_id: str) -> list[Any]: ...

    async def set(self, session_id: str, messages: list[Any]) -> None: ...


class InMemorySessionStore:
    """Development default. Replace with a Redis-backed implementation in an API server."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[Any]] = {}

    async def get(self, session_id: str) -> list[Any]:
        return list(self._sessions.get(session_id, []))

    async def set(self, session_id: str, messages: list[Any]) -> None:
        self._sessions[session_id] = list(messages)
