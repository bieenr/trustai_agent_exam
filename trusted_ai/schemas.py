from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    user_id: int
    message: str = Field(min_length=1)
    session_id: str | None = None


class ScoreComponents(BaseModel):
    """Ranking signals of a recommendation tool (0–1), for display; not the taste verdict."""

    content: float = 0.0
    collaborative: float = 0.0
    genre: float = 0.0
    final: float = 0.0


class Recommendation(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    scores: ScoreComponents = Field(default_factory=ScoreComponents)
    verified: bool | None = None  # score_candidate said "pass"; None when the tool did not check
    evidence: list[str] = Field(default_factory=list)


class ToolEvent(BaseModel):
    name: str
    arguments: dict[str, Any]
    result_count: int | None = None
    output: Any = None  # the payload the LLM received, for grounding checks


class AgentResponse(BaseModel):
    answer: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
