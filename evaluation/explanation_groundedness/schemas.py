from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ClaimType = Literal[
    "movie_ref",       # a movie's title, year or genres
    "user_rating",     # how the user rated a specific movie, or whether they rated it
    "user_stat",       # the user's rating count, average rating, rating habits
    "genre_pref",      # the user's liking / deviation / confidence for a genre, blind spots
    "peer_stat",       # what similar users rated: averages, counts, individual ratings
    "prediction",      # expected rating or confidence for the user on a movie
    "qualitative",     # plot, tone, similarity between movies, other descriptive statements
]
Label = Literal["supported", "contradicted", "unsupported", "external"]


class Claim(BaseModel):
    text: str = Field(min_length=1, description="the claim, restated as one short self-contained sentence")
    type: ClaimType
    movie_title: str | None = Field(None, description="movie the claim is about, as written in the answer")
    value: float | None = Field(None, description="the number the claim states, if any (e.g. 4.3 for 4.3★)")


class ExtractedClaims(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class ClaimJudgement(BaseModel):
    index: int
    label: Label
    evidence_id: str | None = None
    reason: str = Field(min_length=1)


class JudgeOutput(BaseModel):
    judgements: list[ClaimJudgement] = Field(default_factory=list)
    states_not_found: bool | None = Field(
        None, description="only when a lookup found nothing: does the answer say so instead of inventing")


class EvidenceItem(BaseModel):
    id: str
    source: Literal["tool", "conversation"]
    name: str  # tool name, or the speaker for conversation items
    content: Any


class ClaimResult(Claim):
    label: Label
    evidence_id: str | None = None
    reason: str


class AnswerEvaluation(BaseModel):
    case_id: str
    turn: int
    intent: str | None = None
    user_id: int
    answer: str
    claims: list[ClaimResult] = Field(default_factory=list)
    leaked_terms: list[str] = Field(default_factory=list)
    not_found_queries: list[str] = Field(default_factory=list)
    states_not_found: bool | None = None
    error: str | None = None
