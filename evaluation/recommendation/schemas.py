from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RatingLabel = Literal["positive", "neutral", "negative"]


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class HardConstraints(BaseModel):
    must_have_genres: list[str] = Field(default_factory=list)
    must_not_have_genres: list[str] = Field(default_factory=list)
    must_not_be_previously_rated: bool = False
    min_year: int = 0
    max_year: int | Literal["derive_from_train_max_timestamp_of_user_id"]


class ExpectedBehavior(BaseModel):
    hard_constraints: HardConstraints
    semantic_requirement: bool


class ConversationEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    user_id: int
    conversation: list[ConversationMessage] = Field(min_length=1)
    expected: ExpectedBehavior


class JudgeDecision(BaseModel):
    pass_: bool = Field(alias="pass")
    reason: str = Field(min_length=1)

    model_config = {"populate_by_name": True}


class CriterionResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class MovieEvaluation(BaseModel):
    """Independent verdicts for one recommended movie; None means the criterion does not apply."""

    rank: int
    movie_id: int
    title: str | None = None
    valid_movie_id: bool
    hard_constraints: CriterionResult
    semantic: CriterionResult | None = None
    taste: CriterionResult | None = None
    test_rating: float | None = None
    test_rating_label: RatingLabel | None = None


class RankingMetrics(BaseModel):
    """Top-k metrics over the movies a criterion could judge, in recommendation order."""

    k: int
    judged_count: int
    hits: tuple[int, ...]
    precision: float
    ndcg: float
    hit_rate: float


class TasteMetrics(RankingMetrics):
    negative_hits: tuple[int, ...]
    dislike_rate: float
    negative_hit_rate: float


class CaseMetrics(BaseModel):
    hard_constraints: RankingMetrics
    semantic: RankingMetrics | None = None
    taste: TasteMetrics | None = None


class CaseEvaluation(BaseModel):
    case_id: str
    user_id: int
    requested_k: int
    evaluated_count: int
    movies: list[MovieEvaluation]
    metrics: CaseMetrics
