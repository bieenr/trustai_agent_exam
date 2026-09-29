from __future__ import annotations

from typing import Any

from evaluation.recommendation.catalog import Catalog
from evaluation.recommendation.llm_judge import SemanticJudge
from evaluation.recommendation.schemas import ConversationEvalCase, CriterionResult, RatingLabel


UNKNOWN_MOVIE = "movie_id is not present in the catalog"


def check_hard_constraints(case: ConversationEvalCase, movie: dict[str, Any] | None,
                           catalog: Catalog) -> CriterionResult:
    if movie is None:
        return CriterionResult(passed=False, reasons=[UNKNOWN_MOVIE])
    constraints = case.expected.hard_constraints
    genres = {genre.casefold() for genre in movie["genres"]}
    missing = [g for g in constraints.must_have_genres if g.casefold() not in genres]
    forbidden = [g for g in constraints.must_not_have_genres if g.casefold() in genres]
    max_year = (constraints.max_year if isinstance(constraints.max_year, int)
                else catalog.train_max_year(case.user_id))
    year = movie["year"]

    failures: list[str] = []
    if missing:
        failures.append(f"missing required genres: {missing}")
    if forbidden:
        failures.append(f"contains forbidden genres: {forbidden}")
    if constraints.must_not_be_previously_rated and (case.user_id, movie["movie_id"]) in catalog.train_rated:
        failures.append("movie was previously rated in ratings_train")
    if year is None or year < constraints.min_year:
        failures.append(f"movie year does not meet min_year={constraints.min_year}")
    if year is None or year > max_year:
        failures.append(f"movie year exceeds max_year={max_year}")
    return CriterionResult(passed=not failures, reasons=failures)


def check_semantic(case: ConversationEvalCase, movie: dict[str, Any] | None,
                   judge: SemanticJudge) -> CriterionResult | None:
    if not case.expected.semantic_requirement:
        return None
    if movie is None:
        return CriterionResult(passed=False, reasons=[UNKNOWN_MOVIE])
    conversation = [message.model_dump() for message in case.conversation]
    decision = judge.semantic(case_id=case.case_id, conversation=conversation, movie=movie)
    return CriterionResult(passed=decision.pass_, reasons=[decision.reason])


def rating_label(rating: float) -> RatingLabel:
    if rating >= 4.0:
        return "positive"
    if rating <= 2.5:
        return "negative"
    return "neutral"


def check_taste(test_rating: float | None) -> CriterionResult | None:
    """Only held-out ratings count as taste ground truth; unrated movies are not judged."""
    if test_rating is None:
        return None
    label = rating_label(test_rating)
    return CriterionResult(passed=label == "positive",
                           reasons=[f"held-out rating {test_rating:g} ({label})"])
