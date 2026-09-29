from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from evaluation.recommendation.catalog import Catalog
from evaluation.recommendation.criteria import (
    check_hard_constraints,
    check_semantic,
    check_taste,
    rating_label,
)
from evaluation.recommendation.llm_judge import SemanticJudge
from evaluation.recommendation.ranking_metrics import calculate_case_metrics
from evaluation.recommendation.schemas import CaseEvaluation, ConversationEvalCase, MovieEvaluation


class ConversationEvaluator:
    """Judge each recommended movie on hard constraints, semantic fit and taste independently."""

    def __init__(self, judge: SemanticJudge, catalog: Catalog) -> None:
        self.judge = judge
        self.catalog = catalog

    def evaluate(self, case: ConversationEvalCase, recommended_movie_ids: Sequence[int],
                 *, k: int = 10) -> CaseEvaluation:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        ranked = list(dict.fromkeys(int(movie_id) for movie_id in recommended_movie_ids))[:k]
        movies = [self._evaluate_movie(case, movie_id, rank)
                  for rank, movie_id in enumerate(ranked, start=1)]
        return CaseEvaluation(
            case_id=case.case_id, user_id=case.user_id, requested_k=k,
            evaluated_count=len(movies), movies=movies,
            metrics=calculate_case_metrics(movies, k=k),
        )

    def _evaluate_movie(self, case: ConversationEvalCase, movie_id: int, rank: int) -> MovieEvaluation:
        movie = self.catalog.movies.get(movie_id)
        test_rating = self.catalog.test_ratings.get((case.user_id, movie_id)) if movie else None
        return MovieEvaluation(
            rank=rank, movie_id=movie_id, title=movie["title"] if movie else None,
            valid_movie_id=movie is not None,
            hard_constraints=check_hard_constraints(case, movie, self.catalog),
            semantic=check_semantic(case, movie, self.judge),
            taste=check_taste(test_rating),
            test_rating=test_rating,
            test_rating_label=None if test_rating is None else rating_label(test_rating),
        )


def load_cases(path: str | Path) -> list[ConversationEvalCase]:
    raw_cases = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ConversationEvalCase.model_validate(raw_case) for raw_case in raw_cases]
