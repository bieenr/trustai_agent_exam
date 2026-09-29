from __future__ import annotations

from math import log2
from typing import Collection, Sequence

from evaluation.recommendation.schemas import (
    CaseEvaluation,
    CaseMetrics,
    CriterionResult,
    MovieEvaluation,
    RankingMetrics,
    TasteMetrics,
)


def calculate_ranking_metrics(movie_ids: Sequence[int], verdicts: Sequence[bool | None],
                              *, k: int) -> RankingMetrics:
    """Binary top-k metrics; movies with a None verdict are dropped before ranking."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    judged = [(movie_id, verdict) for movie_id, verdict in list(zip(movie_ids, verdicts))[:k]
              if verdict is not None]
    hits = tuple(movie_id for movie_id, verdict in judged if verdict)
    dcg = sum(1.0 / log2(rank + 2) for rank, (_, verdict) in enumerate(judged) if verdict)
    idcg = sum(1.0 / log2(rank + 2) for rank in range(len(hits)))
    return RankingMetrics(
        k=k, judged_count=len(judged), hits=hits,
        precision=len(hits) / len(judged) if judged else 0.0,
        ndcg=dcg / idcg if idcg else 0.0,
        hit_rate=float(bool(hits)),
    )


def calculate_case_metrics(movies: Sequence[MovieEvaluation], *, k: int) -> CaseMetrics:
    """Score each criterion independently; a criterion that judged nothing has no metrics."""
    movie_ids = [movie.movie_id for movie in movies]

    def verdicts(results: list[CriterionResult | None]) -> list[bool | None]:
        return [None if result is None else result.passed for result in results]

    hard = calculate_ranking_metrics(movie_ids, verdicts([m.hard_constraints for m in movies]), k=k)
    semantic = calculate_ranking_metrics(movie_ids, verdicts([m.semantic for m in movies]), k=k)
    taste = calculate_ranking_metrics(movie_ids, verdicts([m.taste for m in movies]), k=k)
    return CaseMetrics(
        hard_constraints=hard,
        semantic=semantic if semantic.judged_count else None,
        taste=_with_dislikes(taste, movies[:k]) if taste.judged_count else None,
    )


def catalog_coverage(evaluations: Sequence[CaseEvaluation],
                     catalog_movie_ids: Collection[int]) -> float:
    """Fraction of catalog movies appearing in evaluated recommendation lists."""
    catalog = {int(movie_id) for movie_id in catalog_movie_ids}
    if not catalog:
        return 0.0
    recommended = {
        movie.movie_id
        for evaluation in evaluations
        for movie in evaluation.movies[: evaluation.requested_k]
        if movie.movie_id in catalog
    }
    return len(recommended) / len(catalog)


def _with_dislikes(metrics: RankingMetrics, movies: Sequence[MovieEvaluation]) -> TasteMetrics:
    negative_hits = tuple(m.movie_id for m in movies if m.test_rating_label == "negative")
    return TasteMetrics(
        **metrics.model_dump(), negative_hits=negative_hits,
        dislike_rate=len(negative_hits) / metrics.judged_count,
        negative_hit_rate=float(bool(negative_hits)),
    )
