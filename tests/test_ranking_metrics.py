from __future__ import annotations

import math
import unittest

from evaluation.recommendation import (
    CaseEvaluation,
    CriterionResult,
    MovieEvaluation,
    calculate_ranking_metrics,
    catalog_coverage,
)
from evaluation.recommendation.ranking_metrics import calculate_case_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_binary_metrics_follow_recommendation_order(self) -> None:
        metrics = calculate_ranking_metrics([30, 10, 99], [False, True, True], k=3)

        self.assertEqual(metrics.hits, (10, 99))
        self.assertEqual(metrics.judged_count, 3)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
        expected_idcg = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(metrics.ndcg, expected_dcg / expected_idcg)
        self.assertEqual(metrics.hit_rate, 1.0)

    def test_unjudged_movies_are_dropped_before_ranking(self) -> None:
        metrics = calculate_ranking_metrics([1, 2, 3], [None, True, None], k=3)

        self.assertEqual(metrics.judged_count, 1)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.ndcg, 1.0)

    def test_k_truncates_before_dropping(self) -> None:
        metrics = calculate_ranking_metrics([1, 2, 3], [None, False, True], k=2)

        self.assertEqual(metrics.judged_count, 1)
        self.assertEqual(metrics.hits, ())

    def test_empty_recommendations_have_zero_metrics(self) -> None:
        metrics = calculate_ranking_metrics([], [], k=10)

        self.assertEqual(metrics.judged_count, 0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.ndcg, 0.0)
        self.assertEqual(metrics.hit_rate, 0.0)

    def test_hard_constraint_failure_is_not_a_dislike(self) -> None:
        failed = MovieEvaluation(
            rank=1, movie_id=10, valid_movie_id=True,
            hard_constraints=CriterionResult(passed=False, reasons=["forbidden genre"]),
            taste=CriterionResult(passed=True), test_rating=4.5, test_rating_label="positive",
        )

        metrics = calculate_case_metrics([failed], k=10)

        self.assertEqual(metrics.hard_constraints.precision, 0.0)
        self.assertEqual(metrics.taste.precision, 1.0)
        self.assertEqual(metrics.taste.dislike_rate, 0.0)
        self.assertIsNone(metrics.semantic)

    def test_catalog_coverage_uses_evaluated_movies(self) -> None:
        movies = [
            MovieEvaluation(rank=rank, movie_id=movie_id, valid_movie_id=True,
                            hard_constraints=CriterionResult(passed=True))
            for rank, movie_id in enumerate([10, 20, 999], start=1)
        ]
        evaluation = CaseEvaluation(
            case_id="case-1", user_id=1, requested_k=3, evaluated_count=3,
            movies=movies, metrics=calculate_case_metrics(movies, k=3),
        )

        self.assertEqual(catalog_coverage([evaluation], {10, 20, 30, 40}), 0.5)

    def test_k_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be greater than zero"):
            calculate_ranking_metrics([], [], k=0)


if __name__ == "__main__":
    unittest.main()
