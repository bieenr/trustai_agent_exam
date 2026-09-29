from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.recommendation import (
    ConversationEvalCase,
    ConversationEvaluator,
    JudgeDecision,
    load_catalog,
)


class FakeJudge:
    """Rejects movie 20 semantically and records every movie it was asked about."""

    def __init__(self) -> None:
        self.movie_ids: list[int] = []

    def semantic(self, *, movie: dict[str, object], **_: object) -> JudgeDecision:
        movie_id = int(movie["movie_id"])
        self.movie_ids.append(movie_id)
        return JudgeDecision(pass_=movie_id != 20, reason="judged")


def make_case(*, semantic: bool) -> ConversationEvalCase:
    return ConversationEvalCase.model_validate({
        "case_id": "flow",
        "user_id": 1,
        "conversation": [{"role": "user", "content": "No horror please"}],
        "expected": {
            "hard_constraints": {"must_not_have_genres": ["Horror"], "max_year": 2020},
            "semantic_requirement": semantic,
        },
    })


class ConversationEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        data_dir = Path(temp_dir.name)
        split_dir = data_dir / "ranking_split"
        split_dir.mkdir(parents=True)
        (data_dir / "movies_with_plots.csv").write_text(
            "movieId,title,year,genres,plot\n"
            "10,Positive,2000,Drama,Plot\n"
            "20,Unrated,2001,Drama,Plot\n"
            "30,Negative Horror,2002,Horror,Plot\n",
            encoding="utf-8",
        )
        (split_dir / "ratings_train.csv").write_text(
            "userId,movieId,timestamp\n1,99,1000000000\n", encoding="utf-8"
        )
        (split_dir / "ratings_test.csv").write_text(
            "userId,movieId,rating\n1,10,4.5\n1,30,2.0\n", encoding="utf-8"
        )
        self.catalog = load_catalog(data_dir)

    def test_criteria_are_judged_independently(self) -> None:
        judge = FakeJudge()
        result = ConversationEvaluator(judge, self.catalog).evaluate(
            make_case(semantic=True), [10, 20, 20, 30, 777], k=10
        )

        self.assertEqual([m.movie_id for m in result.movies], [10, 20, 30, 777])
        # A hard-constraint failure (30) is still sent to the semantic judge; unknown IDs are not.
        self.assertEqual(judge.movie_ids, [10, 20, 30])
        unknown = result.movies[3]
        self.assertFalse(unknown.valid_movie_id)
        self.assertFalse(unknown.hard_constraints.passed)
        self.assertFalse(unknown.semantic.passed)
        self.assertIsNone(unknown.taste)

        metrics = result.metrics
        self.assertEqual(metrics.hard_constraints.hits, (10, 20))
        self.assertAlmostEqual(metrics.hard_constraints.precision, 2 / 4)
        self.assertEqual(metrics.semantic.hits, (10, 30))
        self.assertEqual(metrics.semantic.judged_count, 4)
        # Taste only counts movies with a held-out rating.
        self.assertEqual(metrics.taste.judged_count, 2)
        self.assertEqual(metrics.taste.hits, (10,))
        self.assertAlmostEqual(metrics.taste.precision, 1 / 2)
        self.assertEqual(metrics.taste.negative_hits, (30,))
        self.assertAlmostEqual(metrics.taste.dislike_rate, 1 / 2)

    def test_disabled_semantic_and_unrated_movies_have_no_metrics(self) -> None:
        judge = FakeJudge()
        result = ConversationEvaluator(judge, self.catalog).evaluate(
            make_case(semantic=False), [20], k=10
        )

        self.assertEqual(judge.movie_ids, [])
        self.assertIsNone(result.movies[0].semantic)
        self.assertIsNone(result.metrics.semantic)
        self.assertIsNone(result.metrics.taste)
        self.assertEqual(result.metrics.hard_constraints.precision, 1.0)


if __name__ == "__main__":
    unittest.main()
