from __future__ import annotations

import unittest

from evaluation.explanation_groundedness.evaluator import leaked_terms
from evaluation.explanation_groundedness.evidence import context_turns, evidence_for_turn, not_found_queries


def turn(index: int, **extra) -> dict:
    return {"user": f"q{index}", "answer": f"a{index}",
            "tool_calls": [{"name": "recommend_for_user", "arguments": {}, "output": {"results": [f"m{index}"]}}],
            **extra}


TRANSCRIPT = {
    "user_id": 30,
    "turns": [
        turn(1, replayed=True),
        {"user": "q2", "error": "RuntimeError: boom"},
        {"user": "What about Matrix?", "answer": "I couldn't find it.",
         "tool_calls": [{"name": "get_peer_opinion", "arguments": {"movie_title": "Matrix"},
                         "output": {"found": False, "query": "Matrix"}}]},
    ],
}


class GroundednessCheckTests(unittest.TestCase):
    def test_leaked_terms(self) -> None:
        self.assertEqual(leaked_terms("Content match in the 99.5th percentile; the model is sure."),
                         ["model", "percentile"])
        self.assertEqual(leaked_terms("About 9 in 10 movies judged like this one were liked."), [])
        self.assertEqual(leaked_terms("Your rating_count: 130; movie ID 50; content similarity 0.56"),
                         ["movie id 50", "rating_count", "similarity 0.56"])
        self.assertEqual(leaked_terms("Crime sits above your baseline; individual scores ranged 3-5."), [])

    def test_evidence_is_earlier_context_then_this_turn(self) -> None:
        items = evidence_for_turn(TRANSCRIPT, 2)
        self.assertEqual([item.id for item in items],
                         ["H1.user", "H1.recommend_for_user#0", "H1.answer", "T.user", "T.get_peer_opinion#0"])
        self.assertEqual(items[1].content["output"], {"results": ["m1"]})  # replayed turns are context too
        self.assertEqual(not_found_queries(items), ["Matrix"])

    def test_not_found_counts_only_this_turn(self) -> None:
        transcript = {"user_id": 30, "turns": [*TRANSCRIPT["turns"], turn(4)]}
        self.assertEqual(not_found_queries(evidence_for_turn(transcript, 3)), [])

    def test_compacted_turns_are_replaced_by_the_summary(self) -> None:
        turns = [turn(1), turn(2), turn(3), turn(4),
                 turn(5, history_compaction={"summary": "S1", "summarized_turns": 1}), turn(6),
                 turn(7, history_compaction={"summary": "S2", "summarized_turns": 2}), turn(8)]
        self.assertEqual(context_turns(turns, 4), ([0, 2, 3], "S1"))
        self.assertEqual(context_turns(turns, 6), ([0, 4, 5], "S2"))
        self.assertEqual(context_turns(turns, 7), ([0, 4, 5, 6], "S2"))
        ids = [item.id for item in evidence_for_turn({"user_id": 1, "turns": turns}, 4)]
        self.assertEqual(ids[:4], ["H1.user", "H1.recommend_for_user#0", "H1.answer", "S.summary"])
        self.assertNotIn("H2.user", ids)


if __name__ == "__main__":
    unittest.main()
