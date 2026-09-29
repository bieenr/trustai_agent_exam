from __future__ import annotations

import asyncio
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from trusted_ai.history import SUMMARY_HEADER, compact, is_summary, last_input_tokens, split_turns


def turn(index: int, with_tool: bool = True) -> list:
    messages = [HumanMessage(content=f"q{index}")]
    if with_tool:
        messages += [AIMessage(content="", tool_calls=[{"name": "recommend_for_user", "args": {}, "id": f"c{index}"}]),
                     ToolMessage(content="{}", tool_call_id=f"c{index}", name="recommend_for_user")]
    messages.append(AIMessage(content=f"a{index}", usage_metadata={
        "input_tokens": 1000 * index, "output_tokens": 10, "total_tokens": 1000 * index + 10}))
    return messages


def run(messages: list, keep: int = 2):
    seen = []

    async def summarize(previous, folded):
        seen.append((previous, [message.content for message in folded if isinstance(message, HumanMessage)]))
        return f"summary of {len(folded)}"

    compacted, info = asyncio.run(compact(messages, summarize, keep))
    return compacted, info, seen


class CompactTest(unittest.TestCase):
    def test_keeps_first_and_last_two_turns_around_one_summary(self) -> None:
        history = [message for index in range(1, 6) for message in turn(index)]
        compacted, info, seen = run(history)
        self.assertEqual(seen, [(None, ["q2", "q3"])])
        self.assertEqual(info["summarized_turns"], 2)
        turns, summary = split_turns(compacted)
        self.assertEqual([t[0].content for t in turns], ["q1", "q4", "q5"])
        self.assertEqual(summary, "summary of 8")
        self.assertTrue(is_summary(compacted[4]))  # right after the whole first turn
        self.assertTrue(compacted[4].content.startswith(SUMMARY_HEADER))

    def test_never_splits_a_tool_call_from_its_result(self) -> None:
        history = [message for index in range(1, 6) for message in turn(index)]
        compacted, _, _ = run(history)
        for position, message in enumerate(compacted):
            if isinstance(message, ToolMessage):
                self.assertTrue(compacted[position - 1].tool_calls)

    def test_second_compaction_folds_the_previous_summary(self) -> None:
        history = [message for index in range(1, 6) for message in turn(index)]
        compacted, _, _ = run(history)
        compacted += turn(6) + turn(7, with_tool=False)
        again, _, seen = run(compacted)
        self.assertEqual(seen, [("summary of 8", ["q4", "q5"])])
        self.assertEqual([t[0].content for t in split_turns(again)[0]], ["q1", "q6", "q7"])

    def test_nothing_to_fold_leaves_history_unchanged(self) -> None:
        history = [message for index in range(1, 4) for message in turn(index)]
        compacted, info, seen = run(history)
        self.assertIs(compacted, history)
        self.assertIsNone(info)
        self.assertEqual(seen, [])

    def test_context_size_is_the_last_calls_input_tokens(self) -> None:
        history = [message for index in range(1, 4) for message in turn(index)]
        self.assertEqual(last_input_tokens(history), 3000)
        self.assertIsNone(last_input_tokens([HumanMessage(content="hi")]))


if __name__ == "__main__":
    unittest.main()
