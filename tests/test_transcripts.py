from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from trusted_ai import transcripts
from trusted_ai.schemas import AgentResponse, ToolEvent


class TranscriptTest(unittest.TestCase):
    def build(self) -> dict:
        transcript = transcripts.new_transcript(30, "abc123456", "model", {"max_results": 5}, "PROMPT",
                                                case_id="case_a", intent="why_movie")
        transcripts.add_replayed_turns(transcript, [{"user": "hi", "answer": "hello"}])
        response = AgentResponse(answer="Try Heat.", tool_events=[
            ToolEvent(name="recommend_for_user", arguments={"top_n": 1}, result_count=1,
                      output={"results": [{"movie_id": 6, "title": "Heat"}]})])
        transcripts.add_turn(transcript, "What now?", response, latency_s=1.5)
        transcripts.add_turn(transcript, "And?", error="RuntimeError: boom")
        return transcript

    def test_turns_rebuild_the_session_history(self) -> None:
        messages = transcripts.to_messages(self.build()["turns"])
        self.assertEqual([type(message) for message in messages],
                         [HumanMessage, AIMessage, HumanMessage, AIMessage, ToolMessage, AIMessage])
        call, result = messages[3].tool_calls[0], messages[4]
        self.assertEqual((call["name"], call["args"]), ("recommend_for_user", {"top_n": 1}))
        self.assertEqual(result.tool_call_id, call["id"])
        self.assertEqual(json.loads(result.content), {"results": [{"movie_id": 6, "title": "Heat"}]})
        self.assertEqual(messages[-1].content, "Try Heat.")  # the failed turn left nothing behind

    def test_save_and_load_by_case_id(self) -> None:
        transcript = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = transcripts.save(transcript, Path(directory))
            self.assertEqual(path.name, "case_a.json")
            [(loaded_path, loaded)] = transcripts.load([Path(directory)])
        self.assertEqual(loaded_path, path)
        self.assertEqual([turn.get("replayed", False) for turn in loaded["turns"]], [True, False, False])
        self.assertEqual((loaded["intent"], loaded["turns"][1]["latency_s"]), ("why_movie", 1.5))


if __name__ == "__main__":
    unittest.main()
