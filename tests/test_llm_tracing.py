"""Tests for the LLM trace wiring (previously zero-coverage dead code)."""

import json
import tempfile
import unittest
from pathlib import Path

from photon_tuner.llm.config import LLMConfig
from photon_tuner.llm.clients import MockLLMClient, OpenAICompatibleClient
from photon_tuner.llm.tracing import LLMTraceRecorder


def _messages(task: str = "react_tool_use", agent: str = "model_level"):
    return [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": json.dumps({"task": task, "agent_name": agent})},
    ]


class _StubOpenAI(OpenAICompatibleClient):
    """OpenAI-compatible client with the network call stubbed out."""

    def __init__(self) -> None:
        self.config = LLMConfig(provider="openai-compatible")
        self.retry_attempts = 0
        self.retry_temperature = 0.1
        self.base_url = "http://stub.invalid"
        self.model = "stub-model"

    def _chat_completion(self, payload):  # type: ignore[override]
        body = {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
        }
        return '{"ok": true}', body


class _FailingOpenAI(_StubOpenAI):
    def _chat_completion(self, payload):  # type: ignore[override]
        raise RuntimeError("simulated transport failure")


class TestNoRecorder(unittest.TestCase):
    def test_client_without_recorder_is_silent(self):
        client = MockLLMClient()
        self.assertIsNone(getattr(client, "recorder", None))
        self.assertEqual(client.complete_json_messages(_messages())["action"], "finish")


class TestRecorderWiring(unittest.TestCase):
    def test_mock_call_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = LLMTraceRecorder(root)
            client = MockLLMClient()
            client.recorder = recorder
            client.complete_json_messages(_messages())

            calls = json.loads((root / "prompt_calls.json").read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["status"], "ok")
            self.assertEqual(calls[0]["agent_name"], "model_level")
            self.assertTrue((root / calls[0]["input_json"]).exists())
            self.assertTrue((root / calls[0]["output_json"]).exists())
            # The readable summaries are produced too.
            self.assertTrue((root / "system_prompts.md").exists())
            self.assertTrue((root / "view_md" / "model_level_agent_dialogue.md").exists())

    def test_openai_call_records_full_response_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _StubOpenAI()
            client.recorder = LLMTraceRecorder(root)
            result = client.complete_json_messages(_messages(task="model_level_analysis"))

            self.assertEqual(result, {"ok": True})
            calls = json.loads((root / "prompt_calls.json").read_text(encoding="utf-8"))
            self.assertEqual(calls[0]["status"], "ok")
            self.assertEqual(calls[0]["usage"]["prompt_tokens"], 11)

            raw = json.loads((root / calls[0]["output_json"]).read_text(encoding="utf-8"))
            self.assertIn("usage", raw["response"])
            self.assertIn("choices", raw["response"])

    def test_failed_call_is_recorded_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FailingOpenAI()
            client.recorder = LLMTraceRecorder(root)
            with self.assertRaises(RuntimeError):
                client.complete_json_messages(_messages())

            calls = json.loads((root / "prompt_calls.json").read_text(encoding="utf-8"))
            self.assertEqual(calls[0]["status"], "error")
            self.assertIn("simulated transport failure", json.dumps(calls[0]["error"]))


class TestTracingNeverBreaksARun(unittest.TestCase):
    def test_broken_recorder_does_not_propagate(self):
        class _ExplodingRecorder:
            def begin_call(self, request_body):
                raise RuntimeError("recorder exploded")

            def end_call(self, *args, **kwargs):
                raise RuntimeError("recorder exploded")

        client = MockLLMClient()
        client.recorder = _ExplodingRecorder()
        # Must still return the model answer despite the recorder failing.
        self.assertEqual(client.complete_json_messages(_messages())["action"], "finish")


if __name__ == "__main__":
    unittest.main()
