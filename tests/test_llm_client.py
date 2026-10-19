"""Validate the LLM client provider wiring and offline mock (no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.llm.config import LLMConfig
from photon_tuner.llm.clients import (
    MockLLMClient,
    OpenAICompatibleClient,
    create_llm_client,
    resolve_api_key,
)


class TestLLMClient(unittest.TestCase):
    def test_mock_provider(self):
        client = create_llm_client(LLMConfig(provider="mock"))
        self.assertIsInstance(client, MockLLMClient)

    def test_paratera_preset_fills_model_and_base_url(self):
        client = create_llm_client(LLMConfig(provider="paratera"))
        self.assertIsInstance(client, OpenAICompatibleClient)
        self.assertEqual(client.model, "DeepSeek-V4-Pro-0813")
        self.assertEqual(client.base_url, "https://llmapi.paratera.com/v1")

    def test_explicit_config_overrides_preset(self):
        client = create_llm_client(
            LLMConfig(provider="paratera", model="custom-model", base_url="http://127.0.0.1:9999/v1")
        )
        self.assertEqual(client.model, "custom-model")
        self.assertEqual(client.base_url, "http://127.0.0.1:9999/v1")

    def test_resolve_api_key_prefers_explicit(self):
        key = resolve_api_key(LLMConfig(api_key="sk-explicit"))
        self.assertEqual(key, "sk-explicit")

    def test_mock_solution_is_schema_valid(self):
        result = MockLLMClient().complete_json("", {"task": "solution_generation"})
        self.assertIsInstance(result.get("actions"), list)
        self.assertIsInstance(result.get("link_edits"), list)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            create_llm_client(LLMConfig(provider="does-not-exist"))


if __name__ == "__main__":
    unittest.main()
