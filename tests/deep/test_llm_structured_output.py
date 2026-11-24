"""结构化输出传输方式的回归测试。

背景：``langchain_openai`` 自 0.3.0 起把 ``with_structured_output`` 的默认
``method`` 改成了 ``"json_schema"``，而部分兼容网关会静默忽略 ``response_format``
并返回 Markdown 散文，导致 ``ValidationError: Invalid JSON``、运行在第一个 agent
阶段即中止。``photon_tuner.llm.model.StructuredChat`` 因此默认改用
``function_calling``。

这里不发起任何网络调用：只验证 ``method`` 的注入/覆盖行为与配置校验。
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

try:  # 域测试环境缺 langchain_openai 时整体跳过
    from photon_tuner.llm.config import LLMConfig
    from photon_tuner.llm.model import StructuredChat, build_chat_model

    have_langchain = True
except Exception:  # pragma: no cover - 仅域测试环境
    have_langchain = False


@unittest.skipUnless(have_langchain, "langchain_openai is not installed")
class TestStructuredOutputMethod(unittest.TestCase):
    def _model(self, **overrides):
        cfg = LLMConfig(
            provider="cloudmist",
            model="deepseek-v4.1-flash",
            base_url="https://example.invalid/v1",
            **overrides,
        )
        return build_chat_model(cfg)

    def test_build_chat_model_returns_structured_chat(self):
        self.assertIsInstance(self._model(), StructuredChat)

    def test_config_defaults_to_function_calling(self):
        self.assertEqual(LLMConfig().structured_output_method, "function_calling")

    def test_config_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            LLMConfig(structured_output_method="nonsense")

    def test_with_structured_output_injects_function_calling(self):
        model = self._model()
        seen = {}

        def fake(self, schema=None, **kwargs):
            seen.update(kwargs)
            return "sentinel"

        original = StructuredChat.__mro__[1].with_structured_output
        StructuredChat.__mro__[1].with_structured_output = fake
        try:
            self.assertEqual(model.with_structured_output(object()), "sentinel")
        finally:
            StructuredChat.__mro__[1].with_structured_output = original

        self.assertEqual(seen.get("method"), "function_calling")

    def test_explicit_method_wins(self):
        model = self._model()
        seen = {}

        def fake(self, schema=None, **kwargs):
            seen.update(kwargs)
            return None

        original = StructuredChat.__mro__[1].with_structured_output
        StructuredChat.__mro__[1].with_structured_output = fake
        try:
            model.with_structured_output(object(), method="json_schema")
        finally:
            StructuredChat.__mro__[1].with_structured_output = original

        self.assertEqual(seen.get("method"), "json_schema")

    def test_config_override_reaches_the_model(self):
        model = self._model(structured_output_method="json_schema")
        self.assertEqual(model.structured_output_method, "json_schema")

    def test_official_flash_disables_thinking_for_forced_tool_choice(self):
        model = build_chat_model(LLMConfig(
            provider="deepseek-official",
            model="deepseek-flash",
            base_url="https://api.deepseek.com/v1",
        ))
        self.assertEqual(
            model.extra_body,
            {"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    unittest.main()
