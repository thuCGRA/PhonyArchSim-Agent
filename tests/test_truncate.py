"""``photon_tuner.common.truncate``（上下文长度预算）的测试。"""

import json
import unittest

from photon_tuner.common.truncate import DEFAULT_MAX_PAYLOAD_CHARS, fit_payload, truncate_string


class TestTruncateString(unittest.TestCase):
    def test_short_string_untouched(self):
        self.assertEqual(truncate_string("hello", 100), "hello")

    def test_long_string_keeps_head_and_tail(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 个字符
        out = truncate_string(text, 100)
        self.assertLessEqual(len(out), 100)
        self.assertTrue(out.startswith("abcdefghij"))
        self.assertTrue(out.endswith("yz"))
        self.assertIn("truncated", out)

    def test_tiny_budget_never_exceeds(self):
        out = truncate_string("x" * 500, 10)
        self.assertLessEqual(len(out), 10)


class TestFitPayload(unittest.TestCase):
    def test_flat_payload_preserved(self):
        payload = {"task": "model_level_analysis", "iteration": 1, "metrics": {"latency_s": 0.1}}
        self.assertEqual(fit_payload(payload), payload)

    def test_long_strings_truncated_within_budget(self):
        payload = {"task": "x", "hardware_summary": "long" * 5000}
        out = fit_payload(payload, max_chars=2000)
        self.assertLess(len(json.dumps(out, ensure_ascii=False)), 2000)
        self.assertEqual(out["task"], "x")  # 键被保留

    def test_long_lists_capped(self):
        payload = {"cases": [{"id": i, "detail": "d" * 500} for i in range(200)]}
        out = fit_payload(payload, max_chars=3000)
        self.assertLess(len(json.dumps(out, ensure_ascii=False)), 3000)
        self.assertLess(len(out["cases"]), 200)  # 列表被截断

    def test_keys_never_dropped(self):
        payload = {f"k{i}": "v" * 300 for i in range(30)}
        out = fit_payload(payload, max_chars=1500)
        self.assertEqual(set(out.keys()), set(payload.keys()))

    def test_default_budget_serializes_within_limit(self):
        payload = {"big": "x" * 100_000}
        out = fit_payload(payload)
        self.assertLessEqual(len(json.dumps(out, ensure_ascii=False)), DEFAULT_MAX_PAYLOAD_CHARS)


if __name__ == "__main__":
    unittest.main()
