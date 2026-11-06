"""``photon_tuner.toolbox.tool_cache``（report 工具结果缓存）的测试。"""

import unittest

from photon_tuner.toolbox.tool_cache import ToolResultCache


class TestToolResultCache(unittest.TestCase):
    def test_miss_then_hit(self):
        cache = ToolResultCache()
        key = cache.key("summarize_metrics", {"arch": {"a": 1}}, {"param": "x"})
        self.assertIsNone(cache.get(key))
        cache.put(key, "RESULT")
        self.assertEqual(cache.get(key), "RESULT")

    def test_same_input_same_key(self):
        cache = ToolResultCache()
        k1 = cache.key("summarize_metrics", {"a": 1}, {"p": "x"})
        k2 = cache.key("summarize_metrics", {"a": 1}, {"p": "x"})
        self.assertEqual(k1, k2)

    def test_different_hardware_or_args_differ(self):
        cache = ToolResultCache()
        base = cache.key("summarize_metrics", {"a": 1}, {"p": "x"})
        self.assertNotEqual(base, cache.key("summarize_metrics", {"a": 2}, {"p": "x"}))
        self.assertNotEqual(base, cache.key("summarize_metrics", {"a": 1}, {"p": "y"}))
        self.assertNotEqual(base, cache.key("summarize_area_breakdown", {"a": 1}, {"p": "x"}))

    def test_fifo_eviction(self):
        cache = ToolResultCache(max_entries=2)
        keys = [cache.key("t", {"i": i}, {}) for i in range(3)]
        for i, key in enumerate(keys):
            cache.put(key, i)
        self.assertIsNone(cache.get(keys[0]))  # 最旧的被淘汰
        self.assertIsNotNone(cache.get(keys[2]))


if __name__ == "__main__":
    unittest.main()
