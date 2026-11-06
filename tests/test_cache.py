"""``photon_tuner.llm.cache``（进程内响应缓存）的测试。"""

import unittest

from photon_tuner.llm.cache import ResponseCache


class TestResponseCache(unittest.TestCase):
    def test_miss_then_hit(self):
        cache = ResponseCache()
        key = cache.key("model_agent", {"task": "x"})
        self.assertIsNone(cache.get(key))
        cache.put(key, "OUT", ["t1"], None)
        hit = cache.get(key)
        self.assertEqual(hit["out"], "OUT")
        self.assertEqual(hit["transcript"], ["t1"])
        self.assertIsNone(hit["error"])

    def test_different_payload_misses(self):
        cache = ResponseCache()
        cache.put(cache.key("m", {"a": 1}), "OUT", [], None)
        self.assertIsNone(cache.get(cache.key("m", {"a": 2})))

    def test_disabled_cache_never_hits(self):
        cache = ResponseCache(enabled=False)
        key = cache.key("m", {"a": 1})
        cache.put(key, "OUT", [], None)
        self.assertIsNone(cache.get(key))

    def test_fifo_eviction(self):
        cache = ResponseCache(max_entries=2)
        for i in range(3):
            cache.put(cache.key("m", {"i": i}), i, [], None)
        self.assertIsNone(cache.get(cache.key("m", {"i": 0})))  # 最旧的被淘汰
        self.assertIsNotNone(cache.get(cache.key("m", {"i": 2})))


if __name__ == "__main__":
    unittest.main()
