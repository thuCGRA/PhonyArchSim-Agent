"""多目标 Pareto 归档的测试（吞吐归一化指标）。

跨范式前沿用吞吐归一化轴：能耗→``pJ_per_mac``（每 MAC）、面积→
``area_um2_per_mac_per_s``（每单位吞吐 MAC/s），使并行权重阵列（单输出向量、器件少
但吞吐低）与矩阵阵列（器件多但吞吐高）在同一效率轴上可比。
"""

import unittest

from photon_tuner.core.pareto import DEFAULT_MINIMIZE_KEYS, ParetoArchive, dominates


class TestDominates(unittest.TestCase):
    def test_strict_domination(self):
        a = {"pJ_per_mac": 1, "area_um2_per_mac_per_s": 2}
        b = {"pJ_per_mac": 2, "area_um2_per_mac_per_s": 3}
        self.assertTrue(dominates(a, b, DEFAULT_MINIMIZE_KEYS))
        self.assertFalse(dominates(b, a, DEFAULT_MINIMIZE_KEYS))

    def test_equal_is_not_domination(self):
        a = {"pJ_per_mac": 1, "area_um2_per_mac_per_s": 2}
        self.assertFalse(dominates(a, dict(a), DEFAULT_MINIMIZE_KEYS))

    def test_tradeoff_is_not_domination(self):
        a = {"pJ_per_mac": 1, "area_um2_per_mac_per_s": 3}  # 能量更优、面积更差
        b = {"pJ_per_mac": 2, "area_um2_per_mac_per_s": 2}
        self.assertFalse(dominates(a, b, DEFAULT_MINIMIZE_KEYS))
        self.assertFalse(dominates(b, a, DEFAULT_MINIMIZE_KEYS))

    def test_missing_metric_is_ignored(self):
        # 未报告的轴被丢弃、从不惩罚（禁制造规则）。
        a = {"pJ_per_mac": 1}
        b = {"pJ_per_mac": 2, "area_um2_per_mac_per_s": 999}
        self.assertTrue(dominates(a, b, DEFAULT_MINIMIZE_KEYS))

    def test_throughput_normalization_not_fooled_by_raw_area(self):
        # 吞吐归一化的核心语义：原始面积小的向量阵列（吞吐低）不再假性支配原始面积大的
        # 矩阵阵列。area_per_throughput = area * latency / macs（固定工作负载 macs 相同）。
        vector = {"pJ_per_mac": 1.0, "area_um2_per_mac_per_s": 1.0, "latency_s": 10.0}
        matrix = {"pJ_per_mac": 1.0, "area_um2_per_mac_per_s": 0.5, "latency_s": 1.0}
        # 矩阵的归一化面积效率更优（0.5 < 1.0）→ 矩阵支配向量，反之不成立。
        self.assertTrue(dominates(matrix, vector, DEFAULT_MINIMIZE_KEYS))
        self.assertFalse(dominates(vector, matrix, DEFAULT_MINIMIZE_KEYS))


class TestParetoArchive(unittest.TestCase):
    def test_add_non_dominated_and_prune(self):
        arch = ParetoArchive()
        self.assertTrue(arch.add({"pJ_per_mac": 1, "area_um2_per_mac_per_s": 2}))
        # 被第一个支配 → 拒绝。
        self.assertFalse(arch.add({"pJ_per_mac": 2, "area_um2_per_mac_per_s": 3}))
        # 非支配的权衡点 → 接受（前沿增长）。
        self.assertTrue(arch.add({"pJ_per_mac": 0.5, "area_um2_per_mac_per_s": 4}))
        # 支配第一个 → 将其剪除。
        self.assertTrue(arch.add({"pJ_per_mac": 0.8, "area_um2_per_mac_per_s": 1.8}))
        self.assertEqual(len(arch.frontier()), 2)

    def test_tag_filter(self):
        arch = ParetoArchive()
        arch.add({"pJ_per_mac": 1, "area_um2_per_mac_per_s": 2}, {"strategy": "best", "core_kind": "a"})
        arch.add({"pJ_per_mac": 0.5, "area_um2_per_mac_per_s": 4}, {"strategy": "next", "core_kind": "a"})
        self.assertEqual(len(arch.frontier("strategy", "best")), 1)
        self.assertEqual(len(arch.frontier("strategy", "next")), 1)
        self.assertEqual(len(arch.frontier("core_kind", "a")), 2)
        self.assertEqual(len(arch.frontier("core_kind", "b")), 0)

    def test_round_trip(self):
        arch = ParetoArchive()
        arch.add({"pJ_per_mac": 1, "area_um2_per_mac_per_s": 2}, {"iteration": 0, "strategy": "best"})
        arch.add({"pJ_per_mac": 0.5, "area_um2_per_mac_per_s": 4}, {"iteration": 1, "strategy": "next"})
        restored = ParetoArchive.from_dict(arch.to_dict())
        self.assertEqual(restored.minimize_keys, arch.minimize_keys)
        self.assertEqual(restored.points, arch.points)


if __name__ == "__main__":
    unittest.main()
