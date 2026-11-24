"""规范单位换算的单元测试。

核心不变量：三项单位换算系数之积为 1，因此目标函数在换算前后**数值相同**。
这条性质是"换单位不改变任何结论"的前提，必须被测试钉住，而不是靠注释声明。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.core.units import (
    DISPLAY_UNITS,
    LEGACY_ALIASES,
    describe_units,
    si_mirror,
    to_display_metrics,
)


class TestConversion(unittest.TestCase):
    def test_known_values(self):
        raw = {
            "latency_s": 0.006635,      # s
            "energy_pj": 2.300e9,       # pJ
            "area_um2": 6.650e7,        # um^2
            "pJ_per_mac": 0.2235,       # pJ/MAC
        }
        out = to_display_metrics(raw)
        self.assertAlmostEqual(out["latency_ms"], 6.635, places=9)
        self.assertAlmostEqual(out["energy_mj"], 2.300, places=9)
        self.assertAlmostEqual(out["area_mm2"], 66.50, places=9)
        self.assertAlmostEqual(out["fJ_per_mac"], 223.5, places=9)

    def test_objective_is_invariant_under_conversion(self):
        """J = latency × energy_per_mac × area 在换算前后必须完全相同。"""
        raw = {"latency_s": 0.006635, "energy_pj": 2.3e9,
               "area_um2": 6.65e7, "pJ_per_mac": 0.2235}
        out = to_display_metrics(raw)
        before = raw["latency_s"] * raw["pJ_per_mac"] * raw["area_um2"]
        after = out["latency_ms"] * out["fJ_per_mac"] * out["area_mm2"]
        self.assertAlmostEqual(before / after, 1.0, places=9)

    def test_dimensionless_keys_pass_through(self):
        raw = {"total_macs": 1112638464.0, "core_insertion_loss": 6.22,
               "il_budget_penalty": 1.0, "weight_latency_s": 1.0}
        out = to_display_metrics(raw)
        self.assertEqual(out["total_macs"], 1112638464.0)
        self.assertEqual(out["core_insertion_loss"], 6.22)
        self.assertEqual(out["il_budget_penalty"], 1.0)
        # 权重是指数而非物理量：既不缩放、也不改名。``weight_<key>`` 由
        # metrics_with_objective 在换算之后生成，因此规范键天然就是新的。
        self.assertEqual(out["weight_latency_s"], 1.0)

    def test_throughput_and_area_per_throughput_are_scaled(self):
        raw = {"throughput_mac_per_s": 2.16e9, "area_um2_per_mac_per_s": 5.27e-8}
        out = to_display_metrics(raw)
        self.assertAlmostEqual(out["throughput_mac_per_ms"], 2.16e6, delta=1.0)
        self.assertAlmostEqual(out["area_mm2_per_mac_per_ms"], 5.27e-11, places=20)

    def test_si_mirror_round_trips(self):
        raw = {"latency_s": 0.5, "energy_pj": 1e9, "area_um2": 1e6}
        out = to_display_metrics(raw)
        mirror = si_mirror(out)
        for key, value in raw.items():
            # 反复乘除 1e3/1e-9 会引入 ~1e-16 的相对误差，用相对容差比较。
            self.assertAlmostEqual(mirror[key] / value, 1.0, places=12, msg=key)

    def test_non_finite_and_non_numeric_pass_through(self):
        out = to_display_metrics({"latency_s": float("inf"), "energy_pj": "n/a"})
        self.assertEqual(out["latency_ms"], float("inf"))
        self.assertEqual(out["energy_mj"], "n/a")

    def test_every_alias_has_a_unit_label(self):
        """规范键必须都有单位标签，否则报告里会出现无单位数字。"""
        for canonical in LEGACY_ALIASES.values():
            self.assertIn(canonical, DISPLAY_UNITS, canonical)

    def test_describe_units_returns_a_copy(self):
        units = describe_units()
        units["latency_ms"] = "tampered"
        self.assertEqual(describe_units()["latency_ms"], "ms")


if __name__ == "__main__":
    unittest.main()
