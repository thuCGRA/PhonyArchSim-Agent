"""真实变体 Pareto 前沿摘要（准则 2、选项 B）的测试。"""

import unittest
from pathlib import Path

from photon_tuner.core.device_frontier import summarize_device_frontier
from photon_tuner.core.search_space import default_hardware, discover_device_catalog

ROOT = Path(__file__).resolve().parents[1]
DEVICELIB = ROOT / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"


def _hw(work_freq, in_bit=8, out_bit=8):
    hw = default_hardware()
    hw["arch"]["optical_core"]["work_freq"] = work_freq
    hw["arch"]["optical_core"]["precision"]["in_bit"] = in_bit
    hw["arch"]["optical_core"]["precision"]["out_bit"] = out_bit
    return hw


SYNTHETIC = {
    "dac": {
        "DAC_dom": {"cfgs": {"power": 10, "area": 100, "sample_rate": 50, "prec": 9}},
        "DAC_dominated": {"cfgs": {"power": 20, "area": 200, "sample_rate": 50, "prec": 9}},
        "DAC_tradeoff": {"cfgs": {"power": 5, "area": 300, "sample_rate": 50, "prec": 9}},
        "DAC_slow": {"cfgs": {"power": 3, "area": 50, "sample_rate": 14, "prec": 8}},
    },
}


class TestDeviceFrontier(unittest.TestCase):
    def test_frontier_and_coupling_filter(self):
        # work_freq=20 GHz：DAC_slow（14 GS/s）被耦合排除；其余中 DAC_dominated
        # 在功耗/面积上被 DAC_dom Pareto 支配。
        out = summarize_device_frontier(SYNTHETIC, _hw(20))["dac"]
        self.assertEqual(sorted(out["compatible"]), ["DAC_dom", "DAC_dominated", "DAC_tradeoff"])
        self.assertEqual(out["excluded_by_coupling"], ["DAC_slow"])
        self.assertEqual(out["frontier_power_area"], ["DAC_dom", "DAC_tradeoff"])
        self.assertEqual(out["dominated_power_area"], ["DAC_dominated"])
        self.assertEqual(out["non_comparable"], [])

    def test_slow_device_rejoins_at_low_work_freq(self):
        # work_freq=5 GHz：无器件被耦合排除；DAC_slow 此时出现。
        out = summarize_device_frontier(SYNTHETIC, _hw(5))["dac"]
        self.assertEqual(out["excluded_by_coupling"], [])
        self.assertIn("DAC_slow", out["compatible"])

    def test_missing_power_area_is_non_comparable(self):
        cat = {"tia": {"TIA_1": {"cfgs": {}}}}
        out = summarize_device_frontier(cat, _hw(5))["tia"]
        self.assertEqual(out["compatible"], ["TIA_1"])
        self.assertEqual(out["non_comparable"], ["TIA_1"])
        self.assertEqual(out["frontier_power_area"], [])

    def test_real_catalog_high_work_freq_filters_slow_adc(self):
        catalog = discover_device_catalog(str(DEVICELIB))
        out = summarize_device_frontier(catalog, _hw(20))["adc"]
        # 20 GHz 下 10/4.8/2 GS/s 三个低速 ADC 均被耦合排除；30 GS/s 与 90 GS/s 留下，
        # 其中 ADC_SAR_30GS(60mW/2500um2) 在两个轴上全面优于 90 GS/s 变体。
        self.assertEqual(
            out["excluded_by_coupling"],
            ["ADC_SAR_1", "ADC_SAR_2GS_LP", "ADC_SAR_4P8GS_LP"],
        )
        self.assertEqual(out["compatible"], ["ADC_SAR_30GS", "ADC_SAR_90GS"])
        self.assertEqual(out["frontier_power_area"], ["ADC_SAR_30GS"])

    def test_real_catalog_frontier_uses_reported_values(self):
        catalog = discover_device_catalog(str(DEVICELIB))
        out = summarize_device_frontier(catalog, _hw(5))["dac"]
        # 所有报告的 tradeoff 值均为真实值：DAC_1 power=169 mW、DAC_3 speed=50 GHz。
        self.assertAlmostEqual(out["tradeoffs"]["DAC_1"]["energy_cost"], 169.0)
        self.assertEqual(out["tradeoffs"]["DAC_1"]["energy_unit"], "mW")
        self.assertAlmostEqual(out["tradeoffs"]["DAC_3"]["speed_GHz"], 50.0)
        self.assertIn("prec_bit", out["tradeoffs"]["DAC_3"])

    def test_energy_axis_is_type_specific(self):
        # 光子调制器/相移器的 static_power 恒为 0，必须改用物理上正确的能量度量，
        # 否则整类器件退化到"只比面积"、前沿塌成单点。
        catalog = discover_device_catalog(str(DEVICELIB))
        out = summarize_device_frontier(catalog, _hw(5))

        mzm = out["mzm"]["tradeoffs"]["LT_MZM"]
        self.assertEqual(mzm["energy_unit"], "fJ/bit")
        self.assertEqual(mzm["energy_field"], "efficiency")
        self.assertAlmostEqual(mzm["energy_cost"], 450.0)

        ps = out["phase_shifter"]["tradeoffs"]["IHO_PS"]
        self.assertEqual(ps["energy_unit"], "mW")
        self.assertEqual(ps["energy_field"], "P_pi")
        self.assertAlmostEqual(ps["energy_cost"], 9.6)

    def test_dynamic_power_now_makes_tia_and_pd_comparable(self):
        # tia 只记录 dynamic_power；photodetector 的 LT_PD 同理。修轴后两者都进入
        # 可比集合，从而真正具备权衡（此前 tia 前沿为空）。
        catalog = discover_device_catalog(str(DEVICELIB))
        out = summarize_device_frontier(catalog, _hw(5))

        tia = out["tia"]
        self.assertEqual(tia["non_comparable"], [])
        self.assertEqual(tia["frontier_power_area"], ["TIA_1", "TIA_LP"])
        self.assertAlmostEqual(tia["tradeoffs"]["TIA_HB"]["energy_cost"], 9.0)

        pd = out["photodetector"]
        self.assertIn("LT_PD", pd["frontier_power_area"])
        self.assertAlmostEqual(pd["tradeoffs"]["LT_PD"]["energy_cost"], 1.1)

    def test_photonics_types_now_have_multi_point_frontiers(self):
        catalog = discover_device_catalog(str(DEVICELIB))
        out = summarize_device_frontier(catalog, _hw(5))
        # mzm：LT_MZM(450 fJ/bit, 5200 um2) 与 SL_MZM(50 fJ/bit, 6250 um2) 互换优劣。
        self.assertEqual(out["mzm"]["frontier_power_area"], ["LT_MZM", "SL_MZM"])
        # phase_shifter：P_pi 轴上 IHO_PS(9.6mW/5.5um2) 与 MEMS_PS(10nW/4500um2) 互换优劣。
        # 注：TFLN_EO_PS 在该二维平面上确实被 MEMS_PS 支配——它真正的优势是
        # response_time(1.7ns vs 2000ns)，而速度并不在本前沿的坐标轴内。
        ps = out["phase_shifter"]
        self.assertEqual(ps["frontier_power_area"], ["IHO_PS", "MEMS_PS"])
        self.assertEqual(ps["tradeoffs"]["TFLN_EO_PS"]["energy_field"], "P_pi")
        self.assertIn("TFLN_EO_PS", ps["dominated_power_area"])

    def test_constructed_dac_adc_extend_the_envelope(self):
        catalog = discover_device_catalog(str(DEVICELIB))
        out = summarize_device_frontier(catalog, _hw(20))
        # 物理构造的两个 DAC 各补一个端点：20 GHz 前沿由 2 点扩到 4 点。
        self.assertEqual(
            out["dac"]["frontier_power_area"],
            ["DAC_4", "DAC_5", "DAC_HS_16nm", "DAC_LP_28nm"],
        )
        # 60 GS/s 构造 DAC 解锁 50-60 GHz 区间（此前该区间无任何可用 DAC）。
        d60 = summarize_device_frontier(catalog, _hw(60))["dac"]
        self.assertEqual(d60["compatible"], ["DAC_HS_16nm"])
        self.assertEqual(d60["frontier_power_area"], ["DAC_HS_16nm"])
        # 构造 ADC 在 20-30 GHz 提供低功耗选项（此前只有 667mW/0.45mm2 的 90GS）。
        a20 = summarize_device_frontier(catalog, _hw(20))["adc"]
        self.assertEqual(a20["frontier_power_area"], ["ADC_SAR_30GS"])


if __name__ == "__main__":
    unittest.main()
