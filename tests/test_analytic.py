"""对照 SimPhony 参考值校验离线 analytic 镜像。"""

import sys
import copy
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._subarch import sub_arch
from photon_tuner.core.search_space import default_hardware
from photon_tuner.simulators.analytic import (
    build_sub_arch,
    energy_calculator,
    evaluate,
    initialize_device_power,
    insertion_loss,
    load_device_catalog,
    load_required_devices,
)


def _reference_hardware():
    """固定的 LT crossbar 参考设计点（解析镜像算术的回归输入）。"""
    hw = default_hardware()
    core = hw["arch"]["optical_core"]
    core.update({"height": 12, "width": 12, "num_wavelength": 12, "work_freq": 5})
    for level in ("GLB1", "GLB2", "RF"):
        hw["arch"]["memory"][level]["cache_size"] = 485
    return hw


class TestAnalyticEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony"
        cls.catalog = load_device_catalog(str(cls.root / "configs" / "devices"))
        # 解析镜像的回归锚点必须固定输入，不能跟着 configs/initial_hardware.json
        # 漂移——否则每次调整搜索空间基线，这一组"物理算术是否正确"的断言都会
        # 被误判为回归。这里显式构造历史上的 12x12x12 @5 GHz / 485 KB LT 点。
        cls.hardware = _reference_hardware()
        cls.sub_arch = build_sub_arch(cls.hardware, str(cls.root), cls.catalog)

    def test_metrics_are_positive(self):
        report = evaluate(self.hardware, simphony_root=str(self.root))
        metrics = report["metrics"]
        self.assertGreater(metrics["energy_mj"], 0)
        self.assertGreater(metrics["area_mm2"], 0)
        self.assertGreater(metrics["latency_ms"], 0)

    def test_energy_total_matches_reference(self):
        # 默认工作负载为 vit_tiny；此处钉住解析镜像的 LT 能量作为回归锚点。
        report = evaluate(self.hardware, simphony_root=str(self.root))
        total = sub_arch(report["energy"]["total"])
        self.assertAlmostEqual(float(total), 115562731.63852626, delta=2.0)

    def test_insertion_loss_matches_reference(self):
        il = insertion_loss(self.sub_arch, self.catalog)
        self.assertAlmostEqual(il["node_insertion_loss"], 0.76, delta=1e-6)
        self.assertAlmostEqual(il["core_insertion_loss"], 6.22, delta=1e-6)

    def test_device_counts_match_reference(self):
        components, _, _, _ = load_required_devices(self.sub_arch, self.catalog)
        by_type = {}
        for device, cfg in components.items():
            if device == "node":
                continue
            key = (_instance_index(device), device.split("-")[-1])
            by_type[key] = by_type.get(key, 0) + cfg["count"]

        # LT 核心器件的参考数量。
        self.assertEqual(by_type.get(("i0", "LT_MZM")), 1152)
        self.assertEqual(by_type.get(("i1", "DAC_2")), 1152)
        self.assertEqual(by_type.get(("i2", "ADC_SAR_1")), 576)
        self.assertEqual(by_type.get(("i3", "LT_MRR_REROUTER")), 4608)

    def test_node_bounding_box_matches_reference(self):
        report = evaluate(self.hardware, simphony_root=str(self.root))
        node = sub_arch(report["area"]["by_device"])["node"]
        self.assertAlmostEqual(node["area"], 11688.0, delta=1e-6)
        self.assertAlmostEqual(node["width"], 120.0, delta=1e-6)
        self.assertAlmostEqual(node["length"], 97.4, delta=1e-6)

    def test_area_is_monotonic_in_core_size(self):
        from photon_tuner.core.search_space import step_arch_parameter

        # core_width 的候选是 2 的幂，因此必须从一个落在网格上的点起步，
        # 否则 step 会先被 normalize 折回更小的候选，测出的不是"变大"而是"变小"。
        on_grid = copy.deepcopy(self.hardware)
        on_grid["arch"]["optical_core"]["width"] = 8
        base = evaluate(on_grid, simphony_root=str(self.root))
        larger = evaluate(
            step_arch_parameter(on_grid, "core_width", 1), simphony_root=str(self.root)
        )
        self.assertGreater(larger["metrics"]["area_mm2"], base["metrics"]["area_mm2"])

    def test_sharing_policy_changes_adc_scaling(self):
        hw = default_hardware()
        hw["arch"]["topology"]["sharing_policy"] = "no_sharing"
        sub = build_sub_arch(hw, str(self.root), self.catalog)
        rules = sub["core"]["netlist"]["scaling_rules"]
        self.assertEqual(rules["i2"], "H*W*R*C")  # 每个节点私有的 ADC
        self.assertEqual(rules["i8"], "H*W*R*C")

    def test_temporal_accum_duration(self):
        hw = default_hardware()
        hw["arch"]["topology"]["temporal_accum"] = "deep"
        sub = build_sub_arch(hw, str(self.root), self.catalog)
        self.assertEqual(sub["core"]["netlist"]["temporal_accum_factor"]["duration"], 6)

    def test_laser_distribution_written_to_both_keys(self):
        hw = default_hardware()
        hw["arch"]["topology"]["laser_distribution"] = "H*W*N"
        sub = build_sub_arch(hw, str(self.root), self.catalog)
        self.assertEqual(sub["core"]["netlist"]["laser_power_distribution"], "H*W*N")
        self.assertEqual(sub["core"]["netlist"]["laser_power_distribution_factor"], "H*W*N")

    def test_memory_carries_cacti_config_filenames(self):
        sub = build_sub_arch(self.hardware, str(self.root), self.catalog)
        self.assertEqual(sub["memory"]["GLB2"]["config_file_name"], "GLB2.cfg")
        self.assertEqual(sub["memory"]["RF"]["config_file_name"], "RF.cfg")

    # -- 实例级器件类型切换（mmi / off_chip_laser） --------- #

    def test_instance_choice_switches_netlist_types(self):
        hw = default_hardware()
        hw["devices"]["core"]["laser_source"] = "off_chip"
        hw["devices"]["core"]["splitter_kind"] = "mmi"
        sub = build_sub_arch(hw, str(self.root), self.catalog)
        inst = sub["core"]["netlist"]["instances"]
        self.assertEqual(inst["i4"][0], "off_chip_laser")
        self.assertEqual(inst["i6"][0], "mmi")
        self.assertEqual(sub["core"]["netlist"]["ports_num"]["i6"], 10)
        # 切换回来恢复基线网表。
        back = build_sub_arch(default_hardware(), str(self.root), self.catalog)
        self.assertEqual(back["core"]["netlist"]["instances"]["i4"][0], "on_chip_laser")
        self.assertEqual(back["core"]["netlist"]["instances"]["i6"][0], "laser_splitter")

    def test_mmi_switch_changes_area_and_insertion_loss(self):
        base = evaluate(self.hardware, simphony_root=str(self.root))
        hw = default_hardware()
        hw["devices"]["core"]["splitter_kind"] = "mmi"
        mmi = evaluate(hw, simphony_root=str(self.root))
        self.assertGreater(mmi["metrics"]["area_mm2"], base["metrics"]["area_mm2"])
        self.assertGreater(
            sub_arch(mmi["insertion_loss"])["core_insertion_loss"],
            sub_arch(base["insertion_loss"])["core_insertion_loss"],
        )

    def test_off_chip_laser_changes_energy(self):
        base = evaluate(self.hardware, simphony_root=str(self.root))
        hw = default_hardware()
        hw["devices"]["core"]["laser_source"] = "off_chip"
        off = evaluate(hw, simphony_root=str(self.root))
        self.assertLess(off["metrics"]["energy_mj"], base["metrics"]["energy_mj"])


def _instance_index(device: str) -> str:
    # 设备名形如：core_mzm_0_i0-LT_MZM
    tail = device.split("-")[0]
    return tail.rsplit("_", 1)[-1]


if __name__ == "__main__":
    unittest.main()
