"""校验搜索空间的步进、钳位与器件编辑。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.core.search_space import (
    default_hardware,
    materialize_hardware,
    normalize_hardware_to_design_space,
    set_device_choice,
    step_arch_parameter,
    switch_topology_template,
)


class TestSearchSpace(unittest.TestCase):
    def setUp(self):
        self.space = {
            "arch": {
                "tiles": [1, 2, 4, 8],
                "cores_per_tile": [1, 2, 4],
                "core_height": [8, 12, 16],
                "core_width": [8, 12, 16],
                "num_wavelength": [8, 12, 16],
                "work_freq": [5, 10, 25],
                "dataflow": ["output_stationary", "weight_stationary", "input_stationary"],
                "precision": {"in_bit": [2, 4, 8], "w_bit": [2, 4, 8], "out_bit": [2, 4, 8]},
            }
        }

    def test_materialize_fills_defaults(self):
        hw = materialize_hardware({})
        self.assertEqual(hw["arch"]["optical_core"]["num_wavelength"], 12)

    def test_instance_choice_defaults_present(self):
        hw = default_hardware()
        self.assertEqual(hw["devices"]["core"]["laser_source"], "on_chip")
        self.assertEqual(hw["devices"]["core"]["splitter_kind"], "laser_splitter")

    def test_step_arch_parameter_advances(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "num_wavelength", 1, self.space)
        self.assertEqual(stepped["arch"]["optical_core"]["num_wavelength"], 16)

    def test_step_arch_parameter_clamps(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "num_wavelength", -1, self.space)
        self.assertEqual(stepped["arch"]["optical_core"]["num_wavelength"], 8)
        # 再次向下步进应停留在最小值。
        again = step_arch_parameter(stepped, "num_wavelength", -1, self.space)
        self.assertEqual(again["arch"]["optical_core"]["num_wavelength"], 8)

    def test_step_dataflow_cycles(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "dataflow", 1, self.space)
        self.assertEqual(stepped["arch"]["system"]["dataflow"], "weight_stationary")

    def test_normalize_clamps_out_of_range(self):
        hw = default_hardware()
        hw["arch"]["optical_core"]["num_wavelength"] = 999
        normalized = normalize_hardware_to_design_space(hw, self.space)
        self.assertEqual(normalized["arch"]["optical_core"]["num_wavelength"], 16)

    def test_set_device_choice(self):
        hw = set_device_choice(default_hardware(), "core", "mzm", "LT_MZM_v2")
        self.assertEqual(hw["devices"]["core"]["mzm"], "LT_MZM_v2")
        # 框架不包含 device_cfgs 覆盖字段。
        self.assertNotIn("device_cfgs", hw)

    def test_step_core_kind_advances(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "core_kind", 1, None)
        # 候选顺序按拓扑族分组（CORE_KIND_SEARCH_ORDER），索引即邻接语义。
        # crossbar 家族只有 dptc_crossbar 一个成员，故下一次步进进入网格家族，
        # 而不是跳到字典序的下一个 mrr_weight_bank。
        self.assertEqual(stepped["arch"]["topology"]["core_kind"], "mzi_mesh_clements")

    def test_step_core_kind_stays_within_family(self):
        """±1 步进应保持在同一拓扑族内（族边界除外）。"""
        from photon_tuner.core.topology_space import CORE_KINDS, core_family

        hw = default_hardware()
        current = hw["arch"]["topology"]["core_kind"]
        stepped = step_arch_parameter(hw, "core_kind", 1, None)["arch"]["topology"][
            "core_kind"
        ]
        family = core_family(current)
        # 只有该族的最后一个成员才会跨族；单成员族（crossbar）必然跨族。
        order = [
            k for k in CORE_KINDS
            if core_family(k) == family and CORE_KINDS[k].get("evaluable")
        ]
        if current != order[-1]:
            self.assertEqual(core_family(stepped), family)
        else:
            self.assertNotEqual(core_family(stepped), family)

    def test_step_node_kind_advances(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "node_kind", 1, None)
        # 同 core_kind：按 NODE_KIND_SEARCH_ORDER 的语义顺序前进。
        self.assertEqual(stepped["arch"]["topology"]["node_kind"], "tunable_mzi")

    def test_switch_topology_template_aligns_node_kind(self):
        hw = default_hardware()
        out = switch_topology_template(hw, "mzi_mesh_clements")
        self.assertEqual(out["arch"]["topology"]["core_kind"], "mzi_mesh_clements")
        self.assertEqual(out["arch"]["topology"]["node_kind"], "tunable_mzi")
        # 源硬件不被修改（深拷贝）。
        self.assertEqual(hw["arch"]["topology"]["core_kind"], "dptc_crossbar")

    def test_switch_topology_template_accepts_butterfly(self):
        # butterfly 现在解析可求值（结构事实来自 Cooley-Tukey + Clements/Reck 分解），
        # 切换应被接受并自动对齐 node_kind。
        out = switch_topology_template(default_hardware(), "mzi_mesh_butterfly")
        self.assertEqual(out["arch"]["topology"]["core_kind"], "mzi_mesh_butterfly")
        self.assertEqual(out["arch"]["topology"]["node_kind"], "tunable_mzi")

    def test_switch_unknown_template_raises(self):
        with self.assertRaises(ValueError):
            switch_topology_template(default_hardware(), "nope")


if __name__ == "__main__":
    unittest.main()
