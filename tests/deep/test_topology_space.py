"""拓扑搜索空间的测试：模板注册表 + 耦合矩阵。"""

import unittest
from pathlib import Path

from photon_tuner.core.search_space import default_hardware
from photon_tuner.core.topology_space import (
    CORE_KINDS,
    coupling_compatible_variants,
    inspect_topology_template,
    list_topology_templates,
    validate_coupling,
)


class _FakeDesignSpace:
    """委托探针：返回固定的数值违规列表。"""

    def violations(self, hw):
        return [{"message": "work_freq 25 exceeds MZM symbol rate 1"}]


SYNTHETIC_CATALOG = {
    "dac": {
        "DAC_slow": {"cfgs": {"sample_rate": 14, "prec": 8}},
        "DAC_fast": {"cfgs": {"sample_rate": 50, "prec": 9}},
        "DAC_lowprec": {"cfgs": {"sample_rate": 50, "prec": 4}},
    },
    "adc": {
        "ADC_slow": {"cfgs": {"sample_rate": 10, "prec": 8}},
        "ADC_fast": {"cfgs": {"sample_rate": 90, "prec": 8}},
    },
    "mzm": {
        "MZM_slow": {"cfgs": {"modulation_speed": 10, "testing_bit": 1}},
        "MZM_fast": {"cfgs": {"modulation_speed": 25, "testing_bit": 1}},
    },
    "photodetector": {
        "PD_slow": {"cfgs": {"bandwidth": 10}},
        "PD_fast": {"cfgs": {"bandwidth": 45}},
    },
    # 未报告的速率/精度键 -> 永不被过滤（没有虚构的上限）。
    "tia": {"TIA_1": {"cfgs": {}}},
}


class TestTopologyTemplates(unittest.TestCase):
    def test_evaluable_templates(self):
        tpls = list_topology_templates()
        self.assertEqual(
            tpls["evaluable_templates"],
            [
                "dptc_crossbar", "mrr_weight_bank", "mzi_mesh_butterfly",
                "mzi_mesh_clements", "mzi_mesh_reck", "pcm_crossbar",
                "wdm_link",
            ],
        )
        self.assertIn("mzi_mesh_reck", tpls["legal_templates"])
        self.assertIn("mzi_mesh_clements", tpls["legal_templates"])
        self.assertIn("mzi_mesh_butterfly", tpls["legal_templates"])
        # 1 个 crossbar + 3 个 mesh + 3 个 WDM 权重阵列均可解析求值，
        # 且均打通 real 路径（ONNArchSimulator 消费通用 netlist）。
        self.assertTrue(CORE_KINDS["mzi_mesh_reck"]["evaluable"])
        self.assertTrue(CORE_KINDS["mzi_mesh_clements"]["evaluable"])
        self.assertTrue(CORE_KINDS["mzi_mesh_butterfly"]["evaluable"])
        self.assertTrue(CORE_KINDS["mzi_mesh_butterfly"]["real"])
        self.assertTrue(CORE_KINDS["mrr_weight_bank"]["evaluable"])
        self.assertTrue(CORE_KINDS["pcm_crossbar"]["evaluable"])
        self.assertTrue(CORE_KINDS["wdm_link"]["evaluable"])

    def test_inspect_template(self):
        info = inspect_topology_template("dptc_crossbar")
        self.assertEqual(info["compatible_node_kinds"], ["dode"])
        self.assertTrue(info["evaluable"])

    def test_inspect_mesh_template(self):
        info = inspect_topology_template("mzi_mesh_clements")
        self.assertEqual(info["compatible_node_kinds"], ["tunable_mzi"])
        self.assertTrue(info["evaluable"])

    def test_inspect_unknown_template_raises(self):
        with self.assertRaises(ValueError):
            inspect_topology_template("nope")

    def test_real_mode_advertises_all_topologies(self):
        # 全部 7 个拓扑均已打通 real 路径：ONNArchSimulator 消费通用 netlist
        # （mesh 用 MZI-mesh 网表、权重阵列用微梳+权重+PD 网表）。
        real = list_topology_templates(mode="real")
        self.assertEqual(
            sorted(real["evaluable_templates"]),
            [
                "dptc_crossbar", "mrr_weight_bank", "mzi_mesh_butterfly",
                "mzi_mesh_clements", "mzi_mesh_reck", "pcm_crossbar",
                "wdm_link",
            ],
        )
        for kind in (
            "dptc_crossbar", "mzi_mesh_butterfly",
            "mzi_mesh_clements", "mzi_mesh_reck", "mrr_weight_bank",
            "pcm_crossbar", "wdm_link",
        ):
            self.assertTrue(CORE_KINDS[kind]["real"], kind)


class TestCouplingValidation(unittest.TestCase):
    def test_default_hardware_is_consistent(self):
        self.assertEqual(validate_coupling(default_hardware()), [])

    def test_bad_core_kind(self):
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "bogus"
        msgs = [v["message"] for v in validate_coupling(hw)]
        self.assertTrue(any("core_kind" in m for m in msgs))

    def test_butterfly_accepted_in_analytic_mode(self):
        # butterfly 解析与 real 均可求值（结构事实来自 Cooley-Tukey + Clements/Reck 分解）。
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_butterfly"
        hw["arch"]["topology"]["node_kind"] = "tunable_mzi"
        self.assertEqual(validate_coupling(hw, mode="analytic"), [])

    def test_clements_mesh_default_hardware_is_consistent(self):
        # 切换到可求值网格必须针对网格器件表（devices.mesh）干净地通过校验，
        # 而不是针对 crossbar 表。
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_clements"
        hw["arch"]["topology"]["node_kind"] = "tunable_mzi"
        self.assertEqual(validate_coupling(hw), [])

    def test_clements_mesh_missing_slot_flagged(self):
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_clements"
        hw["arch"]["topology"]["node_kind"] = "tunable_mzi"
        del hw["devices"]["mesh"]["node"]["phase_shifter"]
        msgs = [v["message"] for v in validate_coupling(hw)]
        self.assertTrue(any("missing device slots" in m and "phase_shifter" in m for m in msgs))

    def test_incompatible_node_kind(self):
        hw = default_hardware()
        hw["arch"]["topology"]["node_kind"] = "tunable_mzi"  # dptc_crossbar 需要 dode
        msgs = [v["message"] for v in validate_coupling(hw)]
        self.assertTrue(any("incompatible" in m for m in msgs))

    def test_missing_device_slot(self):
        hw = default_hardware()
        del hw["devices"]["core"]["mzm"]
        msgs = [v["message"] for v in validate_coupling(hw)]
        self.assertTrue(any("missing device slots" in m and "mzm" in m for m in msgs))

    def test_delegates_numeric_constraints(self):
        # design_space 的数值违规被追加到 scope=numeric 下。
        hw = default_hardware()
        out = validate_coupling(hw, _FakeDesignSpace())
        self.assertTrue(any(v["scope"] == "numeric" for v in out))

    def test_mesh_accepted_in_both_modes(self):
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_clements"
        hw["arch"]["topology"]["node_kind"] = "tunable_mzi"
        # 网格镜像在解析模式下*和*真实 ONNArchSimulator 路径上都运行
        # （网格网表 + n_modes x n_modes 分区 + 时间因子）。
        self.assertEqual(validate_coupling(hw, mode="analytic"), [])
        self.assertEqual(validate_coupling(hw, mode="real"), [])

    def test_butterfly_accepted_in_real_mode(self):
        # Butterfly 现以标准 MZI-mesh 网表表达其结构（MZI 数/深度），
        # ONNArchSimulator 可消费 → real 模式接受。
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_butterfly"
        hw["arch"]["topology"]["node_kind"] = "tunable_mzi"
        self.assertEqual(validate_coupling(hw, mode="real"), [])

    def test_crossbar_accepted_in_both_modes(self):
        hw = default_hardware()
        self.assertEqual(validate_coupling(hw, mode="analytic"), [])
        self.assertEqual(validate_coupling(hw, mode="real"), [])


class TestTopologyAxisIsActionable(unittest.TestCase):
    """拓扑层的结构性移动必须既可应用*且*合法。

    求解智能体无法转化为编辑（或会产生耦合不一致设计点）的推荐，不是真正的搜索轴。
    """

    def _space(self):
        from photon_tuner.common.io import read_json
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        return read_json(root / "configs" / "search_space.json")

    def test_core_kind_switch_and_l4_levers_apply_together(self):
        from photon_tuner.core.search_space import modify_arch_parameter

        space = self._space()
        hw = default_hardware()
        updates = [
            {"parameter": "core_kind", "value": "mzi_mesh_clements"},
            {"parameter": "node_kind", "value": "tunable_mzi"},
            {"parameter": "sharing_policy", "value": "adc_tia_shared"},
            {"parameter": "temporal_accum", "value": "deep"},
            {"parameter": "laser_distribution", "value": "H*W*N"},
        ]
        hw = modify_arch_parameter(hw, updates, space)
        topo = hw["arch"]["topology"]
        self.assertEqual(topo["core_kind"], "mzi_mesh_clements")
        self.assertEqual(topo["node_kind"], "tunable_mzi")
        # 结果在真实模式下必须仍是耦合一致的设计点。
        self.assertEqual(validate_coupling(hw, mode="real"), [])

    def test_excluded_idealized_device_variant_is_rejected(self):
        from photon_tuner.core.search_space import set_device_choice

        hw = default_hardware()
        for device_type, variant in (
            ("phase_shifter", "Customized_PS"),
            ("on_chip_laser", "LT_Laser"),
            ("y_branch", "Customized_Y_Branch"),
            ("coupler", "Customized_DC_1x1"),
        ):
            with self.assertRaises(ValueError, msg=f"{device_type}/{variant}"):
                set_device_choice(hw, "core", device_type, variant)

    def test_legitimate_variant_still_applies(self):
        from photon_tuner.core.search_space import set_device_choice

        hw = set_device_choice(default_hardware(), "mesh_node", "phase_shifter", "ITO_PS")
        self.assertEqual(hw["devices"]["mesh"]["node"]["phase_shifter"], "ITO_PS")


class TestCouplingFilteredCandidates(unittest.TestCase):
    def _hw(self, work_freq, in_bit, out_bit):
        hw = default_hardware()
        hw["arch"]["optical_core"]["work_freq"] = work_freq
        hw["arch"]["optical_core"]["precision"]["in_bit"] = in_bit
        hw["arch"]["optical_core"]["precision"]["out_bit"] = out_bit
        return hw

    def test_filters_by_rate_and_precision(self):
        # work_freq=20 GHz、in_bit=8、out_bit=8 -> 慢速/低精度变体被剔除。
        out = coupling_compatible_variants(SYNTHETIC_CATALOG, self._hw(20, 8, 8))
        self.assertEqual(sorted(out["compatible"]["dac"]), ["DAC_fast"])
        self.assertEqual(sorted(out["excluded"]["dac"]), ["DAC_lowprec", "DAC_slow"])
        self.assertEqual(sorted(out["compatible"]["adc"]), ["ADC_fast"])
        self.assertEqual(sorted(out["excluded"]["adc"]), ["ADC_slow"])
        self.assertEqual(sorted(out["compatible"]["mzm"]), ["MZM_fast"])
        self.assertEqual(sorted(out["excluded"]["mzm"]), ["MZM_slow"])
        self.assertEqual(sorted(out["compatible"]["photodetector"]), ["PD_fast"])
        self.assertEqual(sorted(out["excluded"]["photodetector"]), ["PD_slow"])
        # TIA 没有报告的速率/精度 -> 永不被过滤。
        self.assertEqual(out["compatible"]["tia"], ["TIA_1"])
        self.assertNotIn("tia", out["excluded"])

    def test_default_hardware_keeps_all(self):
        # work_freq=5、in_bit=4、out_bit=4 -> 每个合成变体都满足。
        out = coupling_compatible_variants(SYNTHETIC_CATALOG, self._hw(5, 4, 4))
        self.assertEqual(out["excluded"], {})

    def test_no_hardware_filters_nothing(self):
        out = coupling_compatible_variants(SYNTHETIC_CATALOG, None)
        self.assertEqual(out["excluded"], {})
        self.assertEqual(sorted(out["compatible"]["dac"]), ["DAC_fast", "DAC_lowprec", "DAC_slow"])


class TestDeviceCandidateToolFiltering(unittest.TestCase):
    def test_list_device_candidates_filters_by_coupling(self):
        from photon_tuner.toolbox.common_tool.context import ToolContext
        from photon_tuner.toolbox.device_tools.catalog import list_device_candidates

        class _Box:
            INSTANCE_CHOICES = {"laser_source": ["on_chip", "off_chip"]}

            def __init__(self, catalog):
                self.catalog = catalog
                self.variants = {dt: sorted(v) for dt, v in catalog.items()}

        box = _Box(SYNTHETIC_CATALOG)
        hw = default_hardware()
        hw["arch"]["optical_core"]["work_freq"] = 20
        hw["arch"]["optical_core"]["precision"]["in_bit"] = 8
        hw["arch"]["optical_core"]["precision"]["out_bit"] = 8
        ctx = ToolContext(output_dir=Path("."))
        ctx.current_hardware = hw

        result = list_device_candidates(box, ctx, {})
        payload = result.payload
        self.assertTrue(payload["coupling_filtered"])
        self.assertEqual(sorted(payload["variants"]["dac"]), ["DAC_fast"])
        self.assertEqual(sorted(payload["excluded"]["dac"]), ["DAC_lowprec", "DAC_slow"])

        # 实例切换不做能力过滤。
        result2 = list_device_candidates(box, ctx, {"device_type": "laser_source"})
        self.assertEqual(result2.payload, {"laser_source": ["on_chip", "off_chip"]})


if __name__ == "__main__":
    unittest.main()
