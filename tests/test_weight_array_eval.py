"""WDM 权重阵列求值器（MRR 权重组 / PCM 交叉开关 / 单 WDM 链路）的功能测试。

覆盖论文 arXiv:2411.13715 的三种非 mesh 拓扑：
  * mrr_weight_bank — Tait 2017（微环 drop-port 权重，R+，I=2，TO 10µs）
  * pcm_crossbar    — Feldmann 2021（GST 吸收权重，R+，I=4，写入 100ns）
  * wdm_link        — Xu 2021（微梳谱线权重 × 调制器输入，I=1，无慢重配置）
"""

import copy
import unittest
from pathlib import Path

from tests._subarch import sub_arch
from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.core.search_space import switch_topology_template
from photon_tuner.core.topology_space import CORE_KINDS, validate_coupling
from photon_tuner.simulators.analytic import (
    build_weight_array_sub_arch,
    evaluate,
    load_device_catalog,
    load_required_devices,
)

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = str(ROOT / "photon_tuner" / "simulators" / "simphony")

_NODE = {
    "mrr_weight_bank": "weight_mrr",
    "pcm_crossbar": "pcm_cell",
    "wdm_link": "wdm_modulator",
}


def _wa_hw(kind, n_weight=8):
    ss = read_json(ROOT / "configs" / "search_space.json")
    hw = normalize_hardware_to_design_space(default_hardware(), ss)
    hw["arch"]["topology"]["core_kind"] = kind
    hw["arch"]["topology"]["node_kind"] = _NODE[kind]
    hw["arch"]["topology"]["mesh_rank"] = n_weight
    return hw


class TestWeightArrayEvaluation(unittest.TestCase):
    def test_all_three_evaluable_and_positive(self):
        for kind in _NODE:
            with self.subTest(kind=kind):
                report = evaluate(_wa_hw(kind), SIM_ROOT)
                m = report["metrics"]
                self.assertTrue(m["energy_pj"] > 0)
                self.assertTrue(m["area_um2"] > 0)
                self.assertTrue(m["latency_s"] > 0)

    def test_parallel_single_pass_il_is_low(self):
        # 并行权重阵列：光一次通过一个权重单元，核心 IL ≈ 激光 + 权重 + PD，
        # 远低于串行 MZI mesh（深度放大 IL）。
        for kind, expect_il in (
            ("mrr_weight_bank", 0.3),
            ("pcm_crossbar", 0.5),
            ("wdm_link", 1.2),
        ):
            with self.subTest(kind=kind):
                r = evaluate(_wa_hw(kind), SIM_ROOT)
                il = sub_arch(r["insertion_loss"])["core_insertion_loss"]
                self.assertLess(il, 5.0, f"{kind} 的并行插损应远低于 mesh")
                self.assertGreater(il, 0.0)

    def test_range_forwards_matches_paper_table(self):
        # 论文 Tab. I：MRR 阵列 I=2（R+ 差分）、PCM 交叉开关 I=4（R+ 四段）、
        # 单 WDM 链路 I=1（相干全值域）。
        self.assertEqual(CORE_KINDS["mrr_weight_bank"]["range_forwards"], 2)
        self.assertEqual(CORE_KINDS["pcm_crossbar"]["range_forwards"], 4)
        self.assertEqual(CORE_KINDS["wdm_link"]["range_forwards"], 1)

    def test_reconfig_latency_thermal_vs_pcm_vs_dynamic(self):
        # 重配置延时：MRR（TO 10µs）≫ PCM（100ns）≫ WDM 链路（0，动态调制）。
        lat = {}
        for kind in _NODE:
            r = evaluate(_wa_hw(kind), SIM_ROOT)
            lat[kind] = sub_arch(r["latency"]["computation"]).get("total_reconfig_latency", 0.0)
        self.assertGreater(lat["pcm_crossbar"], 0.0)
        self.assertGreater(lat["mrr_weight_bank"], 10 * lat["pcm_crossbar"])
        self.assertEqual(lat["wdm_link"], 0.0)

    def test_weight_device_count_is_parallel_count(self):
        # 权重器件数 = n_weight × tiles × cores（并行，无串行复制）。
        hw = _wa_hw("mrr_weight_bank", 8)
        catalog = load_device_catalog(f"{SIM_ROOT}/configs/devices")
        sub = build_weight_array_sub_arch(hw, SIM_ROOT, catalog, "mrr_weight_bank")
        comps, _, _, _ = load_required_devices(sub, catalog)
        rings = sum(v["count"] for k, v in comps.items() if k.startswith("node_weight_mrr"))
        self.assertEqual(rings, 8 * sub["tiles"] * sub["cores_per_tile"])


class TestWeightArrayCoupling(unittest.TestCase):
    def test_validate_coupling_accepts_weight_arrays(self):
        for kind, node in _NODE.items():
            with self.subTest(kind=kind):
                hw = _wa_hw(kind)
                self.assertEqual(validate_coupling(hw, mode="analytic"), [])

    def test_real_mode_accepts_weight_arrays(self):
        # 权重阵列已打通 real 路径（通用 netlist，range 用 loading_factor 枚举表达）。
        for kind in _NODE:
            with self.subTest(kind=kind):
                hw = _wa_hw(kind)
                self.assertEqual(validate_coupling(hw, mode="real"), [])

    def test_switch_topology_template_aligns_node(self):
        out = switch_topology_template(default_hardware(), "mrr_weight_bank")
        self.assertEqual(out["arch"]["topology"]["core_kind"], "mrr_weight_bank")
        self.assertEqual(out["arch"]["topology"]["node_kind"], "weight_mrr")
        out = switch_topology_template(default_hardware(), "pcm_crossbar")
        self.assertEqual(out["arch"]["topology"]["node_kind"], "pcm_cell")
        out = switch_topology_template(default_hardware(), "wdm_link")
        self.assertEqual(out["arch"]["topology"]["node_kind"], "wdm_modulator")


class TestTempoIsNotASeparateTopology(unittest.TestCase):
    """TeMPO 的时域复用不构成独立 core_kind。

    它曾注册为 ``tempo_crossbar``，但复用 ``dptc_crossbar`` 的 crossbar 网表，
    只改 name/type/ref——网表与 energy/area/latency 全部逐位相同，因此它复制了
    一整份评估结果却在报告里自称另一种拓扑。其唯一"差异"（时域复用）本就是
    ``dptc_crossbar`` 可搜索的 ``temporal_accum`` 轴。
    """

    def test_tempo_crossbar_is_not_registered(self):
        from photon_tuner.core.topology_space import CORE_KINDS, CORE_KIND_SEARCH_ORDER

        self.assertNotIn("tempo_crossbar", CORE_KINDS)
        self.assertNotIn("tempo_crossbar", CORE_KIND_SEARCH_ORDER)

    def test_its_physics_is_expressible_on_dptc_crossbar(self):
        """原 tempo 的区分点 temporal_accum 在 dptc_crossbar 上生效且改变数值。

        它通过 ``temporal_accum_factor.duration``（1/3/6）复用 time-accumulated
        器件的能耗，因此体现在 **energy** 上；latency 与 area 不受影响。
        """
        from photon_tuner.core.search_space import build_design_space

        ss = read_json(ROOT / "configs" / "search_space.json")
        ds = build_design_space()
        base = normalize_hardware_to_design_space(default_hardware(), ss)
        self.assertEqual(base["arch"]["topology"]["core_kind"], "dptc_crossbar")

        metrics = {}
        for mode in ("off", "default", "deep"):
            hw = copy.deepcopy(base)
            hw["arch"]["topology"]["temporal_accum"] = mode
            metrics[mode] = evaluate(hw, SIM_ROOT)["metrics"]

        energies = {m: v["energy_pj"] for m, v in metrics.items()}
        # 三档给出三种不同能耗 → 该自由度在 dptc_crossbar 上确实可用。
        self.assertEqual(len({round(v, 6) for v in energies.values()}), 3, energies)
        # 时域累加越深，复用越多，能耗越低。
        self.assertGreater(energies["off"], energies["default"])
        self.assertGreater(energies["default"], energies["deep"])
        # 面积/延迟与累加深度无关（它只改器件占用时间，不改结构）。
        self.assertEqual(
            len({round(v["area_um2"], 6) for v in metrics.values()}), 1
        )
        self.assertEqual(
            len({round(v["latency_s"], 15) for v in metrics.values()}), 1
        )
        # 并且它在该拓扑下是"生效"参数（不会被 PARAM_SCOPE 折叠掉）。
        self.assertNotEqual(len(ds.effective_candidates("temporal_accum", base)), 0)


class TestWeightArrayProvenance(unittest.TestCase):
    def test_new_device_variants_classified(self):
        from photon_tuner.core.provenance import classify_variant, load_full_catalog

        cat = load_full_catalog(f"{SIM_ROOT}/configs/devices")
        # 真实论文变体 → traceable（DOI）；折中构造变体 → baseline（ref 空 + note）。
        self.assertEqual(
            classify_variant("weight_mrr", "WG_MRR_WEIGHT_TAIT", cat["weight_mrr"]["WG_MRR_WEIGHT_TAIT"]),
            "traceable",
        )
        self.assertEqual(
            classify_variant("weight_mrr", "WG_MRR_WEIGHT_LP", cat["weight_mrr"]["WG_MRR_WEIGHT_LP"]),
            "baseline",
        )
        self.assertEqual(
            classify_variant("pcm_cell", "PCM_CELL_FELDMANN", cat["pcm_cell"]["PCM_CELL_FELDMANN"]),
            "traceable",
        )
        self.assertEqual(
            classify_variant("pcm_cell", "PCM_CELL_LP", cat["pcm_cell"]["PCM_CELL_LP"]),
            "baseline",
        )
        self.assertIn("doi.org", cat["weight_mrr"]["WG_MRR_WEIGHT_TAIT"]["ref"])
        self.assertIn("doi.org", cat["pcm_cell"]["PCM_CELL_FELDMANN"]["ref"])


if __name__ == "__main__":
    unittest.main()
