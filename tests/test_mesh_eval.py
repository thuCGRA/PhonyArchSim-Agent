"""analytic MZI-mesh 求值器的功能测试。"""

import json
import unittest
from pathlib import Path

from tests._subarch import sub_arch
from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.simulators.analytic import (
    build_mesh_sub_arch,
    evaluate,
    insertion_loss,
    load_device_catalog,
    load_required_devices,
)
from photon_tuner.simulators.mesh_model import mesh_topology

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = str(ROOT / "photon_tuner" / "simulators" / "simphony")

# 录制运行基线：这些 ``energy_detail.json`` 原本来自 ``runs/run_agent/``，
# 而 ``runs/*`` 被 .gitignore 忽略 —— CI 全新检出时并不存在。因此把所需
# 起点作为显式 fixture 提交到 tests/fixtures/，测试不再依赖本地运行产物。
RECORDED_RUNS = ROOT / "tests" / "fixtures" / "recorded_runs"


def _mesh_hw(kind, n_modes=4):
    ss = read_json(ROOT / "configs" / "search_space.json")
    hw = normalize_hardware_to_design_space(default_hardware(), ss)
    hw["arch"]["topology"]["core_kind"] = kind
    hw["arch"]["topology"]["node_kind"] = "tunable_mzi"
    # ``mesh_rank`` 是 topology 层的酉秩旋钮；它必须驱动 mesh 尺寸
    # （build_mesh_sub_arch 优先于 core 高度/宽度使用它）。
    hw["arch"]["topology"]["mesh_rank"] = n_modes
    hw["arch"]["optical_core"]["height"] = n_modes
    hw["arch"]["optical_core"]["width"] = n_modes
    return hw


class TestMeshEvaluation(unittest.TestCase):
    def test_clements_metrics_finite_and_positive(self):
        report = evaluate(_mesh_hw("mzi_mesh_clements", 4), SIM_ROOT)
        m = report["metrics"]
        self.assertTrue(m["energy_pj"] > 0)
        self.assertTrue(m["area_um2"] > 0)
        self.assertTrue(m["latency_s"] > 0)

    def test_mzi_device_count_matches_paper(self):
        # N=4 Clements：n_mzi = 6；tiles=4、cores=2 → 6*4*2 = 48 个 MZI。
        hw = _mesh_hw("mzi_mesh_clements", 4)
        catalog = load_device_catalog(f"{SIM_ROOT}/configs/devices")
        sub = build_mesh_sub_arch(hw, SIM_ROOT, catalog, "mzi_mesh_clements")
        comps, _, _, _ = load_required_devices(sub, catalog)
        couplers = sum(v["count"] for k, v in comps.items() if k.startswith("node_coupler"))
        ps = sum(v["count"] for k, v in comps.items() if k.startswith("node_phase_shifter"))
        n_mzi = mesh_topology("clements", 4)["n_mzi"]
        r = sub["tiles"]
        c = sub["cores_per_tile"]
        self.assertEqual(couplers, 2 * n_mzi * r * c)  # 每个 MZI 2 个耦合器
        self.assertEqual(ps, 2 * n_mzi * r * c)        # 每个 MZI 2 个移相器

    def test_insertion_loss_includes_full_depth(self):
        hw = _mesh_hw("mzi_mesh_clements", 4)
        catalog = load_device_catalog(f"{SIM_ROOT}/configs/devices")
        sub = build_mesh_sub_arch(hw, SIM_ROOT, catalog, "mzi_mesh_clements")
        il = insertion_loss(sub, catalog)
        depth = mesh_topology("clements", 4)["depth"]
        mzi_stages = [n for n in il["core_path"] if n.startswith("i_mzi_")]
        self.assertEqual(len(mzi_stages), depth)
        # core IL 必须至少覆盖 depth × node(MZI) IL。
        self.assertGreaterEqual(il["core_insertion_loss"], depth * il["node_insertion_loss"])

    def test_reck_and_butterfly_evaluate(self):
        for kind in ("mzi_mesh_reck", "mzi_mesh_butterfly"):
            report = evaluate(_mesh_hw(kind, 4), SIM_ROOT)
            self.assertTrue(report["metrics"]["energy_pj"] > 0, kind)
            self.assertTrue(report["metrics"]["area_um2"] > 0, kind)

    def test_mesh_component_loss_breakdown_is_populated(self):
        # mesh 实例名（"i_mzi_0"、"i_laser" 等）必须在复制索引剥离后保留，否则
        # 按组件归因会静默坍缩为空列表，topology 智能体无法说明*哪个*组件损耗多少 dB。
        catalog = load_device_catalog(f"{SIM_ROOT}/configs/devices")
        for kind in ("mzi_mesh_clements", "mzi_mesh_reck"):
            sub = build_mesh_sub_arch(_mesh_hw(kind, 4), SIM_ROOT, catalog, kind)
            il = insertion_loss(sub, catalog)
            detail = il.get("core_path_detail") or []
            self.assertTrue(detail, f"{kind}: core_path_detail is empty")
            mzi_entries = [e for e in detail if e["instance"].startswith("i_mzi")]
            self.assertTrue(mzi_entries, f"{kind}: no MZI entries in breakdown")
            # 路径的每一步都必须可归因。
            self.assertEqual(len(detail), len(il["core_path"]))
            # 分解必须与报告的总量吻合。
            self.assertAlmostEqual(
                sum(e["insertion_loss_dB"] for e in detail),
                il["core_insertion_loss"],
                places=6,
            )


class TestMeshDeviceLayerAuthority(unittest.TestCase):
    """器件层必须对 mesh 损耗拥有真实、可度量的决定权。"""

    def test_low_loss_phase_shifter_cuts_energy_not_just_il(self):
        # 器件层值得搜索的因果链：更低的节点 IL → 闭合链路所需光功率更少 → 激光
        # 能量更低。mesh 能量约 97% 为激光静态功耗，因此 IL 下降必须带来能量变化。
        from photon_tuner.core.search_space import set_device_choice

        hw = _mesh_hw("mzi_mesh_clements", 16)
        before = evaluate(hw, SIM_ROOT)
        after = evaluate(
            set_device_choice(hw, "mesh_node", "phase_shifter", "ITO_PS"), SIM_ROOT
        )
        self.assertLess(
            sub_arch(after["insertion_loss"])["core_insertion_loss"],
            sub_arch(before["insertion_loss"])["core_insertion_loss"],
        )
        self.assertLess(after["metrics"]["energy_pj"], before["metrics"]["energy_pj"])

    def test_deeper_mesh_is_worse_than_clements(self):
        # 相同模式数下 Reck（深度 2N-3）必须比 Clements（深度 N）损耗更多光——
        # 范式轴不得退化。
        clements = sub_arch(evaluate(_mesh_hw("mzi_mesh_clements", 4), SIM_ROOT)["insertion_loss"])
        reck = sub_arch(evaluate(_mesh_hw("mzi_mesh_reck", 4), SIM_ROOT)["insertion_loss"])
        self.assertGreater(reck["core_insertion_loss"], clements["core_insertion_loss"])

    def test_switching_mesh_node_phase_shifter_reduces_core_il(self):
        from photon_tuner.core.search_space import set_device_choice

        hw = _mesh_hw("mzi_mesh_clements", 16)
        before = sub_arch(evaluate(hw, SIM_ROOT)["insertion_loss"])
        # ITO_PS（0.03 dB）是*合法*的库变体；Customized_PS 作为理想化器件被排除
        # 在搜索空间外，因此不得使用。
        after_hw = set_device_choice(hw, "mesh_node", "phase_shifter", "ITO_PS")
        after = sub_arch(evaluate(after_hw, SIM_ROOT)["insertion_loss"])
        # 更低损耗的移相器必须降低整个 16 层深 mesh 的 IL。
        self.assertLess(after["core_insertion_loss"], before["core_insertion_loss"])
        self.assertLess(after["node_insertion_loss"], before["node_insertion_loss"])
        # 且逐级分解必须体现该改善。
        stages = [e["insertion_loss_dB"] for e in after["core_path_detail"]
                  if e["instance"].startswith("i_mzi")]
        self.assertTrue(stages)
        self.assertTrue(all(dB < 1.0 for dB in stages), stages)

    def test_excluded_idealized_variant_is_rejected(self):
        from photon_tuner.core.search_space import set_device_choice

        hw = _mesh_hw("mzi_mesh_clements", 4)
        # Customized_PS 具有 P_pi=0 / response_time=0 / 无参考文献，被刻意排除；
        # 接受它会把非物理设计点放到 Pareto 前沿上。
        with self.assertRaises(ValueError):
            set_device_choice(hw, "mesh_node", "phase_shifter", "Customized_PS")


class TestMeshUsesOnlyExistingDevices(unittest.TestCase):
    """禁制造规则：mesh 只能引用库内器件。"""

    def test_mesh_device_choices_exist_in_catalog(self):
        catalog = load_device_catalog(f"{SIM_ROOT}/configs/devices")
        hw = _mesh_hw("mzi_mesh_clements", 4)
        sub = build_mesh_sub_arch(hw, SIM_ROOT, catalog, "mzi_mesh_clements")
        for table in (sub["core"]["devices"], sub["core"]["node"]["devices"]):
            for device_type, names in table.items():
                for name in names:
                    self.assertIn(name, catalog.get(device_type, {}), f"{device_type}/{name}")


class TestCrossParadigmPareto(unittest.TestCase):
    """crossbar 与 Clements mesh 的跨范式 Pareto 权衡。

    - 原始（未归一化）轴上两者共存：mesh 以面积换取插入损耗。
    - 吞吐归一化轴上 mesh 被 crossbar 支配：mesh 的酉通用性不体现在数值效率上。
    """

    def _metrics(self, hw):
        report = evaluate(hw, SIM_ROOT)
        m = dict(report["metrics"])
        m["core_insertion_loss"] = sub_arch(report["insertion_loss"])["core_insertion_loss"]
        return m

    def _crossbar_and_clements(self):
        crossbar = normalize_hardware_to_design_space(
            default_hardware(), read_json(ROOT / "configs" / "search_space.json")
        )
        crossbar["arch"]["optical_core"]["height"] = 8
        crossbar["arch"]["optical_core"]["width"] = 8
        crossbar["arch"]["optical_core"]["num_wavelength"] = 8
        return crossbar, _mesh_hw("mzi_mesh_clements", 8)

    def test_both_paradigms_on_frontier_on_raw_axes(self):
        # 原始轴上 crossbar 与 mesh 仍有面积换 IL 的权衡（两者互不支配）。
        from photon_tuner.core.pareto import ParetoArchive

        archive = ParetoArchive(minimize_keys=["energy_pj", "area_um2", "latency_s", "core_insertion_loss"])
        crossbar, clements = self._crossbar_and_clements()
        archive.add(self._metrics(crossbar), meta={"core_kind": "dptc_crossbar"})
        archive.add(self._metrics(clements), meta={"core_kind": "mzi_mesh_clements"})

        kinds = {p["meta"]["core_kind"] for p in archive.frontier()}
        self.assertEqual(kinds, {"dptc_crossbar", "mzi_mesh_clements"})
        # mesh 以面积换取插入损耗：面积更小、IL 更高。
        by = {p["meta"]["core_kind"]: p["metrics"] for p in archive.frontier()}
        self.assertLess(by["mzi_mesh_clements"]["area_um2"], by["dptc_crossbar"]["area_um2"])
        self.assertGreater(by["mzi_mesh_clements"]["core_insertion_loss"], by["dptc_crossbar"]["core_insertion_loss"])

    def test_throughput_normalization_dominates_mesh(self):
        # 吞吐归一化轴上 mesh 被 crossbar 支配：面积归一化到吞吐后 mesh 的面积优势被
        # 低吞吐抵消，且其 IL/能耗/延迟更高。
        from photon_tuner.core.pareto import ParetoArchive

        archive = ParetoArchive()  # 默认归一化轴
        crossbar, clements = self._crossbar_and_clements()
        self.assertTrue(archive.add(self._metrics(crossbar), meta={"core_kind": "dptc_crossbar"}))
        self.assertFalse(archive.add(self._metrics(clements), meta={"core_kind": "mzi_mesh_clements"}))
        self.assertEqual(
            {p["meta"]["core_kind"] for p in archive.frontier()}, {"dptc_crossbar"}
        )

    def test_reck_is_evaluable_and_dominated_by_clements(self):
        """Reck 求值正确（最深路径 2N-3），且被 Clements Pareto 支配。

        Reck 与 Clements MZI 数相同，但 Reck 为 crossing-bearing：既含布线交叉的
        面积，又因更深（2N-3）最坏路径 + 交叉插损抬高 core IL 与能量 → 三轴全面更差。
        """
        from photon_tuner.core.pareto import ParetoArchive

        clements = _mesh_hw("mzi_mesh_clements", 8)
        reck = _mesh_hw("mzi_mesh_reck", 8)
        m_clem, m_reck = self._metrics(clements), self._metrics(reck)

        # 面积：MZI 数相同，但 Reck 含交叉 → 面积更高。
        # Reck 更深的（2N-3）最坏路径 + 交叉插损抬高了 core IL 以及（经由激光链路预算）能量。
        self.assertGreater(m_reck["area_um2"], m_clem["area_um2"])
        self.assertGreater(m_reck["core_insertion_loss"], m_clem["core_insertion_loss"])
        self.assertGreater(m_reck["energy_pj"], m_clem["energy_pj"])

        archive = ParetoArchive()
        self.assertTrue(archive.add(m_clem, meta={"core_kind": "mzi_mesh_clements"}))
        # Reck 被支配（能量/面积/IL 全面更差）→ 不在前沿上。
        self.assertFalse(archive.add(m_reck, meta={"core_kind": "mzi_mesh_reck"}))
        kinds = {p["meta"]["core_kind"] for p in archive.frontier()}
        self.assertEqual(kinds, {"mzi_mesh_clements"})


if __name__ == "__main__":
    unittest.main()


class TestBottleneckIsParadigmDependent(unittest.TestCase):
    """主导能量器件必须随拓扑范式而变化。

    真实运行记录的证据：
      dptc_crossbar (run_007):      core_dac_0_i1-DAC_2          占能量 61.2%
      mzi_mesh_clements (run_012):  off_chip_laser LT_Laser      占能量 96.7%

    crossbar 的成本由转换器主导；深层 mesh 的成本由激光主导，因为插入损耗决定光功率
    预算。architecture 阶段必须按范式重新诊断，而不是照搬单一根因。
    """

    def _dominant_device(self, path):
        report = json.loads(Path(path).read_text())
        totals = {}

        def walk(node):
            for key, value in node.items():
                if isinstance(value, dict) and "total_energy" in value:
                    totals[key] = float(value["total_energy"] or 0.0)
                elif isinstance(value, dict):
                    walk(value)

        walk(report["by_device"])
        self.assertTrue(totals, path)
        return max(totals, key=totals.get), totals

    def test_crossbar_is_converter_dominated(self):
        dominant, totals = self._dominant_device(
            RECORDED_RUNS / "run_iteration_007/iter_000/energy_detail.json"
        )
        self.assertIn("dac", dominant.lower())
        share = totals[dominant] / sum(totals.values())
        self.assertGreater(share, 0.5, f"{dominant} share {share:.3f}")

    def test_mesh_is_laser_dominated(self):
        dominant, totals = self._dominant_device(
            RECORDED_RUNS / "run_iteration_012/iter_000/energy_detail.json"
        )
        self.assertIn("laser", dominant.lower())
        share = totals[dominant] / sum(totals.values())
        self.assertGreater(share, 0.9, f"{dominant} share {share:.3f}")

    def test_the_two_paradigms_do_not_share_a_root_cause(self):
        crossbar, _ = self._dominant_device(
            RECORDED_RUNS / "run_iteration_007/iter_000/energy_detail.json"
        )
        mesh, _ = self._dominant_device(
            RECORDED_RUNS / "run_iteration_012/iter_000/energy_detail.json"
        )
        self.assertNotEqual(crossbar, mesh)
