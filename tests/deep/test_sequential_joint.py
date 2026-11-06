"""顺序搜索与联合搜索的可消融对比。

对主机侧层级过滤器的单元测试，外加一个离线编排测试，
证明两种模式以不同方式应用编辑，并被标注用于消融对比。
"""

import tempfile
import unittest
from pathlib import Path

from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.llm.config import LLMConfig
from photon_tuner.rag.vector_store import HistoryVectorStore
from photon_tuner.runtime.orchestrator import (
    DeepAgentTuner,
    _action_layer,
    _edit_tool_name,
    _filter_edits_for_layer,
)
from photon_tuner.simulators.simphony import SimphonyAdapter
from photon_tuner.testing.fake_model import FakeToolCallingModel

ROOT = Path(__file__).resolve().parents[2]

ARCH_EDIT = {"name": "step_hardware_parameter", "arguments": {"parameter": "cores_per_tile", "direction": -1}}
TOPO_EDIT = {"name": "modify_hardware_parameter", "arguments": {"updates": [{"parameter": "core_kind", "value": "mzi_mesh_clements"}]}}
NODE_EDIT = {"name": "modify_hardware_parameter", "arguments": {"updates": [{"parameter": "node_kind", "value": "tunable_mzi"}]}}
DEVICE_EDIT = {"name": "modify_device_parameter", "arguments": {"level": "core", "device_type": "on_chip_laser", "device_name": "Customized_Laser"}}
MIXED_EDIT = {"name": "modify_hardware_parameter", "arguments": {"updates": [
    {"parameter": "num_wavelength", "value": 16},
    {"parameter": "node_kind", "value": "tunable_mzi"},
]}}


class TestLayerFilter(unittest.TestCase):
    def test_action_layer_classification(self):
        self.assertEqual(_action_layer(ARCH_EDIT), "architecture")
        self.assertEqual(_action_layer(TOPO_EDIT), "topology")
        self.assertEqual(_action_layer(NODE_EDIT), "topology")
        self.assertEqual(_action_layer(DEVICE_EDIT), "device")

    def test_modify_topology_template_classifies_as_topology(self):
        edit = {"name": "modify_topology_template",
                "arguments": {"core_kind": "mzi_mesh_clements"}}
        self.assertEqual(_action_layer(edit), "topology")
        # 在顺序模式下，原子模板切换仅在拓扑轮应用，并原样通过。
        kept, _ = _filter_edits_for_layer([edit], [], "topology")
        self.assertEqual(kept, [edit])
        dropped, _ = _filter_edits_for_layer([edit], [], "architecture")
        self.assertEqual(dropped, [])

    def test_edit_tool_name_accepts_action_alias(self):
        # 模型有时输出 ``action`` 而非 ``name``；应用环节仍须识别该工具，
        # 而不是报 "unknown tool" 失败。
        alias = {"action": "modify_device_parameter",
                 "arguments": {"level": "core", "device_type": "mzm", "device_name": "LT_MZM"}}
        self.assertEqual(_edit_tool_name(alias), "modify_device_parameter")
        self.assertEqual(_action_layer(alias), "device")
        # 空/None 编辑降级为 ""（由第 5 行标记，绝不崩溃）。
        self.assertEqual(_edit_tool_name({}), "")
        self.assertEqual(_edit_tool_name(None), "")

    def test_circuit_override_params_classify_as_topology(self):
        # sharing_policy / temporal_accum / laser_distribution 位于搜索空间的
        # *拓扑*层，可由真实仿真器的 apply_topology 路径求值——它们必须算作拓扑层
        # 编辑而非架构编辑，这样顺序模式才能真正搜索电路轴。
        for param in ("sharing_policy", "temporal_accum", "laser_distribution"):
            edit = {"name": "modify_hardware_parameter",
                    "arguments": {"updates": [{"parameter": param, "value": "x"}]}}
            self.assertEqual(_action_layer(edit), "topology", param)
            step = {"name": "step_hardware_parameter",
                    "arguments": {"parameter": param, "direction": 1}}
            self.assertEqual(_action_layer(step), "topology", param)

    def test_joint_keeps_everything(self):
        actions, links = _filter_edits_for_layer([ARCH_EDIT, TOPO_EDIT], [DEVICE_EDIT], "joint")
        self.assertEqual(len(actions), 2)
        self.assertEqual(len(links), 1)

    def test_architecture_layer_only(self):
        actions, links = _filter_edits_for_layer([ARCH_EDIT, TOPO_EDIT], [DEVICE_EDIT], "architecture")
        self.assertEqual([a.get("name") for a in actions], ["step_hardware_parameter"])
        self.assertEqual(links, [])

    def test_topology_layer_only(self):
        actions, links = _filter_edits_for_layer([ARCH_EDIT, TOPO_EDIT], [DEVICE_EDIT], "topology")
        self.assertEqual([a.get("name") for a in actions], ["modify_hardware_parameter"])
        self.assertEqual(links, [])

    def test_device_layer_only(self):
        actions, links = _filter_edits_for_layer([ARCH_EDIT, TOPO_EDIT], [DEVICE_EDIT], "device")
        self.assertEqual(actions, [])
        self.assertEqual(len(links), 1)

    def test_mixed_update_split_by_layer(self):
        # 架构迭代把混合动作拆分成仅其架构更新。
        actions, _ = _filter_edits_for_layer([MIXED_EDIT], [], "architecture")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["arguments"]["updates"], [{"parameter": "num_wavelength", "value": 16}])
        actions, _ = _filter_edits_for_layer([MIXED_EDIT], [], "topology")
        self.assertEqual(actions[0]["arguments"]["updates"], [{"parameter": "node_kind", "value": "tunable_mzi"}])


def _build_tuner(tmp: str) -> DeepAgentTuner:
    ss = read_json(ROOT / "configs" / "search_space.json")
    adapter = SimphonyAdapter(
        simphony_root=ROOT / "photon_tuner" / "simulators" / "simphony",
        output_root=Path(tmp),
        search_space=ss,
        mode="analytic",
        workload_name="vit_tiny",
    )
    return DeepAgentTuner(
        adapter=adapter,
        llm_config=LLMConfig(provider="mock"),
        devicelib_root=str(ROOT / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"),
        output_root=Path(tmp) / "runs",
        search_space=ss,
        store=HistoryVectorStore(Path(tmp) / "history.db"),
        model=FakeToolCallingModel(),
    )


class TestSequentialVsJoint(unittest.TestCase):
    def _run(self, tmp, search_strategy, iterations):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        tuner = _build_tuner(tmp)
        summary = tuner.tune(
            initial_hardware=hardware,
            max_iterations=iterations,
            search_strategy=search_strategy,
        )
        return summary

    def test_joint_mode_applies_all_layers_every_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(tmp, "joint", 1)
            self.assertEqual(summary["search_strategy"], "joint")
            proposal = read_json(Path(summary["run_dir"]) / "iter_001" / "analysis" / "solution_proposal.json")
            self.assertEqual(proposal["focus_layer"], "joint")
            # 假模型提出 arch + device；joint 应用两者，不推迟任何一项。
            self.assertEqual(sorted(proposal["applied_actions"]), ["step_hardware_parameter"])
            self.assertEqual(len(proposal["applied_link_edits"]), 1)
            self.assertEqual(proposal["deferred_edits"], [])

    def test_sequential_mode_cycles_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(tmp, "sequential", 3)
            self.assertEqual(summary["search_strategy"], "sequential")
            run_dir = Path(summary["run_dir"])
            expected = ["architecture", "topology", "device"]
            for i, layer in enumerate(expected, start=1):
                proposal = read_json(run_dir / f"iter_{i:03d}" / "analysis" / "solution_proposal.json")
                self.assertEqual(proposal["focus_layer"], layer, f"iter {i}")

            # 迭代 1（architecture）：应用 arch，推迟 device。
            p1 = read_json(run_dir / "iter_001" / "analysis" / "solution_proposal.json")
            self.assertEqual(p1["applied_actions"], ["step_hardware_parameter"])
            self.assertEqual(p1["applied_link_edits"], [])
            self.assertEqual(p1["deferred_edits"], ["modify_device_parameter"])

            # 迭代 3（device）：应用 device，推迟 arch。
            p3 = read_json(run_dir / "iter_003" / "analysis" / "solution_proposal.json")
            self.assertEqual(p3["applied_actions"], [])
            self.assertEqual(len(p3["applied_link_edits"]), 1)
            self.assertEqual(p3["deferred_edits"], ["step_hardware_parameter"])

    def test_pareto_meta_tagged_with_search_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(tmp, "sequential", 1)
            meta = summary["pareto_frontier"]["points"][0]["meta"]
            self.assertEqual(meta["search_strategy"], "sequential")

    def test_invalid_search_strategy_raises(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp)
            with self.assertRaises(ValueError):
                tuner.tune(initial_hardware=hardware, max_iterations=1, search_strategy="bogus")


class TestTopologySweep(unittest.TestCase):
    def test_sweep_archives_every_evaluable_template(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp)
            summary = tuner.tune(
                initial_hardware=hardware, max_iterations=1, topology_sweep=True,
            )
            kinds = {p["meta"]["core_kind"] for p in summary["pareto_frontier"]["points"]}
            # 扫描评估全部 7 个可求值模板。吞吐归一化后（能耗→pJ/MAC、面积→单位吞吐），
            # 前沿由并行权重阵列（MRR/PCM/WDM，低 IL/能耗）与 crossbar（dptc，矩阵并行、
            # 低延迟）共同构成；butterfly 与 Clements/Reck mesh 因 IL 高、能耗高且面积
            # 效率不占优而被支配。
            #
            # 注意历史上这里断言 >=5——但那个"第 5 种范式"是 tempo_crossbar，
            # 它与 dptc_crossbar 逐位相同，等于用一份重复的评估结果虚增了前沿多样性。
            # 删除后前沿是诚实的 4 个范式，因此改为精确断言。
            self.assertIn("mrr_weight_bank", kinds)
            self.assertIn("pcm_crossbar", kinds)
            self.assertIn("wdm_link", kinds)
            self.assertIn("dptc_crossbar", kinds)
            self.assertEqual(len(kinds), 4, f"前沿拓扑种类应为 4，实际 {sorted(kinds)}")

    def test_sweep_off_keeps_single_paradigm(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp)
            summary = tuner.tune(initial_hardware=hardware, max_iterations=1)
            kinds = {p["meta"]["core_kind"] for p in summary["pareto_frontier"]["points"]}
            self.assertEqual(kinds, {"dptc_crossbar"})


if __name__ == "__main__":
    unittest.main()
