"""LangGraph DSE 流水线的离线端到端测试（使用假模型）。"""

import tempfile
import unittest
from pathlib import Path

from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.llm.config import LLMConfig
from photon_tuner.rag.vector_store import HistoryVectorStore
from photon_tuner.runtime.orchestrator import DeepAgentTuner
from photon_tuner.simulators.simphony import SimphonyAdapter
from photon_tuner.testing.fake_model import FakeToolCallingModel

ROOT = Path(__file__).resolve().parents[2]


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


class TestDeepPipeline(unittest.TestCase):
    def test_agents_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp)
            agents = tuner._build_agents()
            self.assertEqual(set(agents), {"model", "architecture", "topology", "device", "solution", "evaluate"})

    def test_all_tools_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp)
            names = set(tuner.all_tools)
            self.assertEqual(len(names), 28)  # 22 个工具箱工具 + 6 个拓扑工具
            self.assertIn("summarize_topology", names)
            self.assertIn("inspect_light_path", names)
            self.assertIn("list_topology_templates", names)
            self.assertIn("validate_coupling", names)
            self.assertIn("searchable_parameters", names)
            self.assertIn("summarize_device_frontier", names)
            self.assertIn("summarize_topology_evidence", names)

    def test_offline_end_to_end_improves(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp)
            summary = tuner.tune(initial_hardware=hardware, max_iterations=2)
            baseline = summary["history"][0]["metrics"]["objective"]
            self.assertLess(summary["best"]["objective"], baseline)
            self.assertEqual(summary["iterations"], 2)
            # 五个阶段产物外加 trajectory。
            artifacts = sorted(p.name for p in Path(summary["run_dir"]).rglob("analysis/*.json"))
            for name in ("model_level_result.json", "architecture_result.json",
                         "topology_result.json", "device_result.json", "solution_proposal.json"):
                self.assertIn(name, artifacts)
            self.assertTrue((Path(summary["run_dir"]) / "trajectory.json").exists())
            # 每个阶段都会写出自己的检查点门控报告（5 个检查点）。
            for name in ("model_gate_report.json", "architecture_gate_report.json",
                         "topology_gate_report.json", "device_gate_report.json",
                         "gate_report.json"):
                self.assertTrue((Path(summary["run_dir"]) / "iter_002" / "analysis" / name).exists())
            # 返工会保留首次尝试的门控报告（含评判结论），使触发返工的评估可追溯。
            first_attempts = sorted(
                p.name for p in (Path(summary["run_dir"]) / "iter_002" / "analysis")
                .glob("*_gate_report_first_attempt.json")
            )
            self.assertTrue(first_attempts)
            self.assertIn("model_gate_report_first_attempt.json", first_attempts)
            # 假模型不产生工具记录，因此第 4 行在每个阶段都会失败。每轮重执行
            # 预算（max_stage_rework=3）在四个分析阶段间共享，求解阶段则拥有独立的
            # max_gate_retries=2 重新提议额度。
            gate = read_json(Path(summary["run_dir"]) / "iter_002" / "analysis" / "gate_report.json")
            self.assertEqual(gate["final"]["decision"]["decision"], "request-info")
            self.assertLessEqual(gate["retried"], 2)  # 受 max_gate_retries 限制
            # 全轮重执行预算被遵守：每迭代 <=3（分析阶段）+ <=2（求解）。
            self.assertLessEqual(summary["rework_total"], 5 * summary["iterations"])
            # 协调器的跨层一致性依据被持久化。
            proposal = read_json(Path(summary["run_dir"]) / "iter_002" / "analysis" / "solution_proposal.json")
            self.assertIn("coupling_rationale", proposal)

    def test_deterministic_across_runs(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        bests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                tuner = _build_tuner(tmp)
                bests.append(tuner.tune(initial_hardware=hardware, max_iterations=1)["best"]["objective"])
        self.assertEqual(bests[0], bests[1])


if __name__ == "__main__":
    unittest.main()
