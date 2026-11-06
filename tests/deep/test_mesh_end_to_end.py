"""通过智能体流水线进行的端到端网格搜索。

脚本化的求解模型提出 crossbar -> MZI 网格的拓扑切换，编排器必须应用它、
评估该网格设计点，并用正确的范式标签将其归档到 Pareto 前沿。
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.llm.config import LLMConfig
from photon_tuner.rag.vector_store import HistoryVectorStore
from photon_tuner.runtime.orchestrator import DeepAgentTuner
from photon_tuner.simulators.simphony import SimphonyAdapter
from photon_tuner.testing.fake_model import FakeToolCallingModel

ROOT = Path(__file__).resolve().parents[2]


class MeshProposingModel(FakeToolCallingModel):
    """假模型：其求解阶段提出 crossbar -> Clements 网格切换。"""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        task = self._detect_task_override(messages)
        if task == "solution_generation":
            fields = {
                "strategy": "switch to a Clements MZI mesh to shrink area at equal MACs",
                "actions": [
                    {"name": "modify_hardware_parameter",
                     "arguments": {"updates": [
                         {"parameter": "core_kind", "value": "mzi_mesh_clements"},
                         {"parameter": "node_kind", "value": "tunable_mzi"},
                     ]}},
                ],
                "link_edits": [],
                "rationale": "Clements mesh: same N(N-1)/2 MZIs, crossing-free, depth N.",
                "coupling_rationale": "mesh keeps work_freq <= DAC/PD/TIA/ADC chain; in/out bit unchanged.",
                "notes": {"expected_objective_direction": "decrease"},
            }
            text = json.dumps({"task": task, **fields}, ensure_ascii=False)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _detect_task_override(self, messages: List[BaseMessage]) -> str:
        for m in reversed(messages):
            if getattr(m, "type", "") != "human":
                continue
            try:
                payload = json.loads(str(m.content))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("task"):
                return str(payload["task"])
        return "unknown"


def _build_tuner(tmp: str, model) -> DeepAgentTuner:
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
        model=model,
    )


class AtomicTemplateModel(FakeToolCallingModel):
    """假模型：其求解阶段提出原子模板切换工具。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        task = self._detect_task_override(messages)
        if task == "solution_generation":
            fields = {
                "strategy": "switch to Clements mesh with the atomic template tool",
                "actions": [
                    {"name": "modify_topology_template",
                     "arguments": {"core_kind": "mzi_mesh_clements"}},
                ],
                "link_edits": [],
                "rationale": "modify_topology_template auto-aligns node_kind to tunable_mzi.",
                "coupling_rationale": "atomic switch keeps core/node coupling consistent.",
                "notes": {"expected_objective_direction": "decrease"},
            }
            text = json.dumps({"task": task, **fields}, ensure_ascii=False)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _detect_task_override(self, messages):
        for m in reversed(messages):
            if getattr(m, "type", "") != "human":
                continue
            try:
                payload = json.loads(str(m.content))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("task"):
                return str(payload["task"])
        return "unknown"


class TestMeshEndToEnd(unittest.TestCase):
    def test_agent_proposes_and_archives_mesh_design_point(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp, MeshProposingModel())
            summary = tuner.tune(initial_hardware=hardware, max_iterations=1)

            # 提案持久化了拓扑切换。
            proposal = read_json(Path(summary["run_dir"]) / "iter_001" / "analysis" / "solution_proposal.json")
            self.assertIn("modify_hardware_parameter", proposal["actions"])

            # 被评估的硬件确实把 core_kind 切换到了网格。
            hw_after = read_json(Path(summary["run_dir"]) / "iter_001" / "hardware.json")
            self.assertEqual(hw_after["arch"]["topology"]["core_kind"], "mzi_mesh_clements")
            self.assertEqual(hw_after["arch"]["topology"]["node_kind"], "tunable_mzi")

            # 被评估的设计点是网格（core_kind 在 normalize 后保留），且其指标被记录。
            m = read_json(Path(summary["run_dir"]) / "iter_001" / "metrics.json")
            self.assertGreater(m["energy_pj"], 0)
            self.assertGreater(m["area_um2"], 0)
            self.assertGreater(m["core_insertion_loss"], 8.0)  # 2.0 laser + depth*MZI_IL

            # 吞吐归一化后 mesh 被初始 crossbar 支配（IL/能耗高、面积效率不占优），
            # 故不进前沿——mesh 的酉通用性不体现在数值效率轴上。
            frontier = summary["pareto_frontier"]
            kinds = {p["meta"]["core_kind"] for p in frontier["points"]}
            self.assertNotIn("mzi_mesh_clements", kinds)
            self.assertIn("dptc_crossbar", kinds)

    def test_atomic_template_switch_applied(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        with tempfile.TemporaryDirectory() as tmp:
            tuner = _build_tuner(tmp, AtomicTemplateModel())
            summary = tuner.tune(initial_hardware=hardware, max_iterations=1)

            hw_after = read_json(Path(summary["run_dir"]) / "iter_001" / "hardware.json")
            self.assertEqual(hw_after["arch"]["topology"]["core_kind"], "mzi_mesh_clements")
            # 原子工具自动对齐 node_kind，因此无需单独的编辑。
            self.assertEqual(hw_after["arch"]["topology"]["node_kind"], "tunable_mzi")


if __name__ == "__main__":
    unittest.main()
