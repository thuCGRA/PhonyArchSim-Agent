"""针对共享遗留域路径的目标对齐验证。

LangGraph 流水线必须通过*相同的* ``design_space.normalize -> adapter.evaluate ->
metrics_with_objective`` 路径评估硬件点。这些测试证明编排器的 ``_evaluate``
对基线工作负载集与该直接路径逐位一致，并钉住规范的 vit_tiny 基线作为回归锚点。
"""

import tempfile
import unittest
from pathlib import Path

from photon_tuner.common.io import read_json
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space, step_arch_parameter
from photon_tuner.llm.config import LLMConfig
from photon_tuner.rag.vector_store import HistoryVectorStore
from photon_tuner.runtime.orchestrator import DeepAgentTuner
from photon_tuner.simulators.simphony import SimphonyAdapter
from photon_tuner.testing.fake_model import FakeToolCallingModel

ROOT = Path(__file__).resolve().parents[2]


def _build_tuner(tmp: str, workload: str) -> tuple[DeepAgentTuner, dict]:
    ss = read_json(ROOT / "configs" / "search_space.json")
    adapter = SimphonyAdapter(
        simphony_root=ROOT / "photon_tuner" / "simulators" / "simphony",
        output_root=Path(tmp),
        search_space=ss,
        mode="analytic",
        workload_name=workload,
    )
    tuner = DeepAgentTuner(
        adapter=adapter,
        llm_config=LLMConfig(provider="mock"),
        devicelib_root=str(ROOT / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"),
        output_root=Path(tmp) / "runs",
        search_space=ss,
        store=HistoryVectorStore(Path(tmp) / "history.db"),
        model=FakeToolCallingModel(),
    )
    return tuner, ss


class TestObjectiveAlignment(unittest.TestCase):
    def test_evaluate_matches_direct_domain_path(self):
        """验证 Orchestrator._evaluate 与直接（normalize -> evaluate -> objective）路径一致。"""
        for workload in ("vit_tiny", "vit_base"):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(workload=workload):
                tuner, ss = _build_tuner(tmp, workload)
                hw = normalize_hardware_to_design_space(default_hardware(), ss)

                via_tuner = tuner._evaluate(hw, 0).metrics

                norm = tuner.design_space.normalize(hw)
                direct = tuner.adapter.evaluate(norm, 0)
                direct.metrics = metrics_with_objective(direct.metrics, direct.hardware, ss)

                self.assertEqual(via_tuner, direct.metrics)

    def test_vit_tiny_baseline_objective_pinned(self):
        """回归锚点：规范的 vit_tiny 基线目标值。"""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, ss = _build_tuner(tmp, "vit_tiny")
            hw = normalize_hardware_to_design_space(default_hardware(), ss)
            obj = tuner._evaluate(hw, 0).metrics["objective"]
            # 锚点演进历史（每一次都是刻意的建模/搜索空间修正，不是漂移）：
            #   433.7976 —— 12x12x12 @5 GHz / 485 KB（旧基线，不在 2 的幂网格上）
            #   418.7218 —— 16x16x16 @4 GHz / 512 KB（网格对齐后）
            #   5897.5086 —— **能耗口径修正**：目标此前只计计算能耗，而延时却含
            #               存储延时。实测存储占总能耗 75–90%（vit_tiny 为 93%），
            #               单计计算能耗会把优化方向系统性带偏。现目标取
            #               计算 + 存储，故量级上升约 14 倍。
            # 该改动的正当性在于口径一致性：计时与计价必须覆盖同一物理范围。
            self.assertAlmostEqual(obj, 5897.508605748704, places=6)

    def test_deterministic_sweep_improves_baseline(self):
        """离线端到端目标改进：一次合法的确定性单变量编辑必须把目标降到基线以下（不使用 LLM）。"""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, ss = _build_tuner(tmp, "vit_tiny")
            base_hw = normalize_hardware_to_design_space(default_hardware(), ss)
            base = tuner._evaluate(base_hw, 0).metrics["objective"]

            # 从分层规格读全部架构层候选。旧写法 ss["optical_core"] 并不存在
            # （规格是 {"layers": {...}}），.get 兜底会静默退化成单个值 [16]，
            # 使这条断言变成空转；改为遍历架构层所有数值轴。
            spec = (ss.get("layers") or {}).get("architecture", {}).get("params", {})
            improved = False
            found = None
            for param in ("work_freq", "cores_per_tile", "hbm_bandwidth", "tiles"):
                path = (spec.get(param) or {}).get("map")
                if not path:
                    continue
                for value in (spec.get(param) or {}).get("values", []):
                    hw = normalize_hardware_to_design_space(default_hardware(), ss)
                    node = hw
                    for key in path[:-1]:
                        node = node[key]
                    node[path[-1]] = value
                    hw = normalize_hardware_to_design_space(hw, ss)
                    cand = tuner._evaluate(hw, 1).metrics["objective"]
                    if cand < base:
                        improved = True
                        found = (param, value, cand)
                        break
                if improved:
                    break
            self.assertTrue(improved, "no legal deterministic edit improved the objective")
            # 同时确认改善是真实的、不是浮点噪声。
            self.assertLess(found[2], base * 0.99, f"{found[0]}={found[1]} barely moved")

    def test_stepped_hardware_aligned(self):
        """合法编辑通过深度流水线的评估与直接路径一致。"""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, ss = _build_tuner(tmp, "vit_tiny")
            hw = normalize_hardware_to_design_space(default_hardware(), ss)
            stepped = step_arch_parameter(hw, "cores_per_tile", -1, ss)

            via_tuner = tuner._evaluate(stepped, 1).metrics
            direct = tuner.adapter.evaluate(tuner.design_space.normalize(stepped), 1)
            direct.metrics = metrics_with_objective(direct.metrics, direct.hardware, ss)

            self.assertEqual(via_tuner, direct.metrics)


if __name__ == "__main__":
    unittest.main()
