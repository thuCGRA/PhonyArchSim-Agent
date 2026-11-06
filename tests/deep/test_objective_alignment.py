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
            # 随 miniblock 拓扑敏感化而更新：工作负载循环现用 [tiles, cores,
            # matrix_height, matrix_width]（12x12 crossbar → arch 48x24）而非固定
            # [2,2,4,4]，iter 大幅减少、延迟下降。
            self.assertAlmostEqual(obj, 433.7975586049893, places=6)

    def test_deterministic_sweep_improves_baseline(self):
        """离线端到端目标改进：一次合法的确定性单变量编辑必须把目标降到基线以下（不使用 LLM）。"""
        with tempfile.TemporaryDirectory() as tmp:
            tuner, ss = _build_tuner(tmp, "vit_tiny")
            base_hw = normalize_hardware_to_design_space(default_hardware(), ss)
            base = tuner._evaluate(base_hw, 0).metrics["objective"]

            improved = False
            for nw in ss.get("optical_core", {}).get("num_wavelength", [16]):
                hw = normalize_hardware_to_design_space(default_hardware(), ss)
                hw["arch"]["optical_core"]["num_wavelength"] = nw
                hw = normalize_hardware_to_design_space(hw, ss)
                cand = tuner._evaluate(hw, 1).metrics["objective"]
                if cand < base:
                    improved = True
                    break
            self.assertTrue(improved, "no legal deterministic edit improved the objective")

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
