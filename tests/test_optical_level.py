"""Tests for the optical-level agent and its insertion-loss evidence.

Covers the Option-A re-scope: the fourth agent diagnoses the optical link
(insertion-loss + power budgets) and recommends library *variant selections*,
backed by per-component insertion-loss evidence from the analytic simulator.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.agents.optical_level import ALLOWED_TOOLS, OpticalLevelAgent
from photon_tuner.agents.solution import LINK_EDIT_TOOLS
from photon_tuner.common.io import read_json
from photon_tuner.core.schemas import (
    BottleneckState,
    EvaluationResult,
    OpticalAnalysisResult,
)
from photon_tuner.llm.config import LLMConfig
from photon_tuner.core.search_space import default_hardware
from photon_tuner.llm.clients import create_llm_client
from photon_tuner.simulators.simphony import SimphonyAdapter
from photon_tuner.toolbox.device_tools import DeviceToolbox


class TestInsertionLossEvidence(unittest.TestCase):
    def test_per_component_detail_sums_to_path_total(self):
        """The enriched report must decompose each path into components whose
        individual losses sum to the reported path total."""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root=root / "photon_tuner" / "simulators" / "simphony",
                output_root=Path(tmp),
                search_space=read_json(root / "configs" / "search_space.json"),
                mode="analytic",
            )
            result = adapter.evaluate(default_hardware(), 0)
            il = read_json(result.detail_files["insertion_loss"])["LT"]

        for prefix in ("node", "core"):
            detail = il[f"{prefix}_path_detail"]
            total = il[f"{prefix}_insertion_loss"]
            self.assertGreater(len(detail), 0, f"{prefix} path detail empty")
            for comp in detail:
                self.assertIn("instance", comp)
                self.assertIn("device_type", comp)
                self.assertIsNotNone(comp["device_name"], f"{prefix} component missing device_name")
                self.assertIn("insertion_loss_dB", comp)
            self.assertAlmostEqual(
                sum(c["insertion_loss_dB"] for c in detail),
                total,
                places=2,
                msg=f"{prefix} per-component loss does not sum to path total",
            )


class TestOpticalLevelAgent(unittest.TestCase):
    def _toolbox(self):
        root = Path(__file__).resolve().parents[1]
        return DeviceToolbox(
            str(root / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"),
            search_space=read_json(root / "configs" / "search_space.json"),
        )

    def test_analyze_returns_optical_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = create_llm_client(LLMConfig(provider="mock"))
            agent = OpticalLevelAgent(llm, self._toolbox())
            bottleneck = BottleneckState(
                primary_impact="energy",
                dominant_root_cause="device",
                layer_diagnoses=[],
                retrieval_description="test",
                root_cause_summary={},
                recommended_focus=["mzm"],
            )
            evaluation = EvaluationResult(
                run_dir=Path(tmp),
                hardware=default_hardware(),
                metrics={
                    "objective": 1.0,
                    "latency_s": 1.0,
                    "energy_pj": 1.0,
                    "area_um2": 1.0,
                },
                detail_files={},
            )
            result = agent.analyze(
                bottleneck_state=bottleneck,
                evaluation=evaluation,
                output_dir=Path(tmp),
                simulator_schema={"search_space": {}},
            )

            self.assertIsInstance(result, OpticalAnalysisResult)
            self.assertTrue(result.loss_budget_diagnosis)
            self.assertTrue(result.power_budget_margin)
            self.assertIsInstance(result.dominant_loss_contributors, list)
            self.assertIsInstance(result.recommended_link_edits, list)
            self.assertTrue((Path(tmp) / "optical_level_result.json").exists())

    def test_optical_agent_tools_are_selection_only(self):
        # The optical agent must NOT offer the cfgs-override edit tool that
        # fabricates parameter values; it sees the optical link + the device
        # library, and only recommends variant selections.
        self.assertNotIn("step_device_parameter", ALLOWED_TOOLS)
        self.assertNotIn("modify_device_parameter", ALLOWED_TOOLS)
        self.assertIn("summarize_insertion_loss_detail", ALLOWED_TOOLS)
        self.assertIn("summarize_optical_structure", ALLOWED_TOOLS)
        self.assertIn("list_device_candidates", ALLOWED_TOOLS)
        # The only legal device edit (applied by the solution agent) is variant
        # selection, never the fabricated parameter override.
        self.assertEqual(LINK_EDIT_TOOLS, {"modify_device_parameter"})

    def test_solution_rejects_cfgs_override_in_link_edits(self):
        """A link edit carrying ``parameter`` must be rejected: the framework
        removed device-cfgs overrides, so only variant selection is legal."""
        from photon_tuner.agents.solution import SolutionGenerationAgent
        from photon_tuner.core.search_space import default_hardware
        from photon_tuner.toolbox.analysis_tools import AnalysisToolbox

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            agent = SolutionGenerationAgent(
                llm=create_llm_client(LLMConfig(provider="mock")),
                store=None,
                toolbox=AnalysisToolbox(search_space={}),
                device_toolbox=self._toolbox(),
                simulator_schema={"search_space": {}},
                top_k=5,
            )
            _, applied, failed = agent._apply_edits(
                default_hardware(),
                [],
                [{
                    "name": "modify_device_parameter",
                    "arguments": {
                        "level": "core",
                        "device_type": "mzm",
                        "device_name": "SL_MZM",
                        "parameter": "efficiency",
                        "value": 50,
                    },
                }],
                Path(tmp),
                None,
                {},
            )

        self.assertEqual(applied, [])
        self.assertEqual(len(failed), 1)
        self.assertIn("device cfgs overrides are removed", failed[0]["error"])


if __name__ == "__main__":
    unittest.main()
