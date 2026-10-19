"""Offline end-to-end smoke test of the multi-agent photonic DSE loop."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.common.io import read_json
from photon_tuner.llm.config import LLMConfig
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.llm.clients import create_llm_client
from photon_tuner.runtime.tuner import MultiAgentTuner
from photon_tuner.rag.vector_store import HistoryVectorStore
from photon_tuner.simulators.simphony import SimphonyAdapter


class TestPipeline(unittest.TestCase):
    def test_offline_end_to_end(self):
        root = Path(__file__).resolve().parents[1]
        search_space = read_json(root / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), search_space)
        llm = create_llm_client(LLMConfig(provider="mock"))

        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root=root / "photon_tuner" / "simulators" / "simphony",
                output_root=Path(tmp),
                search_space=search_space,
                mode="analytic",
            )
            tuner = MultiAgentTuner(
                adapter=adapter,
                llm_client=llm,
                devicelib_root=str(root / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"),
                output_root=Path(tmp),
                search_space=search_space,
                store=HistoryVectorStore(Path(tmp) / "history.db"),
            )
            summary = tuner.tune(initial_hardware=hardware, max_iterations=2)

        self.assertEqual(summary["iterations"], 2)
        self.assertIn("objective", summary["best"])
        # The mock solution improves energy, so the best objective must be below
        # the initial baseline objective.
        self.assertGreater(summary["history"][0]["metrics"]["objective"], summary["best"]["objective"])

    def test_offline_end_to_end_with_real_workload(self):
        """P0: the loop must run and persist a real per-layer workload."""
        root = Path(__file__).resolve().parents[1]
        search_space = read_json(root / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), search_space)
        llm = create_llm_client(LLMConfig(provider="mock"))

        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root=root / "photon_tuner" / "simulators" / "simphony",
                output_root=Path(tmp),
                search_space=search_space,
                mode="analytic",
                workload_name="vit_tiny",
            )
            tuner = MultiAgentTuner(
                adapter=adapter,
                llm_client=llm,
                devicelib_root=str(root / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"),
                output_root=Path(tmp),
                search_space=search_space,
                store=HistoryVectorStore(Path(tmp) / "history.db"),
            )
            summary = tuner.tune(initial_hardware=hardware, max_iterations=1)
            workload = read_json(Path(tmp) / "run_iteration_001" / "iter_000" / "workload.json")

        self.assertEqual(summary["iterations"], 1)
        self.assertEqual(workload["name"], "vit_tiny")
        # patch_embed + 4 blocks x 6 sub-layers
        self.assertEqual(len(workload["partition_cycles"]["LT"]), 25)
        self.assertEqual(adapter.schema()["workload"], "vit_tiny")

    def test_auto_mode_fallback_forwards_workload(self):
        """Regression: the ``auto`` mode ImportError fallback dropped the workload.

        ``auto`` is the CLI default, so this path -- not the explicit
        ``analytic`` one -- is what users actually hit when torch is absent.
        """
        root = Path(__file__).resolve().parents[1]
        search_space = read_json(root / "configs" / "search_space.json")
        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root=root / "photon_tuner" / "simulators" / "simphony",
                output_root=Path(tmp),
                search_space=search_space,
                mode="auto",
                workload_name="vit_tiny",
            )
            result = adapter.evaluate(default_hardware(), 0)
            workload = read_json(result.detail_files["workload"])
            report = read_json(result.detail_files["report"])

        self.assertEqual(workload["name"], "vit_tiny")
        self.assertEqual(len(workload["partition_cycles"]["LT"]), 25)
        # The native per-layer energy breakdown must survive the fallback too.
        self.assertIn("by_layer", report["energy"])

    def test_report_files_written(self):
        root = Path(__file__).resolve().parents[1]
        search_space = read_json(root / "configs" / "search_space.json")
        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root=root / "photon_tuner" / "simulators" / "simphony",
                output_root=Path(tmp),
                search_space=search_space,
                mode="analytic",
            )
            result = adapter.evaluate(default_hardware(), iteration=0)
            for name in ("report", "energy", "area", "insertion_loss", "workload", "memory"):
                self.assertTrue(result.detail_files[name].exists(), f"{name} detail missing")
            report = read_json(result.detail_files["report"])
            self.assertIn("metrics", report)
            self.assertIn("energy", report)


if __name__ == "__main__":
    unittest.main()
