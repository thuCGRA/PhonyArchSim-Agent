"""Tests for the search-space validity verification evidence layer."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.core.search_space import LAYERED_DESIGN_SPACE, DesignSpace
from photon_tuner.core.search_space import default_hardware
from photon_tuner.evidence.store import EvidenceStore
from photon_tuner.evidence.sub_arch_map import (
    SUB_ARCH_SCALAR_MAP,
    confirm_device_injection,
    confirm_injection,
    read_sub_arch_param,
)
from photon_tuner.evidence.workload_features import workload_features
from photon_tuner.simulators.analytic import build_sub_arch, load_device_catalog


class TestSubArchMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony"
        cls.catalog = load_device_catalog(str(cls.root / "configs" / "devices"))
        cls.hardware = default_hardware()
        cls.sub_arch = build_sub_arch(cls.hardware, str(cls.root), cls.catalog)

    def test_scalar_map_reads_back_baseline(self):
        ds = DesignSpace(LAYERED_DESIGN_SPACE, device_catalog=self.catalog)
        for param in SUB_ARCH_SCALAR_MAP:
            value = ds.get(self.hardware, param)
            confirmed, summary = confirm_injection(self.sub_arch, param, value)
            self.assertTrue(
                confirmed,
                f"{param}: expected {value!r} but read back {summary['actual']!r}",
            )

    def test_topology_reads_back_baseline(self):
        topo = self.hardware["arch"]["topology"]
        for param, value in [
            ("sharing_policy", topo["sharing_policy"]),
            ("temporal_accum", topo["temporal_accum"]),
            ("laser_distribution", topo["laser_distribution"]),
        ]:
            confirmed, summary = confirm_injection(self.sub_arch, param, value)
            self.assertTrue(
                confirmed,
                f"{param}: expected {value!r} but read back {summary['actual']!r}",
            )

    def test_device_choice_reads_back(self):
        mzm = self.hardware["devices"]["core"]["mzm"]
        pd = self.hardware["devices"]["node"]["photodetector"]
        ok_core, _ = confirm_device_injection(self.sub_arch, "core", "mzm", mzm)
        ok_node, _ = confirm_device_injection(self.sub_arch, "node", "photodetector", pd)
        self.assertTrue(ok_core, "core mzm readback mismatch")
        self.assertTrue(ok_node, "node photodetector readback mismatch")

    def test_false_when_value_absent(self):
        confirmed, _ = confirm_injection(self.sub_arch, "work_freq", 9999)
        self.assertFalse(confirmed)

    def test_instance_choices_read_back(self):
        from photon_tuner.evidence.sub_arch_map import SUB_ARCH_MAP

        hw = default_hardware()
        hw["devices"]["core"]["laser_source"] = "off_chip"
        hw["devices"]["core"]["splitter_kind"] = "mmi"
        sub = build_sub_arch(hw, str(self.root), self.catalog)
        for param, value in [("laser_source", "off_chip"), ("splitter_kind", "mmi")]:
            confirmed, summary = confirm_injection(sub, param, value)
            self.assertTrue(confirmed, f"{param}: {summary}")
            self.assertIn(param, SUB_ARCH_MAP)
        # Defaults read back as the baseline netlist types.
        confirmed, _ = confirm_injection(self.sub_arch, "laser_source", "on_chip")
        self.assertTrue(confirmed)
        confirmed, _ = confirm_injection(self.sub_arch, "splitter_kind", "laser_splitter")
        self.assertTrue(confirmed)


class TestEvidenceStore(unittest.TestCase):
    def _record(self, **overrides):
        record = {
            "param": "work_freq",
            "layer": "optical_core",
            "workload": "vit_tiny",
            "workload_features": {"name": "vit_tiny", "layer_count": 30},
            "old_value": "5",
            "new_value": "10",
            "hardware_map": ["arch", "optical_core", "work_freq"],
            "sub_arch_map": ["core", "work_freq"],
            "constraint_result": "pass",
            "clamp_before": None,
            "injection_confirmed": 1,
            "injection_readback": {"param": "work_freq", "actual": 10},
            "baseline_energy_pj": 1.0,
            "baseline_area_um2": 2.0,
            "baseline_latency_s": 3.0,
            "baseline_objective": 6.0,
            "after_energy_pj": 1.0,
            "after_area_um2": 2.0,
            "after_latency_s": 1.5,
            "after_objective": 3.0,
            "memory_energy_delta_pct": 0.0,
            "metric_deltas": {"latency_pct": -50.0, "objective_pct": -50.0},
            "affected_metrics": ["latency_s", "objective"],
            "is_null": 0,
            "trustworthy": 1,
        }
        record.update(overrides)
        return record

    def test_record_and_query_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence.db")
            store.record(self._record())
            self.assertEqual(store.count(), 1)
            rows = store.query(param="work_freq")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["new_value"], "10")
            # JSON fields decode back to Python objects.
            self.assertIsInstance(rows[0]["metric_deltas"], dict)
            self.assertAlmostEqual(rows[0]["metric_deltas"]["latency_pct"], -50.0)
            store.close()

    def test_query_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence.db")
            store.record(self._record())
            store.record(self._record(workload="vit_base", is_null=1, affected_metrics=[]))
            self.assertEqual(len(store.query(is_null=True)), 1)
            self.assertEqual(len(store.query(workload="vit_base")), 1)
            self.assertEqual(len(store.query(injection_confirmed=False)), 0)
            store.close()

    def test_export_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence.db")
            store.record(self._record())
            out = store.export(Path(tmp) / "evidence.jsonl")
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["param"], "work_freq")
            store.close()


class TestWorkloadFeatures(unittest.TestCase):
    def test_features_extracted(self):
        report = {
            "workload": {
                "layer_sizes": {
                    "layers.0.self_attn.qkv.linear": [1, 2, 3],
                    "layers.0.linear1.linear": [4, 5, 6],
                    "patch_embed.proj.conv": [7, 8, 9],
                },
                "partition_cycles": {
                    "LT": {
                        "layers.0.self_attn.qkv.linear": [48, 64],
                        "layers.0.linear1.linear": [64, 64],
                        "patch_embed.proj.conv": [16, 64],
                    }
                },
            }
        }
        feats = workload_features("vit_tiny", report)
        self.assertEqual(feats["name"], "vit_tiny")
        self.assertEqual(feats["layer_count"], 3)
        self.assertEqual(feats["operator_groups"]["attention"], 1)
        self.assertEqual(feats["operator_groups"]["ffn"], 1)
        self.assertEqual(feats["operator_groups"]["embed"], 1)
        self.assertGreater(feats["total_cycles"], 0)


if __name__ == "__main__":
    unittest.main()
