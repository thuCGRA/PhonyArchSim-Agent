"""Structural and feasibility checks for grouped MRR weight banks."""

from copy import deepcopy
from pathlib import Path
import unittest

from photon_tuner.core.search_space import DesignSpace, default_hardware, switch_topology_template
from photon_tuner.core.topology_space import validate_coupling
from photon_tuner.evidence.sub_arch_map import confirm_injection
from photon_tuner.simulators.netlist import build_weight_array_sub_arch, load_device_catalog


SIM_ROOT = str(Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony")


class TestWeightBankGrouping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_device_catalog(f"{SIM_ROOT}/configs/devices")

    def _hardware(self, groups=2, rank=16, kind="mrr_weight_bank"):
        hardware = switch_topology_template(default_hardware(), kind)
        topology = hardware["arch"]["topology"]
        topology.update(weight_bank_groups=groups, mesh_rank=rank)
        return hardware

    def _build(self, groups=2, rank=16, kind="mrr_weight_bank"):
        return build_weight_array_sub_arch(self._hardware(groups, rank, kind), SIM_ROOT, self.catalog, kind)

    def test_group_count_keeps_workload_geometry_and_primitive_totals(self):
        single = self._build(groups=1)["core"]
        double = self._build(groups=2)["core"]
        for key in ("matrix_height", "matrix_width", "num_wavelength", "weight_count", "range_forwards"):
            self.assertEqual(single[key], double[key], key)
        self.assertEqual((double["matrix_height"], double["matrix_width"]), (1, 1))
        self.assertEqual(double["num_wavelength"], 16)
        self.assertEqual(double["group_wavelengths"], 8)
        self.assertEqual(double["node"]["netlist"]["scaling_rules"], single["node"]["netlist"]["scaling_rules"])
        core_netlist = double["netlist"]
        rules = core_netlist["scaling_rules"]
        self.assertEqual(
            [rules[name] for name, spec in core_netlist["instances"].items() if spec[0] == "dac"],
            ["8*R*C", "8*R*C"],
        )

    def test_evidence_reads_back_disjoint_banks_and_actual_paths(self):
        for groups in (1, 2):
            sub_arch = self._build(groups=groups)
            confirmed, summary = confirm_injection(sub_arch, "weight_bank_groups", groups)
            self.assertTrue(confirmed, summary)
            self.assertEqual(summary["netlist_instance_counts"]["adc"], groups)

        sub_arch = self._build()
        bad = deepcopy(sub_arch)
        netlist = bad["core"]["netlist"]
        edge, _ = next(
            (key, endpoints) for key, endpoints in netlist["nets"].items()
            if netlist["instances"][endpoints[1].partition("(")[0]][0] == "photodetector"
        )
        del netlist["nets"][edge]
        confirmed, summary = confirm_injection(bad, "weight_bank_groups", 2)
        self.assertFalse(confirmed)
        self.assertFalse(summary["independent_readout_paths"])

    def test_two_banks_preserve_the_ideal_dot_product_and_add_a_guard_bit(self):
        sub_arch = self._build()
        core = sub_arch["core"]
        partition = core["weight_bank_partition"]
        bands = partition["wavelength_indices"]
        self.assertEqual(sorted(bands[0] + bands[1]), list(range(16)))
        self.assertFalse(set(bands[0]) & set(bands[1]))
        values = [(-1) ** index * index for index in range(16)]
        weights = [2 * index - 7 for index in range(16)]
        partials = [sum(values[index] * weights[index] for index in band) for band in bands]
        self.assertEqual(sum(partials), sum(value * weight for value, weight in zip(values, weights)))
        bitwidth = core["precision"]["out_bit"]
        self.assertEqual(partition["accumulator_bits"], bitwidth + 1)
        signed_min, signed_max = -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1
        wide_min, wide_max = -(2 ** bitwidth), 2 ** bitwidth - 1
        self.assertLessEqual(wide_min, 2 * signed_min)
        self.assertGreaterEqual(wide_max, 2 * signed_max)

    def test_unsupported_grouping_is_rejected(self):
        for kwargs in ({"groups": 3}, {"rank": 15}, {"kind": "pcm_crossbar"}, {"kind": "wdm_link"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._build(**kwargs)
        for kind in ("dptc_crossbar", "mzi_mesh_clements"):
            with self.subTest(kind=kind):
                violations = validate_coupling(self._hardware(kind=kind), mode="real")
                self.assertIn("weight_bank_groups", {entry["scope"] for entry in violations})
        violations = validate_coupling(self._hardware(), mode="analytic")
        self.assertIn("weight_bank_groups", {entry["scope"] for entry in violations})

    def test_axis_is_only_searchable_for_the_real_mrr_template(self):
        space = DesignSpace()
        mrr = self._hardware()
        self.assertEqual(space.layer_of("weight_bank_groups"), "topology")
        self.assertEqual(space.effective_candidates("weight_bank_groups", mrr, "real"), [1, 2])
        self.assertFalse(space.effective_candidates("weight_bank_groups", mrr, "analytic"))
        for kind in ("pcm_crossbar", "wdm_link", "mzi_mesh_clements", "dptc_crossbar"):
            hardware = self._hardware(groups=1, kind=kind)
            self.assertFalse(space.effective_candidates("weight_bank_groups", hardware, "real"))
