"""Validate the real-mode hardware injection bridge.

``_inject_hardware`` replaces the fixed-YAML sub-architecture that
``ONNArchSimulator`` resolved with the searched ``hardware`` dict.  These tests
exercise that logic against a minimal mock of the simulator's structure
(``sub_arch_dictionary`` / ``sub_arch_to_layer_mapping`` / ``arch_blocks`` /
``layer_to_sub_arch_mapping``) without importing the real SimPhony stack.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.core.search_space import default_hardware
from photon_tuner.simulators.simphony import _inject_hardware

ROOT = Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony"


class MockSimulator:
    """Minimal stand-in for the attributes ``_inject_hardware`` touches."""

    def __init__(self):
        self.sub_arch_dictionary = {
            "LT": {"tiles": 1, "cores_per_tile": 1, "core": {"height": 8, "width": 8}},
        }
        self.sub_arch_to_layer_mapping = {"LT": ["layer1", "layer2"]}
        self.arch_blocks = {"layer1": [1, 1, 8, 8], "layer2": [1, 1, 8, 8]}
        self.layer_to_sub_arch_mapping = {"layer1": None, "layer2": None}


class TestRealHardwareInjection(unittest.TestCase):
    def test_inject_hardware_updates_all_structures(self):
        sim = MockSimulator()
        _inject_hardware(sim, default_hardware(), str(ROOT))

        sub = sim.sub_arch_dictionary["LT"]
        self.assertEqual(sub["tiles"], 4)
        self.assertEqual(sub["cores_per_tile"], 2)
        self.assertEqual(sub["core"]["height"], 12)
        self.assertEqual(sub["core"]["width"], 12)

        # Per-layer miniblock follows the injected geometry
        # ([tiles, cores_per_tile, height, width]).
        self.assertEqual(sim.arch_blocks["layer1"], [4, 2, 12, 12])
        self.assertEqual(sim.arch_blocks["layer2"], [4, 2, 12, 12])

        # layer->sub-arch mapping now points at the injected config, so the
        # memory workflow reads the injected dataflow / temporal factor.
        self.assertEqual(sim.layer_to_sub_arch_mapping["layer1"]["tiles"], 4)

        # L4 topology overrides landed on the netlist.
        netlist = sub["core"]["netlist"]
        self.assertIn("scaling_rules", netlist)
        self.assertEqual(netlist["temporal_accum_factor"]["duration"], 3)

    def test_inject_hardware_applies_l4_override(self):
        sim = MockSimulator()
        hw = default_hardware()
        hw["arch"]["topology"]["sharing_policy"] = "no_sharing"
        hw["arch"]["topology"]["laser_distribution"] = "H*W*N"
        _inject_hardware(sim, hw, str(ROOT))

        netlist = sim.sub_arch_dictionary["LT"]["core"]["netlist"]
        self.assertEqual(netlist["scaling_rules"]["i2"], "H*W*R*C")
        self.assertEqual(netlist["laser_power_distribution"], "H*W*N")
        self.assertEqual(netlist["laser_power_distribution_factor"], "H*W*N")


if __name__ == "__main__":
    unittest.main()
