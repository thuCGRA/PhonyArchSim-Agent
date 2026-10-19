"""Validate search-space stepping, clamping, and device edits."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.core.search_space import (
    default_hardware,
    materialize_hardware,
    normalize_hardware_to_design_space,
    set_device_choice,
    step_arch_parameter,
)


class TestSearchSpace(unittest.TestCase):
    def setUp(self):
        self.space = {
            "arch": {
                "tiles": [1, 2, 4, 8],
                "cores_per_tile": [1, 2, 4],
                "core_height": [8, 12, 16],
                "core_width": [8, 12, 16],
                "num_wavelength": [8, 12, 16],
                "work_freq": [5, 10, 25],
                "dataflow": ["output_stationary", "weight_stationary", "input_stationary"],
                "precision": {"in_bit": [2, 4, 8], "w_bit": [2, 4, 8], "out_bit": [2, 4, 8]},
            }
        }

    def test_materialize_fills_defaults(self):
        hw = materialize_hardware({})
        self.assertEqual(hw["arch"]["optical_core"]["num_wavelength"], 12)

    def test_instance_choice_defaults_present(self):
        hw = default_hardware()
        self.assertEqual(hw["devices"]["core"]["laser_source"], "on_chip")
        self.assertEqual(hw["devices"]["core"]["splitter_kind"], "laser_splitter")

    def test_step_arch_parameter_advances(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "num_wavelength", 1, self.space)
        self.assertEqual(stepped["arch"]["optical_core"]["num_wavelength"], 16)

    def test_step_arch_parameter_clamps(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "num_wavelength", -1, self.space)
        self.assertEqual(stepped["arch"]["optical_core"]["num_wavelength"], 8)
        # Stepping down again should stay at the minimum.
        again = step_arch_parameter(stepped, "num_wavelength", -1, self.space)
        self.assertEqual(again["arch"]["optical_core"]["num_wavelength"], 8)

    def test_step_dataflow_cycles(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "dataflow", 1, self.space)
        self.assertEqual(stepped["arch"]["system"]["dataflow"], "weight_stationary")

    def test_normalize_clamps_out_of_range(self):
        hw = default_hardware()
        hw["arch"]["optical_core"]["num_wavelength"] = 999
        normalized = normalize_hardware_to_design_space(hw, self.space)
        self.assertEqual(normalized["arch"]["optical_core"]["num_wavelength"], 16)

    def test_set_device_choice(self):
        hw = set_device_choice(default_hardware(), "core", "mzm", "LT_MZM_v2")
        self.assertEqual(hw["devices"]["core"]["mzm"], "LT_MZM_v2")
        # Device cfgs overrides have been removed from the framework.
        self.assertNotIn("device_cfgs", hw)


if __name__ == "__main__":
    unittest.main()
