"""Regression checks for topology-conditioned architecture/device search axes."""

import unittest

from photon_tuner.core.search_space import (
    DesignSpace,
    default_hardware,
    normalize_hardware_to_design_space,
    set_device_choice,
    switch_topology_template,
)
from photon_tuner.evidence.sub_arch_map import confirm_device_injection
from photon_tuner.core.topology_space import COUPLING_MATRIX


class TestTopologyConditionedSpace(unittest.TestCase):
    def test_normalization_repairs_core_node_pair_atomically(self):
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_clements"
        hw["arch"]["topology"]["node_kind"] = "dode"

        normalized = normalize_hardware_to_design_space(hw)
        topology = normalized["arch"]["topology"]
        self.assertIn(topology["node_kind"], COUPLING_MATRIX["mzi_mesh_clements"]["node_kinds"])
        self.assertEqual(topology["node_kind"], "tunable_mzi")

    def test_node_kind_is_not_an_independent_axis_when_pair_is_one_to_one(self):
        ds = DesignSpace()
        hw = default_hardware()
        self.assertEqual(ds.effective_candidates("node_kind", hw, "real"), [])
        entry = ds.scope_table("dptc_crossbar", "real")["params"]["node_kind"]
        self.assertEqual(entry["status"], "determined_by_core_kind")
        self.assertEqual(entry["candidates"], ["dode"])

    def test_real_core_candidates_are_only_registered_real_templates(self):
        ds = DesignSpace()
        hw = default_hardware()
        candidates = ds.effective_candidates("core_kind", hw, "real")
        self.assertTrue(candidates)
        self.assertTrue(all(ds.param_spec("core_kind")["values"].__contains__(x) for x in candidates))

    def test_device_cfg_uses_active_topology_bank(self):
        catalog = {
            "dac": {
                "crossbar_dac": {"cfgs": {"sample_rate": 5}},
                "mesh_dac": {"cfgs": {"sample_rate": 16}},
            }
        }
        ds = DesignSpace(device_catalog=catalog)
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_clements"
        hw["devices"]["mesh"]["core"]["dac"] = "mesh_dac"
        hw["devices"]["core"]["dac"] = "crossbar_dac"
        self.assertEqual(ds._device_cfg(hw, "dac")["sample_rate"], 16)

    def test_weight_array_device_choice_and_readback(self):
        hw = switch_topology_template(default_hardware(), "mrr_weight_bank")
        choice = set_device_choice(
            hw, "weight_array_core", "photodetector", "ACADEMIA_Analog_PD"
        )
        sub_arch = {
            "core": {
                "devices": {"photodetector": ["ACADEMIA_Analog_PD"]}
            }
        }
        confirmed, _ = confirm_device_injection(
            sub_arch, "weight_array_core", "photodetector", "ACADEMIA_Analog_PD"
        )
        self.assertTrue(confirmed)
        self.assertEqual(
            choice["devices"]["weight_array"]["core"]["photodetector"], "ACADEMIA_Analog_PD"
        )


if __name__ == "__main__":
    unittest.main()
