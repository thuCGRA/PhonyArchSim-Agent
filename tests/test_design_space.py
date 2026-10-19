"""Validate the hierarchical / multi-granularity / adaptive design space."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.core.search_space import (
    LAYERED_DESIGN_SPACE,
    DesignSpace,
    build_design_space,
)
from photon_tuner.core.search_space import (
    default_hardware,
    modify_arch_parameter,
    normalize_hardware_to_design_space,
    step_arch_parameter,
)


# Minimal device catalog mirroring discover_device_catalog()'s shape.
CATALOG = {
    "mzm": {"LT_MZM": {"cfgs": {"modulation_speed": 25, "testing_bit": 1}, "type": "MZM"}},
    "dac": {"DAC_2": {"cfgs": {"prec": 8}, "type": "DAC"}},
    "adc": {"ADC_SAR_1": {"cfgs": {"prec": 8}, "type": "ADC"}},
}

# Strict catalog: slow MZM + low-precision DAC/ADC, so feasibility bites.
CATALOG_STRICT = {
    "mzm": {"LT_MZM": {"cfgs": {"modulation_speed": 10, "testing_bit": 1}, "type": "MZM"}},
    "dac": {"DAC_2": {"cfgs": {"prec": 4}, "type": "DAC"}},
    "adc": {"ADC_SAR_1": {"cfgs": {"prec": 4}, "type": "ADC"}},
}


class TestDesignSpaceHierarchy(unittest.TestCase):
    def test_five_layers_present(self):
        ds = DesignSpace()
        self.assertEqual(
            set(ds.layers.keys()),
            {"system", "optical_core", "topology", "device", "device_param"},
        )

    def test_layer_of_routes_params(self):
        ds = DesignSpace()
        self.assertEqual(ds.layer_of("tiles"), "system")
        self.assertEqual(ds.layer_of("work_freq"), "optical_core")
        self.assertEqual(ds.layer_of("hbm_bandwidth"), "system")
        self.assertEqual(ds.layer_of("sharing_policy"), "topology")

    def test_device_param_layer_is_fixed(self):
        ds = DesignSpace()
        self.assertEqual(ds.layers["device_param"]["kind"], "fixed")
        self.assertEqual(ds.layers["device"]["kind"], "selection")


class TestMultiGranularity(unittest.TestCase):
    def test_kinds_are_declared(self):
        ds = DesignSpace()
        self.assertEqual(ds.param_spec("dataflow")["kind"], "enum")
        self.assertEqual(ds.param_spec("precision_in_bit")["kind"], "bits")
        self.assertEqual(ds.param_spec("work_freq")["kind"], "int")

    def test_work_freq_candidates(self):
        ds = DesignSpace()
        self.assertEqual(ds.candidates("work_freq"), [5, 10, 15, 20, 25])

    def test_memory_params_are_searchable(self):
        ds = DesignSpace()
        self.assertEqual(ds.candidates("hbm_bandwidth"), [512, 1024, 2048])
        self.assertEqual(ds.candidates("glb1_cache_size"), [64, 256, 485, 1024, 4096])
        self.assertEqual(ds.param_spec("glb1_cache_size")["mode"], "real")
        self.assertEqual(ds.param_spec("hbm_bandwidth")["mode"], "both")

    def test_topology_params_are_enums(self):
        ds = DesignSpace()
        self.assertEqual(ds.param_spec("sharing_policy")["kind"], "enum")
        self.assertEqual(ds.candidates("sharing_policy"), ["no_sharing", "adc_shared", "adc_tia_shared"])
        self.assertEqual(ds.candidates("temporal_accum"), ["off", "default", "deep"])
        self.assertEqual(ds.candidates("laser_distribution"), ["H*W", "H*W*N", "H*W*N*C"])


class TestRealFeasibility(unittest.TestCase):
    def test_default_hardware_is_feasible(self):
        ds = DesignSpace(device_catalog=CATALOG)
        self.assertEqual(ds.violations(default_hardware()), [])

    def test_work_freq_violation_detected_and_clamped(self):
        ds = DesignSpace(device_catalog=CATALOG_STRICT)
        hw = default_hardware()
        hw["arch"]["optical_core"]["work_freq"] = 25  # > MZM symbol rate 10 GHz
        violations = ds.violations(hw)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["param"], "work_freq")
        fixed = ds.normalize(hw)
        self.assertEqual(fixed["arch"]["optical_core"]["work_freq"], 10)

    def test_precision_violation_detected_and_clamped(self):
        ds = DesignSpace(device_catalog=CATALOG_STRICT)
        hw = default_hardware()
        hw["arch"]["optical_core"]["precision"]["in_bit"] = 8  # > DAC prec 4
        hw["arch"]["optical_core"]["precision"]["out_bit"] = 8  # > ADC prec 4
        violations = {v["param"] for v in ds.violations(hw)}
        self.assertEqual(violations, {"precision_in_bit", "precision_out_bit"})
        fixed = ds.normalize(hw)
        self.assertEqual(fixed["arch"]["optical_core"]["precision"]["in_bit"], 4)
        self.assertEqual(fixed["arch"]["optical_core"]["precision"]["out_bit"], 4)

    def test_no_catalog_no_violations(self):
        ds = DesignSpace()  # no device_catalog
        hw = default_hardware()
        hw["arch"]["optical_core"]["work_freq"] = 99
        self.assertEqual(ds.violations(hw), [])

    def test_converter_and_detector_bandwidth_caps(self):
        catalog = {
            "mzm": {"LT_MZM": {"cfgs": {"modulation_speed": 25, "testing_bit": 1}}},
            "dac": {"DAC_2": {"cfgs": {"prec": 8, "sample_rate": 18}}},
            "adc": {"ADC_SAR_1": {"cfgs": {"prec": 8, "sample_rate": 14}}},
            "photodetector": {"LT_PD": {"cfgs": {"bandwidth": 10}}},
        }
        ds = DesignSpace(device_catalog=catalog)
        hw = default_hardware()
        hw["arch"]["optical_core"]["work_freq"] = 20  # > DAC 18, ADC 14, PD 10
        constraints = {v["constraint"] for v in ds.violations(hw)}
        self.assertEqual(constraints, {
            "work_freq <= dac.sample_rate",
            "work_freq <= adc.sample_rate",
            "work_freq <= photodetector.bandwidth",
        })


class TestInstanceChoices(unittest.TestCase):
    """Instance-level device-type switches (laser_source / splitter_kind)."""

    def test_device_layer_params_exposed(self):
        ds = DesignSpace()
        self.assertEqual(ds.layer_of("laser_source"), "device")
        self.assertEqual(ds.layer_of("splitter_kind"), "device")
        self.assertEqual(ds.candidates("laser_source"), ["on_chip", "off_chip"])
        self.assertEqual(ds.candidates("splitter_kind"), ["laser_splitter", "mmi"])

    def test_set_get_roundtrip(self):
        ds = DesignSpace()
        hw = default_hardware()
        ds.set(hw, "laser_source", "off_chip")
        ds.set(hw, "splitter_kind", "mmi")
        self.assertEqual(ds.get(hw, "laser_source"), "off_chip")
        self.assertEqual(ds.get(hw, "splitter_kind"), "mmi")

    def test_normalize_clamps_enum(self):
        ds = DesignSpace()
        hw = default_hardware()
        hw["devices"]["core"]["laser_source"] = "bogus"
        hw["devices"]["core"]["splitter_kind"] = "bogus"
        fixed = ds.normalize(hw)
        self.assertEqual(fixed["devices"]["core"]["laser_source"], "on_chip")
        self.assertEqual(fixed["devices"]["core"]["splitter_kind"], "laser_splitter")


class TestAdaptiveRefine(unittest.TestCase):
    def test_refine_narrows_work_freq_around_center(self):
        ds = DesignSpace()
        hw = default_hardware()
        hw["arch"]["optical_core"]["work_freq"] = 10
        refined = ds.refine(hw, factor=5)
        self.assertEqual(
            refined.candidates("work_freq"),
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )

    def test_refine_keeps_geometric_grids(self):
        ds = DesignSpace()
        hw = default_hardware()
        refined = ds.refine(hw, factor=4)
        self.assertEqual(refined.candidates("tiles"), [1, 2, 4, 8])
        self.assertEqual(refined.candidates("cores_per_tile"), [1, 2, 4])

    def test_refine_is_centered_and_bounded(self):
        ds = DesignSpace()
        hw = default_hardware()
        hw["arch"]["optical_core"]["num_wavelength"] = 8
        refined = ds.refine(hw, factor=2)
        self.assertEqual(refined.candidates("num_wavelength"), [8, 9, 10])


class TestBuildDesignSpace(unittest.TestCase):
    def test_layered_spec_loads(self):
        ds = build_design_space(LAYERED_DESIGN_SPACE, device_catalog=CATALOG)
        self.assertEqual(ds.candidates("work_freq"), [5, 10, 15, 20, 25])

    def test_legacy_flat_spec_migrates(self):
        flat = {"arch": {"work_freq": [5, 10, 25], "num_wavelength": [8, 16]}}
        ds = build_design_space(flat, device_catalog=CATALOG)
        self.assertEqual(ds.candidates("work_freq"), [5, 10, 25])
        self.assertEqual(ds.candidates("num_wavelength"), [8, 16])
        # Unmentioned params keep their layered defaults.
        self.assertEqual(ds.candidates("hbm_bandwidth"), [512, 1024, 2048])

    def test_summary_exposes_layers_and_mode(self):
        ds = DesignSpace(device_catalog=CATALOG)
        rows = {r["parameter"]: r for r in ds.summary()}
        self.assertEqual(rows["glb1_cache_size"]["layer"], "system")
        self.assertEqual(rows["glb1_cache_size"]["mode"], "real")
        self.assertEqual(rows["work_freq"]["constraint"], "work_freq <= mzm.symbol_rate")


class TestFlatMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.space = {"arch": {
            "hbm_bandwidth": [512, 1024, 2048],
            "glb1_cache_size": [64, 256, 485, 1024, 4096],
        }}

    def test_modify_arch_parameter_memory_clamps(self):
        hw = default_hardware()
        out = modify_arch_parameter(
            hw, [{"parameter": "hbm_bandwidth", "value": 777}], self.space
        )
        self.assertEqual(out["arch"]["memory"]["HBM"]["bandwidth"], 1024)

    def test_step_arch_parameter_memory_advances(self):
        hw = default_hardware()
        stepped = step_arch_parameter(hw, "glb1_cache_size", 1, self.space)
        self.assertEqual(stepped["arch"]["memory"]["GLB1"]["cache_size"], 1024)

    def test_normalize_memory_clamps(self):
        hw = default_hardware()
        hw["arch"]["memory"]["HBM"]["bandwidth"] = 999
        out = normalize_hardware_to_design_space(hw, self.space)
        self.assertEqual(out["arch"]["memory"]["HBM"]["bandwidth"], 1024)

    def test_hardware_summary_includes_memory(self):
        from photon_tuner.core.search_space import hardware_summary

        summary = hardware_summary(default_hardware())
        self.assertIn("memory", summary["arch"])
        self.assertEqual(summary["arch"]["memory"]["HBM"]["bandwidth"], 1024)


class TestL4Topology(unittest.TestCase):
    def test_step_sharing_policy(self):
        hw = default_hardware()
        # adc_shared (default) -> adc_tia_shared; clamps at the end (no wrap).
        stepped = step_arch_parameter(hw, "sharing_policy", 1)
        self.assertEqual(stepped["arch"]["topology"]["sharing_policy"], "adc_tia_shared")
        stepped = step_arch_parameter(stepped, "sharing_policy", 1)
        self.assertEqual(stepped["arch"]["topology"]["sharing_policy"], "adc_tia_shared")

    def test_modify_laser_distribution(self):
        hw = default_hardware()
        out = modify_arch_parameter(
            hw, [{"parameter": "laser_distribution", "value": "H*W*N"}]
        )
        self.assertEqual(out["arch"]["topology"]["laser_distribution"], "H*W*N")
        # An illegal value is clamped back to the first legal candidate.
        out2 = modify_arch_parameter(
            hw, [{"parameter": "laser_distribution", "value": "bogus"}]
        )
        self.assertEqual(out2["arch"]["topology"]["laser_distribution"], "H*W")

    def test_normalize_clamps_enum(self):
        hw = default_hardware()
        hw["arch"]["topology"]["temporal_accum"] = "bogus"
        out = normalize_hardware_to_design_space(hw)
        self.assertEqual(out["arch"]["topology"]["temporal_accum"], "off")

    def test_summary_exposes_topology(self):
        from photon_tuner.core.search_space import hardware_summary

        summary = hardware_summary(default_hardware())
        self.assertEqual(summary["arch"]["sharing_policy"], "adc_shared")
        self.assertEqual(summary["arch"]["temporal_accum"], "default")
        self.assertEqual(summary["arch"]["laser_distribution"], "H*W")


if __name__ == "__main__":
    unittest.main()
