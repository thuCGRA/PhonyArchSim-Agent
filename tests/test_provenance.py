"""出处审计：每个器件的 cfgs 数值必须可追溯到论文（或被标记）。"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICELIB = ROOT / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"

from photon_tuner.core.provenance import (  # noqa: E402
    BASELINE_SOURCE,
    audit_device_provenance,
    classify_variant,
    load_full_catalog,
    resolve_variant,
)


class TestProvenanceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit_device_provenance(str(DEVICELIB))

    def test_internally_consistent_counts(self):
        r = self.report
        self.assertEqual(
            r["total"],
            len(r["traceable"]) + len(r["baseline"]) + len(r["placeholder"]),
        )

    def test_round1_devices_are_traceable(self):
        # 五个获准的 R1 新增器件必须各自带有论文引用。
        for key in (
            "adc/ADC_SAR_90GS",
            "dac/DAC_3",
            "dac/DAC_4",
            "phase_shifter/IHO_PS",
            "photodetector/Sandia_GeWG_PD",
        ):
            self.assertIn(key, self.report["traceable"], key)

    def test_vendored_baseline_is_flagged(self):
        # LT 基线器件不携带单独引用 → 以内置 SimPhony 批量来源报告，绝不臆造逐器件的 DOI。
        for key in ("mzm/LT_MZM", "photodetector/LT_PD", "dac/DAC_1"):
            self.assertIn(key, self.report["baseline"], key)
        rec = self.report["variants"]["mzm/LT_MZM"]
        self.assertEqual(rec["class"], "baseline")
        self.assertEqual(rec["ref"], BASELINE_SOURCE["ref"])
        self.assertIn("SimPhony", rec["source"])

    def test_idealized_placeholders_are_separate(self):
        # 仿真器结构默认值（P_pi=0 等）不是论文器件。
        self.assertIn("phase_shifter/Customized_PS", self.report["placeholder"])
        rec = self.report["variants"]["phase_shifter/Customized_PS"]
        self.assertEqual(rec["class"], "placeholder")

    def test_every_baseline_device_has_explicit_note(self):
        # 准则 5「未报告字段明确省略或标注」：每个无 ref 的可搜索器件都必须带有逐器件
        # 的诚实出处说明（绝不静默无标注）。
        from photon_tuner.core.provenance import BASELINE_PROVENANCE_NOTES

        missing = sorted(set(self.report["baseline"]) - set(BASELINE_PROVENANCE_NOTES))
        self.assertEqual(missing, [], f"baseline devices without a note: {missing}")
        # 且说明必须非平凡（不能是笼统的兜底值）。
        for key in self.report["baseline"]:
            rec = self.report["variants"][key]
            self.assertNotEqual(rec["note"], "", key)
            self.assertNotEqual(rec["note"], BASELINE_SOURCE["note"], key)

    def test_no_traceable_device_is_misclassified(self):
        for dt, variants in load_full_catalog(str(DEVICELIB)).items():
            for name, body in variants.items():
                if str((body or {}).get("ref", "") or "").strip():
                    self.assertEqual(classify_variant(dt, name, body), "traceable")

    def test_searchable_and_placeholder_partition(self):
        from photon_tuner.core.search_space import EXCLUDED_SEARCH_VARIANTS

        for dt, variants in load_full_catalog(str(DEVICELIB)).items():
            excluded = EXCLUDED_SEARCH_VARIANTS.get(dt, frozenset())
            for name, body in variants.items():
                cls = classify_variant(dt, name, body)
                if name in excluded:
                    self.assertEqual(cls, "placeholder", f"{dt}/{name}")
                else:
                    self.assertIn(cls, ("traceable", "baseline"), f"{dt}/{name}")


class TestResolveVariant(unittest.TestCase):
    def test_traceable_resolution(self):
        body = {"ref": "https://doi.org/10.1002/lpor.202201032"}
        rec = resolve_variant("phase_shifter", "IHO_PS", body)
        self.assertEqual(rec["class"], "traceable")
        self.assertEqual(rec["ref"], "https://doi.org/10.1002/lpor.202201032")

    def test_baseline_resolution_uses_batch_source(self):
        rec = resolve_variant("mzm", "LT_MZM", {"ref": ""})
        self.assertEqual(rec["class"], "baseline")
        self.assertEqual(rec["ref"], "https://arxiv.org/abs/2411.13715")


class TestInspectDeviceConfigProvenance(unittest.TestCase):
    def test_inspect_returns_ref_platform_and_provenance(self):
        from photon_tuner.toolbox.common_tool.context import ToolContext
        from photon_tuner.toolbox.device_tools import DeviceToolbox

        box = DeviceToolbox(devicelib_root=str(DEVICELIB))
        ctx = ToolContext(output_dir=Path("."))

        p = box.run("inspect_device_config", ctx, {
            "device_type": "phase_shifter", "device_name": "IHO_PS",
        }).payload
        self.assertEqual(p["ref"], "https://doi.org/10.1002/lpor.202201032")
        self.assertEqual(p["platform"], "strip-waveguide-550nm")
        self.assertEqual(p["provenance"], "traceable")

        b = box.run("inspect_device_config", ctx, {
            "device_type": "mzm", "device_name": "LT_MZM",
        }).payload
        self.assertEqual(b["provenance"], "baseline")
        self.assertIn("SimPhony", b["source"])


if __name__ == "__main__":
    unittest.main()
