"""参数 × 拓扑族有效性元数据的回归测试。

两层验证：

1. **声明层** —— ``PARAM_SCOPE`` / ``param_status`` / ``hidden_params`` 等 API 的
   语义正确（四类性质分开，未被混同）。
2. **事实层** —— 元数据声明的 ``effective`` / 非 ``effective`` 必须与三个网表构建器
   （``netlist.build_*_sub_arch``）的**实际行为**一致。这是防止漂移的关键：新增一种
   拓扑或改了构建器却忘了更新 ``PARAM_SCOPE`` 时，本测试会失败。
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from photon_tuner.core.search_space import (
    build_design_space,
    materialize_hardware,
    switch_topology_template,
)
from photon_tuner.core.topology_space import (
    CORE_KINDS,
    FAMILY_LABELS,
    PARAM_SCOPE,
    SCOPED_PARAMS,
    TOPOLOGY_FAMILIES,
    core_family,
    effective_params,
    hidden_params,
    implementable_gaps,
    is_param_effective,
    param_scope_table,
    param_status,
)
from photon_tuner.simulators.netlist import (
    build_mesh_sub_arch,
    build_sub_arch,
    build_weight_array_sub_arch,
    load_device_catalog,
)

ROOT = str(Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony")
CATALOG = load_device_catalog(f"{ROOT}/configs/devices")

MESH_KINDS = ("mzi_mesh_reck", "mzi_mesh_clements", "mzi_mesh_butterfly")
WEIGHT_KINDS = ("mrr_weight_bank", "pcm_crossbar", "wdm_link")
CROSSBAR_KINDS = ("dptc_crossbar",)


def _build(hardware, kind):
    if kind.startswith("mzi_mesh_"):
        return build_mesh_sub_arch(hardware, ROOT, CATALOG, kind)
    if kind in WEIGHT_KINDS:
        return build_weight_array_sub_arch(hardware, ROOT, CATALOG, kind)
    return build_sub_arch(hardware, ROOT, CATALOG)


def _fingerprint(sub_arch):
    """对决定仿真结果的字段取指纹（排除 name/version 等纯标识）。"""
    core = sub_arch.get("core", {}) or {}
    netlist = core.get("netlist", {}) or {}
    keep = {
        "tiles": sub_arch.get("tiles"),
        "cores_per_tile": sub_arch.get("cores_per_tile"),
        "dataflow": sub_arch.get("dataflow"),
        "height": core.get("height"),
        "width": core.get("width"),
        "num_wavelength": core.get("num_wavelength"),
        "work_freq": core.get("work_freq"),
        "precision": core.get("precision"),
        "matrix_height": core.get("matrix_height"),
        "matrix_width": core.get("matrix_width"),
        "scaling_rules": netlist.get("scaling_rules"),
        "laser_dist": netlist.get("laser_power_distribution_factor"),
        "temporal_accum": netlist.get("temporal_accum_factor"),
        "device_slots": sorted((core.get("devices") or {}).keys()),
    }
    blob = json.dumps(keep, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _mutator(param):
    """返回一个把 ``param`` 改成明显不同值的函数。"""
    table = {
        "core_height": (("arch", "optical_core"), "height", 32),
        "core_width": (("arch", "optical_core"), "width", 32),
        "num_wavelength": (("arch", "optical_core"), "num_wavelength", 32),
        "work_freq": (("arch", "optical_core"), "work_freq", 25),
        "dataflow": (("arch", "system"), "dataflow", "input_stationary"),
        "tiles": (("arch", "system"), "tiles", 8),
        "cores_per_tile": (("arch", "system"), "cores_per_tile", 4),
        "core_kind": (("arch", "topology"), "core_kind", "dptc_crossbar"),
        "node_kind": (("arch", "topology"), "node_kind", "dode"),
        "sharing_policy": (("arch", "topology"), "sharing_policy", "no_sharing"),
        "temporal_accum": (("arch", "topology"), "temporal_accum", "deep"),
        "laser_distribution": (("arch", "topology"), "laser_distribution", "H*W*N*C"),
        "mesh_rank": (("arch", "topology"), "mesh_rank", 32),
    }
    container, key, value = table[param]

    def apply(hw):
        node = hw
        for part in container:
            node = node.setdefault(part, {})
        node[key] = value

    return apply


class TestParamScopeDeclarations(unittest.TestCase):
    """声明层：四类性质必须分开，不能混同。"""

    def test_every_core_kind_has_a_family(self):
        for kind in CORE_KINDS:
            self.assertIn(kind, TOPOLOGY_FAMILIES, f"{kind} 缺少拓扑族归属")
        self.assertEqual(set(TOPOLOGY_FAMILIES), set(CORE_KINDS))

    def test_families_have_labels(self):
        for family in set(TOPOLOGY_FAMILIES.values()):
            self.assertIn(family, FAMILY_LABELS)

    def test_crossbar_all_architecture_params_effective(self):
        for kind in CROSSBAR_KINDS:
            self.assertEqual(hidden_params(kind), ["mesh_rank"])

    def test_mesh_hides_geometry_and_dataflow(self):
        for kind in MESH_KINDS:
            self.assertEqual(
                sorted(hidden_params(kind)),
                ["core_height", "core_width", "dataflow", "num_wavelength"],
            )

    def test_mesh_rank_is_the_mesh_lever(self):
        for kind in MESH_KINDS + WEIGHT_KINDS:
            self.assertIn("mesh_rank", effective_params(kind))
        for kind in CROSSBAR_KINDS:
            self.assertNotIn("mesh_rank", effective_params(kind))

    def test_dataflow_is_physically_fixed_not_backend_gap(self):
        """dataflow 在网格下是物理约束（fixed_by_design），不是待实现缺口。"""
        for kind in MESH_KINDS + WEIGHT_KINDS:
            st = param_status("dataflow", kind)
            self.assertEqual(st["status"], "fixed_by_design")
            self.assertIn("reason", st)
            self.assertNotIn("implementable", st)

    def test_mesh_num_wavelength_is_backend_gap_not_fixed(self):
        """网格的 num_wavelength 物理有意义、只是后端未接线——必须标为可补齐。"""
        for kind in MESH_KINDS:
            st = param_status("num_wavelength", kind)
            self.assertEqual(st["status"], "unsupported_by_backend")
            self.assertTrue(st.get("implementable"))

    def test_weight_array_num_wavelength_is_superseded(self):
        """权重阵列的 num_wavelength 与权重数物理恒等，属取代关系而非缺口。"""
        for kind in WEIGHT_KINDS:
            st = param_status("num_wavelength", kind)
            self.assertEqual(st["status"], "superseded_by")
            self.assertEqual(st["by"], "mesh_rank")

    def test_superseded_params_name_their_replacement(self):
        for param, spec in PARAM_SCOPE.items():
            for family, body in (spec.get("families") or {}).items():
                if body.get("status") == "superseded_by":
                    self.assertIn("by", body, f"{param}@{family} 缺少替代参数")
                    self.assertIn(
                        body["by"], SCOPED_PARAMS,
                        f"{param}@{family} 的替代参数 {body['by']} 不在受管参数内",
                    )

    def test_non_effective_statuses_carry_actionable_guidance(self):
        for param in SCOPED_PARAMS:
            for kind in CORE_KINDS:
                st = param_status(param, kind)
                if st["status"] == "effective":
                    continue
                self.assertIn("reason", st, f"{param}@{kind} 缺少 reason")
                self.assertIn("action", st, f"{param}@{kind} 缺少 action")

    def test_implementable_gaps_lists_mesh_num_wavelength(self):
        gaps = implementable_gaps()
        pairs = {(g["param"], g["family"]) for g in gaps}
        self.assertIn(("num_wavelength", "mesh_family"), pairs)
        for g in gaps:
            self.assertTrue(g["implementable"], f"{g} 被标为缺口却不可实现")

    def test_core_family_falls_back_safely(self):
        self.assertEqual(core_family(None), "crossbar_family")
        self.assertEqual(core_family("nonexistent_kind"), "crossbar_family")

    def test_scope_table_shape(self):
        table = param_scope_table("mzi_mesh_clements")
        self.assertEqual(table["family"], "mesh_family")
        self.assertEqual(set(table["params"]), set(SCOPED_PARAMS))


class TestParamScopeMatchesBuilders(unittest.TestCase):
    """事实层：元数据声明必须与真实构建器行为一致（防漂移护栏）。

    对每个 ``(参数, core_kind)`` 组合，改动 hardware 后重建 ``sub_arch``：
      * 声明 effective      → 指纹必须变化
      * 声明非 effective    → 指纹必须不变（除非该参数被 mesh_rank 遮蔽后仍有
                              回退路径，见 ``_shadowed_but_wired``）
    """

    def _probe(self, param, kind):
        base = materialize_hardware(None)
        hw0 = copy.deepcopy(base)
        hw0["arch"]["topology"]["core_kind"] = kind
        before = _fingerprint(_build(hw0, kind))

        hw1 = copy.deepcopy(base)
        hw1["arch"]["topology"]["core_kind"] = kind
        _mutator(param)(hw1)
        after = _fingerprint(_build(hw1, kind))
        return before != after

    def test_declared_effective_actually_changes_subarch(self):
        checked = 0
        for param in SCOPED_PARAMS:
            if param in ("core_kind", "node_kind"):
                continue  # 改 core_kind 会切换家族，无意义
            for kind in CORE_KINDS:
                if not is_param_effective(param, kind):
                    continue
                with self.subTest(param=param, kind=kind):
                    self.assertTrue(
                        self._probe(param, kind),
                        f"{param} 声明在 {kind} 下 effective，但改动后 sub_arch 未变",
                    )
                checked += 1
        self.assertGreater(checked, 0)

    def test_declared_ineffective_actually_noop(self):
        checked = 0
        for param in SCOPED_PARAMS:
            if param in ("core_kind", "node_kind"):
                continue
            for kind in CORE_KINDS:
                if is_param_effective(param, kind):
                    continue
                with self.subTest(param=param, kind=kind):
                    self.assertFalse(
                        self._probe(param, kind),
                        f"{param} 声明在 {kind} 下无效，但改动后 sub_arch 却变了——"
                        f" PARAM_SCOPE 已过时",
                    )
                checked += 1
        self.assertGreater(checked, 0)


class TestTopologyAwareSearchSpace(unittest.TestCase):
    """``DesignSpace.effective_candidates`` 收窄行为。"""

    def setUp(self):
        self.ds = build_design_space({})

    def _hw(self, kind):
        hw = materialize_hardware(None)
        hw["arch"]["topology"]["core_kind"] = kind
        return hw

    def test_crossbar_exposes_geometry_and_hides_mesh_rank(self):
        hw = self._hw("dptc_crossbar")
        self.assertEqual(self.ds.effective_candidates("core_height", hw), [8, 12, 16])
        self.assertEqual(self.ds.effective_candidates("mesh_rank", hw), [])

    def test_mesh_exposes_mesh_rank_and_hides_geometry(self):
        hw = self._hw("mzi_mesh_clements")
        self.assertEqual(self.ds.effective_candidates("mesh_rank", hw), [4, 8, 12, 16, 32])
        self.assertEqual(self.ds.effective_candidates("core_height", hw), [])
        self.assertEqual(self.ds.effective_candidates("dataflow", hw), [])

    def test_unscoped_params_unaffected(self):
        for kind in ("dptc_crossbar", "mzi_mesh_clements", "mrr_weight_bank"):
            hw = self._hw(kind)
            self.assertEqual(
                self.ds.effective_candidates("work_freq", hw),
                self.ds.candidates("work_freq"),
            )

    def test_missing_hardware_degrades_to_raw_candidates(self):
        self.assertEqual(
            self.ds.effective_candidates("core_height", None),
            self.ds.candidates("core_height"),
        )


class TestGatesRejectIneffectiveParams(unittest.TestCase):
    """闸门必须拒绝不生效的参数，并给出可操作理由。"""

    def setUp(self):
        from types import SimpleNamespace

        self.SimpleNamespace = SimpleNamespace
        self.ds = build_design_space({})

    def _output(self, recs, template=None):
        return self.SimpleNamespace(
            circuit_recommendations=recs, recommended_template=template
        )

    def _ctx(self, kind):
        hw = materialize_hardware(None)
        hw["arch"]["topology"]["core_kind"] = kind
        return {"design_space": self.ds, "hardware": hw, "mode": "real"}

    def test_architecture_gate_rejects_core_height_under_mesh(self):
        from photon_tuner.agents.evaluate import architecture_legality_check

        out = self._output([{"parameter": "core_height", "value": 16}])
        row = architecture_legality_check(out, self._ctx("mzi_mesh_clements"))
        self.assertEqual(row["status"], "fail")
        self.assertIn("mesh_rank", row["detail"])

    def test_architecture_gate_rejects_dataflow_under_mesh(self):
        from photon_tuner.agents.evaluate import architecture_legality_check

        out = self._output([{"parameter": "dataflow", "value": "input_stationary"}])
        row = architecture_legality_check(out, self._ctx("mzi_mesh_clements"))
        self.assertEqual(row["status"], "fail")
        self.assertIn("weight_stationary", row["detail"])

    def test_architecture_gate_accepts_effective_params(self):
        from photon_tuner.agents.evaluate import architecture_legality_check

        for kind, rec in (
            ("dptc_crossbar", {"parameter": "core_height", "value": 16}),
            ("mzi_mesh_clements", {"parameter": "work_freq", "value": 10}),
        ):
            with self.subTest(kind=kind):
                row = architecture_legality_check(self._output([rec]), self._ctx(kind))
                self.assertEqual(row["status"], "pass", row["detail"])

    def test_structure_gate_rejects_mesh_rank_under_crossbar(self):
        from photon_tuner.agents.evaluate import structure_legality_check

        out = self._output([{"parameter": "mesh_rank", "value": 16}])
        row = structure_legality_check("topology", out, self._ctx("dptc_crossbar"))
        self.assertEqual(row["status"], "fail")
        self.assertIn("core_height", row["detail"])

    def test_gate_uses_target_template_when_switching(self):
        """推荐切到网格 + 同时推 core_height：按目标结构判定为无效。"""
        from photon_tuner.agents.evaluate import architecture_legality_check

        out = self._output(
            [{"parameter": "core_height", "value": 16}], template="mzi_mesh_clements"
        )
        row = architecture_legality_check(out, self._ctx("dptc_crossbar"))
        self.assertEqual(row["status"], "fail")


if __name__ == "__main__":
    unittest.main()


class TestPhysicallyForcedValues(unittest.TestCase):
    """拓扑物理强制值：设计点声称的值必须与装配器实际使用的一致。

    这类"看似合法、实际被网表忽略"的组合是静默的：不加检查时，一个设计点可以
    声明 ``dataflow=output_stationary`` / ``num_wavelength=12``，而网表按
    ``weight_stationary`` / ``1`` 求值，agent 于是基于一个从未被仿真过的配置做决策。
    """

    def test_constant_forced_values_per_family(self):
        from photon_tuner.core.topology_space import forced_params

        # crossbar 家族对这三者没有物理约束。
        self.assertEqual(forced_params("dptc_crossbar"), {})

        mesh = forced_params("mzi_mesh_clements")
        self.assertEqual(mesh["core_width"], 1)
        self.assertEqual(mesh["num_wavelength"], 1)
        self.assertEqual(mesh["dataflow"], "weight_stationary")

        for kind in ("mrr_weight_bank", "pcm_crossbar", "wdm_link"):
            with self.subTest(kind=kind):
                wa = forced_params(kind)
                self.assertEqual(wa["core_width"], 1)
                self.assertEqual(wa["dataflow"], "weight_stationary")
                # 波长数不是常量，而由 mesh_rank 派生，故不在此表。
                self.assertNotIn("num_wavelength", wa)

    def test_derived_forced_value_follows_mesh_rank(self):
        from photon_tuner.core.topology_space import effective_forced_params

        hw = switch_topology_template(materialize_hardware(None), "mrr_weight_bank")
        rank = int(hw["arch"]["topology"]["mesh_rank"])
        self.assertEqual(effective_forced_params(hw)["num_wavelength"], rank)

    def test_switch_topology_yields_physically_valid_point(self):
        """仅切换拓扑就必须得到合法设计点——不能留下 crossbar 的取值。"""
        from photon_tuner.core.topology_space import forced_value_violations

        for kind in CORE_KINDS:
            with self.subTest(kind=kind):
                hw = switch_topology_template(materialize_hardware(None), kind)
                self.assertEqual(forced_value_violations(hw), [])

    def test_illegal_combinations_are_detected(self):
        """物理上不成立的组合必须被显式检出，而不是静默生效。"""
        from photon_tuner.core.topology_space import forced_value_violations

        cases = (
            ("mzi_mesh_clements", "core", "num_wavelength", 12),
            ("mzi_mesh_clements", "core", "width", 12),
            ("mzi_mesh_clements", "system", "dataflow", "output_stationary"),
            ("mrr_weight_bank", "system", "dataflow", "input_stationary"),
            ("wdm_link", "core", "width", 8),
        )
        # 违规里用的是设计空间参数名，而不是 hardware 里的原始字段名。
        param_of = {"num_wavelength": "num_wavelength", "width": "core_width",
                    "dataflow": "dataflow"}
        for kind, where, field, value in cases:
            with self.subTest(kind=kind, field=field, value=value):
                hw = switch_topology_template(materialize_hardware(None), kind)
                container = hw["arch"]["optical_core" if where == "core" else "system"]
                container[field] = value
                violations = forced_value_violations(hw)
                self.assertEqual(len(violations), 1, violations)
                self.assertEqual(violations[0]["param"], param_of[field])
                self.assertEqual(violations[0]["current"], value)
                self.assertIn("physically impossible", violations[0]["message"])

    def test_crossbar_is_not_constrained(self):
        """同样的取值在 crossbar 下合法——约束必须依赖拓扑，而非全局。"""
        from photon_tuner.core.topology_space import forced_value_violations

        hw = switch_topology_template(materialize_hardware(None), "dptc_crossbar")
        self.assertEqual(hw["arch"]["optical_core"]["num_wavelength"], 12)
        self.assertEqual(hw["arch"]["system"]["dataflow"], "output_stationary")
        self.assertEqual(forced_value_violations(hw), [])

    def test_normalize_rewrites_to_effective_values(self):
        from photon_tuner.core.topology_space import forced_value_violations

        ds = build_design_space()
        hw = switch_topology_template(materialize_hardware(None), "mzi_mesh_clements")
        hw["arch"]["optical_core"]["num_wavelength"] = 12
        hw["arch"]["optical_core"]["width"] = 12
        hw["arch"]["system"]["dataflow"] = "output_stationary"
        self.assertEqual(len(forced_value_violations(hw)), 3)

        fixed = ds.normalize(hw)
        self.assertEqual(fixed["arch"]["optical_core"]["num_wavelength"], 1)
        self.assertEqual(fixed["arch"]["optical_core"]["width"], 1)
        self.assertEqual(fixed["arch"]["system"]["dataflow"], "weight_stationary")
        self.assertEqual(forced_value_violations(fixed), [])

    def test_gate_rejects_nonforced_value(self):
        from photon_tuner.agents.evaluate import _forced_value_fail

        self.assertIsNone(_forced_value_fail("num_wavelength", 1, "mzi_mesh_clements"))
        self.assertIsNotNone(_forced_value_fail("num_wavelength", 12, "mzi_mesh_clements"))
        self.assertIsNotNone(_forced_value_fail("dataflow", "output_stationary", "mzi_mesh_clements"))
        # crossbar 下同样的值合法。
        self.assertIsNone(_forced_value_fail("num_wavelength", 12, "dptc_crossbar"))
        self.assertIsNone(_forced_value_fail("dataflow", "output_stationary", "dptc_crossbar"))

    def test_gate_uses_hardware_for_derived_values(self):
        """权重阵列下 num_wavelength 必须等于 mesh_rank（派生强制）。"""
        from photon_tuner.agents.evaluate import _forced_value_fail

        hw = switch_topology_template(materialize_hardware(None), "mrr_weight_bank")
        rank = int(hw["arch"]["topology"]["mesh_rank"])
        self.assertIsNone(_forced_value_fail("num_wavelength", rank, "mrr_weight_bank", hw))
        self.assertIsNotNone(_forced_value_fail("num_wavelength", rank + 4, "mrr_weight_bank", hw))

    def test_physical_violations_includes_forced_scope(self):
        from photon_tuner.core.topology_space import physical_violations

        hw = switch_topology_template(materialize_hardware(None), "mzi_mesh_clements")
        hw["arch"]["optical_core"]["num_wavelength"] = 12
        scopes = {v["scope"] for v in physical_violations(hw)}
        self.assertIn("forced_value", scopes)


class TestUnknownCoreKindIsRejected(unittest.TestCase):
    """未登记的 core_kind 必须报错，而不是静默退回 crossbar 网表。

    旧行为的后果是：agent 以为自己在评测一个新拓扑，拿到的却是 DPTC crossbar 的
    数值——错误的证据比缺失的证据更糟。
    """

    def test_analytic_raises_on_unknown_core_kind(self):
        from photon_tuner.simulators import analytic

        hw = materialize_hardware(None)
        hw["arch"]["topology"]["core_kind"] = "not_a_real_topology"
        with self.assertRaises(ValueError) as ctx:
            analytic.evaluate(hw, ROOT)
        self.assertIn("unknown core_kind", str(ctx.exception))

    def test_sub_arch_name_tracks_core_kind(self):
        """子架构名与报告键都取拓扑名，而不是 YAML 的静态查表名（"LT"）。"""
        from photon_tuner.simulators.analytic import evaluate
        from photon_tuner.simulators.netlist import sub_arch_key

        for kind in CORE_KINDS:
            with self.subTest(kind=kind):
                hw = switch_topology_template(materialize_hardware(None), kind)
                self.assertEqual(sub_arch_key(hw), kind)
                report = evaluate(hw, ROOT)
                self.assertEqual(list(report["workload"]["partition_cycles"]), [kind])
                self.assertEqual(list(report["energy"]["total"]), [kind])
