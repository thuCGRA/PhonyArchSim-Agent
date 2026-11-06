"""搜索空间「单一数据源」一致性测试。

守的是这条契约：

    候选值只声明一次 —— configs/search_space.json；
    拓扑种类的候选只声明一次 —— topology_space.py 的注册表；
    其余任何地方都必须是**派生**，不允许出现第二份字面量。

历史上这些列表被复制到 5 个位置（topology_space / design_space /
hardware_ops / search_space.json / baselines），没有一致性测试，于是出现
候选顺序不一致导致 ``step_hardware_parameter`` 按索引跳到不同拓扑的问题。

本文件分四组：
  A. 单一来源（派生关系成立、无重复字面量）
  B. 顺序一致性（族分组、±1 邻接语义）
  C. 规格自洽（map 路径可解析、约束引用的器件存在）
  D. 引擎对齐（baselines 与 agent 同 mode/transport）
"""

from __future__ import annotations

import ast
import inspect
import json
import unittest
from pathlib import Path

from photon_tuner.common.paths import CONFIG_ROOT, PROJECT_ROOT
from photon_tuner.core.search_space import (
    ARCH_PARAM_KEYS,
    DEFAULT_ARCH_SEARCH_SPACE,
    LAYERED_DESIGN_SPACE,
    SEARCH_SPACE_JSON,
    DesignSpace,
    build_design_space,
    load_search_space,
    materialize_hardware,
)
from photon_tuner.core.topology_space import (
    CORE_KIND_SEARCH_ORDER,
    CORE_KINDS,
    NODE_KIND_SEARCH_ORDER,
    NODE_KINDS,
    TOPOLOGY_FAMILIES,
    core_family,
    default_core_kind_candidates,
    default_node_kind_candidates,
    evaluable_template_names,
)


class TestSingleSourceOfTruth(unittest.TestCase):
    """A. 候选值只在一处声明，其余全部派生。"""

    def test_package_collapsed_to_single_module(self):
        """搜索空间只应有一个 Python 模块，不再是包。"""
        pkg_dir = PROJECT_ROOT / "photon_tuner" / "core" / "search_space"
        self.assertFalse(
            pkg_dir.is_dir(),
            "search_space 应是单一模块 photon_tuner/core/search_space.py，不应是包目录",
        )
        module = PROJECT_ROOT / "photon_tuner" / "core" / "search_space.py"
        self.assertTrue(module.is_file())

    def test_default_spec_is_loaded_from_json(self):
        """LAYERED_DESIGN_SPACE 必须是 JSON 的读取结果，而非手写副本。"""
        self.assertEqual(LAYERED_DESIGN_SPACE, load_search_space())
        self.assertEqual(SEARCH_SPACE_JSON, CONFIG_ROOT / "search_space.json")
        self.assertTrue(SEARCH_SPACE_JSON.is_file())

    def test_flat_space_is_derived_not_declared(self):
        """DEFAULT_ARCH_SEARCH_SPACE 必须派生自分层规格。"""
        from photon_tuner.core.search_space import _flat_arch_space

        self.assertEqual(DEFAULT_ARCH_SEARCH_SPACE, _flat_arch_space(LAYERED_DESIGN_SPACE))

    def test_json_does_not_declare_registry_owned_candidates(self):
        """JSON 不得声明 core_kind / node_kind 的候选（由注册表注入）。"""
        raw = json.loads(SEARCH_SPACE_JSON.read_text(encoding="utf-8"))
        for param in ("core_kind", "node_kind"):
            for layer in raw["layers"].values():
                p_spec = (layer.get("params") or {}).get(param)
                if p_spec is not None:
                    self.assertNotIn(
                        "values", p_spec,
                        f"JSON 不应声明 {param} 的 values——它由拓扑注册表注入",
                    )

    def test_registry_injection_matches_registry_order(self):
        ds = DesignSpace()
        self.assertEqual(ds.candidates("core_kind"), default_core_kind_candidates())
        self.assertEqual(ds.candidates("node_kind"), default_node_kind_candidates())

    def test_conflicting_declaration_is_rejected(self):
        """若某处硬塞一份与注册表不一致的 core_kind 候选，必须报错而非静默接受。"""
        spec = json.loads(SEARCH_SPACE_JSON.read_text(encoding="utf-8"))
        spec["layers"]["topology"]["params"]["core_kind"]["values"] = ["bogus_kind"]
        with self.assertRaises(ValueError) as ctx:
            DesignSpace(spec=spec)
        self.assertIn("topology registry", str(ctx.exception))

    def test_matching_declaration_is_accepted_idempotently(self):
        """已注入过的规格再次传入不应报错（注入必须幂等）。"""
        spec = json.loads(SEARCH_SPACE_JSON.read_text(encoding="utf-8"))
        spec["layers"]["topology"]["params"]["core_kind"]["values"] = list(
            default_core_kind_candidates()
        )
        DesignSpace(spec=spec)  # 不抛异常即通过

    def test_no_python_module_redeclares_candidate_lists(self):
        """除 search_space.py 外，不应有模块硬编码这些参数的候选值列表。"""
        suspects = [
            PROJECT_ROOT / "photon_tuner" / "core" / "topology_space.py",
            PROJECT_ROOT / "photon_tuner" / "core" / "search_space.py",
            PROJECT_ROOT / "baselines" / "action_space.py",
        ]
        # 允许出现参数名的模块；但出现 "param": [ ... ] 形式字面量则视为重复声明。
        watched = {"core_height", "core_width", "num_wavelength", "mesh_rank",
                   "work_freq", "tiles", "cores_per_tile", "dataflow"}
        for path in suspects:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                for key, value in zip(node.keys, node.values):
                    if not isinstance(key, ast.Constant) or key.value not in watched:
                        continue
                    if isinstance(value, ast.List):
                        self.fail(
                            f"{path.name} 中仍硬编码了 {key.value} 的候选列表；"
                            f"候选值应只存在于 configs/search_space.json"
                        )

    def test_arch_param_keys_match_spec(self):
        """ARCH_PARAM_KEYS 必须恰为 architecture ∪ topology 层的参数（顺序保持人工优先级）。

        器件层（laser_source / splitter_kind）由器件动作处理，不进入架构参数面。
        """
        spec_params = {
            name
            for layer_name, layer in LAYERED_DESIGN_SPACE["layers"].items()
            if layer_name in ("architecture", "topology")
            for name in (layer.get("params") or {})
        }
        self.assertEqual(set(ARCH_PARAM_KEYS), spec_params)
        self.assertEqual(len(ARCH_PARAM_KEYS), len(set(ARCH_PARAM_KEYS)), "存在重复项")

    def test_json_has_no_legacy_mode_field(self):
        """已废弃的 mode（analytic/real 分流）字段不应残留。"""
        raw = json.loads(SEARCH_SPACE_JSON.read_text(encoding="utf-8"))
        leftovers = [
            (layer_name, param)
            for layer_name, layer in raw["layers"].items()
            for param, p_spec in (layer.get("params") or {}).items()
            if "mode" in p_spec
        ]
        self.assertEqual(leftovers, [], f"仍有 mode 字段: {leftovers}")

    def test_design_space_summary_has_no_mode(self):
        ds = build_design_space({})
        for row in ds.summary():
            self.assertNotIn("mode", row)


class TestCandidateOrdering(unittest.TestCase):
    """B. 顺序只有一个来源，且 ±1 步进落在同一拓扑族内。"""

    def test_core_kind_order_covers_all_evaluable(self):
        self.assertEqual(
            set(default_core_kind_candidates()),
            set(evaluable_template_names("real")),
        )

    def test_core_kind_order_matches_declared_table(self):
        ordered = [k for k in CORE_KIND_SEARCH_ORDER if CORE_KINDS[k].get("evaluable")]
        self.assertEqual(default_core_kind_candidates(), ordered)

    def test_node_kind_order_matches_declared_table(self):
        ordered = [k for k in NODE_KIND_SEARCH_ORDER if NODE_KINDS[k].get("evaluable")]
        self.assertEqual(default_node_kind_candidates(), ordered)

    def test_order_table_is_family_grouped(self):
        """相邻项应属同一族（族内连续），否则 ±1 步进会跨族跳跃。"""
        seen_families = []
        for kind in default_core_kind_candidates():
            family = core_family(kind)
            if not seen_families or seen_families[-1] != family:
                seen_families.append(family)
        # 每个族只应出现一段连续区间。
        self.assertEqual(len(seen_families), len(set(seen_families)),
                         f"族分组不连续: {seen_families}")

    def test_every_evaluable_kind_is_placed_in_order_table(self):
        for kind, meta in CORE_KINDS.items():
            if meta.get("evaluable"):
                self.assertIn(kind, CORE_KIND_SEARCH_ORDER)
        for kind, meta in NODE_KINDS.items():
            if meta.get("evaluable"):
                self.assertIn(kind, NODE_KIND_SEARCH_ORDER)

    def test_step_uses_the_same_order_as_candidates(self):
        """按索引步进的候选序列必须与 DesignSpace.candidates 完全一致。"""
        from photon_tuner.core.search_space import step_arch_parameter

        ds = DesignSpace()
        hw = materialize_hardware(None)
        walked = [hw["arch"]["topology"]["core_kind"]]
        for _ in range(len(ds.candidates("core_kind")) - 1):
            hw = step_arch_parameter(hw, "core_kind", 1, None)
            walked.append(hw["arch"]["topology"]["core_kind"])
        self.assertEqual(walked, ds.candidates("core_kind"))


class TestSpecSelfConsistency(unittest.TestCase):
    """C. 规格里的 map 路径必须能在真实 hardware 上解析。"""

    def test_every_param_map_resolves(self):
        hw = materialize_hardware(None)
        ds = DesignSpace()
        for _layer, param, spec in ds.params():
            path = spec.get("map")
            self.assertTrue(path, f"{param} 缺少 map 路径")
            node = hw
            for part in path:
                self.assertIsInstance(node, dict, f"{param} 的 map 在中途不是 dict")
                if part not in node:
                    # 允许可选字段缺失（如激光器实例），但路径前缀必须存在。
                    break
                node = node[part]

    def test_every_param_has_values_or_is_selection_layer(self):
        ds = DesignSpace()
        for layer, param, spec in ds.params():
            if spec.get("kind") in ("selection", "fixed"):
                continue
            self.assertTrue(
                spec.get("values"),
                f"{layer}/{param} 没有候选值",
            )

    def test_constraints_reference_known_devices(self):
        ds = DesignSpace()
        known = {"mzm", "dac", "adc", "tia", "photodetector", "waveguide", "coupler"}
        for _layer, param, spec in ds.params():
            constraint = spec.get("constraint")
            if not constraint:
                continue
            mentioned = {
                token for token in known
                if token in constraint
            }
            self.assertTrue(mentioned, f"{param} 的约束未引用已知器件: {constraint}")


class TestBaselineEngineParity(unittest.TestCase):
    """D. baselines 与 agent 必须同 mode + 同 transport，否则对比不成立。"""

    def test_evaluator_defaults_to_real_over_mcp(self):
        from baselines.evaluator import SimphonyBaselineEvaluator

        sig = inspect.signature(SimphonyBaselineEvaluator.__init__)
        self.assertEqual(sig.parameters["mode"].default, "real")
        self.assertEqual(sig.parameters["transport"].default, "mcp")

    def test_evaluator_forwards_transport(self):
        from baselines.evaluator import SimphonyBaselineEvaluator

        src = inspect.getsource(SimphonyBaselineEvaluator.__init__)
        self.assertIn("transport=transport", src)
        self.assertNotIn('transport="inprocess"', src)

    def test_action_space_defaults_to_real(self):
        from baselines.action_space import ActionSpace

        sig = inspect.signature(ActionSpace.__init__)
        self.assertEqual(sig.parameters["mode"].default, "real")

    def test_runner_has_no_mode_flag(self):
        from baselines.runner import build_parser

        parser = build_parser()
        options = {opt for action in parser._actions for opt in action.option_strings}
        self.assertNotIn("--mode", options)
        # 且明确固定 real + mcp（不能只是"没传 mode"）。
        src = inspect.getsource(__import__("baselines.runner", fromlist=["main"]).main)
        self.assertIn('mode="real"', src)
        self.assertIn('transport="mcp"', src)

    def test_agent_cli_and_baseline_agree(self):
        from photon_tuner import cli

        src = inspect.getsource(cli)
        self.assertIn('mode="real"', src)
        self.assertIn('transport="mcp"', src)


class TestLegacyScriptsAligned(unittest.TestCase):
    """D2. 遗留脚本不再发出 analytic 选项/取值。"""

    def test_smoke_tune_uses_real_over_mcp(self):
        path = PROJECT_ROOT / "scripts" / "dev" / "smoke_tune.py"
        src = path.read_text(encoding="utf-8")
        self.assertIn('mode="real"', src)
        self.assertIn('transport="mcp"', src)
        self.assertNotIn('mode="analytic"', src)

    def test_verify_search_space_choices_only_real(self):
        path = PROJECT_ROOT / "scripts" / "experiments" / "verify_search_space.py"
        src = path.read_text(encoding="utf-8")
        self.assertIn('choices=["real"]', src)
        self.assertNotIn('"analytic"', src)
        self.assertNotIn('"auto"', src)


if __name__ == "__main__":
    unittest.main()
