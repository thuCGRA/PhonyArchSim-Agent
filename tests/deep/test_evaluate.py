"""各阶段证据评估门控的测试（确定性各行 + 决策）。"""

import json
import unittest

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from photon_tuner.agents.evaluate import (
    STRUCTURE_LEGALITY_ROLES,
    decide,
    deterministic_checks,
    semantic_fails,
    structure_legality_check,
    topology_binding_check,
)
from photon_tuner.agents.outputs import ArchitectureOutput, EvaluateOutput, SolutionOutput


class _Out(BaseModel):
    summary: str


def _evidence_msgs():
    return [
        AIMessage(content="", tool_calls=[{"name": "t1", "args": {}, "id": "c1"}]),
        ToolMessage(content="{}", tool_call_id="c1", name="t1"),
    ]


class TestEvaluateGate(unittest.TestCase):
    def test_deterministic_rows_pass_with_evidence(self):
        checks = deterministic_checks("model", _Out(summary="x"), messages=_evidence_msgs())
        self.assertEqual([c["row"] for c in checks], [1, 4])
        self.assertTrue(all(c["status"] == "pass" for c in checks))

    def test_solution_adds_edit_consistency_row(self):
        checks = deterministic_checks(
            "solution", _Out(summary="x"), messages=_evidence_msgs(),
            applied_edits=["step_hardware_parameter"], failed_edits=[],
            declared_edits=["step_hardware_parameter"],
        )
        self.assertIn(5, [c["row"] for c in checks])
        self.assertTrue(all(c["status"] == "pass" for c in checks))

    def test_accept_when_all_pass(self):
        checks = deterministic_checks(
            "solution", _Out(summary="x"), messages=_evidence_msgs(),
            applied_edits=["a"], failed_edits=[], declared_edits=["a"],
        )
        checks.append(structure_legality_check("solution", _Out(summary="x")))
        self.assertEqual(decide(checks)["decision"], "accept")

    def test_edit_mismatch_triggers_reexecute(self):
        checks = deterministic_checks(
            "solution", _Out(summary="x"), messages=_evidence_msgs(),
            applied_edits=[], failed_edits=[],
            declared_edits=["step_hardware_parameter"],
        )
        self.assertEqual(decide(checks)["decision"], "re-execute")

    def test_missing_evidence_triggers_request_info(self):
        checks = deterministic_checks("model", _Out(summary="x"), messages=None)
        self.assertEqual(decide(checks)["decision"], "request-info")

    def test_structure_legality_pass_when_nothing_supplied(self):
        self.assertEqual(structure_legality_check("topology", _Out(summary="x"))["status"], "pass")
        self.assertIn("topology", STRUCTURE_LEGALITY_ROLES)
        self.assertNotIn("model", STRUCTURE_LEGALITY_ROLES)

    def test_structure_legality_rejects_unknown_template(self):
        from photon_tuner.agents.outputs import TopologyOutput

        out = TopologyOutput(
            loss_budget_diagnosis="x", dominant_loss_contributors=[],
            power_budget_margin="y", recommended_template="bogus_template",
        )
        # 未知模板名（不在 CORE_KINDS 里）被确定性拒绝。
        check = structure_legality_check("topology", out, context={"mode": "real"})
        self.assertEqual(check["status"], "fail")
        self.assertIn("not a legal template", check["detail"])

    def test_structure_legality_accepts_default_hardware(self):
        from photon_tuner.core.search_space import default_hardware

        check = structure_legality_check("solution", _Out(summary="x"),
                                          context={"hardware": default_hardware()})
        self.assertEqual(check["status"], "pass")

    def test_structure_legality_rejects_bad_core_kind(self):
        from photon_tuner.core.search_space import default_hardware

        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "bogus_kind"
        check = structure_legality_check("solution", _Out(summary="x"), context={"hardware": hw})
        self.assertEqual(check["status"], "fail")
        self.assertIn("core_kind", check["detail"])

    def test_structure_legality_flags_numeric_coupling_violation(self):
        # 协调器提案把 work_freq 提到 MZM 符号速率上限之上时，必须使第 5.5 行失败
        # （数值耦合在此强制约束）。
        from photon_tuner.core.search_space import DesignSpace, default_hardware

        strict = DesignSpace(device_catalog={
            "mzm": {"LT_MZM": {"cfgs": {"modulation_speed": 10, "testing_bit": 1}, "type": "MZM"}},
            "dac": {"DAC_2": {"cfgs": {"prec": 8}, "type": "DAC"}},
            "adc": {"ADC_SAR_1": {"cfgs": {"prec": 8}, "type": "ADC"}},
        })
        hw = default_hardware()
        hw["arch"]["optical_core"]["work_freq"] = 25  # > 10 GHz 上限
        check = structure_legality_check(
            "solution", _Out(summary="x"), context={"hardware": hw, "design_space": strict}
        )
        self.assertEqual(check["status"], "fail")
        self.assertIn("work_freq", check["detail"])

    def test_json_safe_serializes_design_space_context(self):
        import json

        from photon_tuner.core.search_space import DesignSpace
        from photon_tuner.runtime.orchestrator import _json_safe

        ds = DesignSpace(device_catalog={
            "mzm": {"LT_MZM": {"cfgs": {"modulation_speed": 25}}, "type": "MZM"},
        })
        context = {
            "hardware": {"arch": {"topology": {"core_kind": "dptc_crossbar"}}},
            "design_space": ds,
        }
        safe = _json_safe(context)
        # DesignSpace 对象必须被概括（行列表）而非 repr()，且结果须能通过
        # json.dumps 往返而不抛异常。
        self.assertIsInstance(safe["design_space"], list)
        self.assertTrue(safe["design_space"])
        self.assertEqual(safe["design_space"][0]["parameter"], "tiles")
        json.dumps(safe)  # 不得抛出 TypeError

    def test_semantic_fail_flattens_and_triggers_reexecute(self):
        verdict = EvaluateOutput(
            stage="solution",
            content_completeness="pass",
            evidence_completeness="pass",
            reasoning_correctness="fail - wrong root cause",
            evidence_applicability="pass",
            feedback=["re-derive the root cause from the energy breakdown"],
        )
        fails = semantic_fails(verdict)
        self.assertEqual(len(fails), 1)
        self.assertEqual(fails[0]["row"], 6)
        decision = decide([], verdict)
        self.assertEqual(decision["decision"], "re-execute")
        self.assertTrue(any("re-derive the root cause" in r for r in decision["reasons"]))
        # 确定性失败短路评判 -> 无结论 -> 无语义失败。
        self.assertEqual(semantic_fails(None), [])


class TestMemoryLatencyValue(unittest.TestCase):
    def test_analytic_mirror_keys(self):
        from photon_tuner.toolbox.common_tool._report_utils import _memory_latency_value
        self.assertEqual(_memory_latency_value({"input_latency_s": 1.5e-6, "total_size": 4}), 1.5e-6)
        self.assertEqual(_memory_latency_value({"total": 2.0}), 2.0)

    def test_real_simulator_per_level_keys_sum(self):
        # 真实 ONN 仿真器按层级（GLB2/GLB1/RF、输入/输出）报告存储延迟；
        # 这些必须求和，而不是当作零。
        from photon_tuner.toolbox.common_tool._report_utils import _memory_latency_value
        per_level = {
            "GLB2_input_latency": 2.1600730368e-05, "GLB2_output_latency": 3.624780096e-06,
            "GLB1_input_latency": 7.101609984e-06, "GLB1_output_latency": 3.94533888e-07,
            "RF_input_latency": 1.4795020799999998e-07, "RF_output_latency": 3.94533888e-07,
        }
        self.assertAlmostEqual(_memory_latency_value(per_level), 3.3264138432e-05, places=12)
        self.assertEqual(_memory_latency_value({}), 0.0)
        self.assertEqual(_memory_latency_value(None), 0.0)

    def test_memory_energy_flattens_per_level(self):
        # 真实 ONN 仿真器按层级（HBM/GLB2/...）在子架构（"LT"）下嵌套报告存储能量；
        # 将其展平并校验总和。
        from photon_tuner.toolbox.common_tool._report_utils import _flatten_memory_energy
        nested = {"LT": {"HBM": 541090368.0, "GLB2": 70938455.0, "RF": 106351701.0}}
        flat = _flatten_memory_energy(nested)
        self.assertEqual(set(flat), {"HBM", "GLB2", "RF"})
        total = sum(float(v) for v in flat.values() if isinstance(v, (int, float)))
        self.assertEqual(total, 718380524.0)


class TestL4RecommendationLegality(unittest.TestCase):
    def _ctx(self):
        from photon_tuner.core.search_space import build_design_space
        from photon_tuner.common.io import read_json
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        return {"design_space": build_design_space(read_json(root / "configs" / "search_space.json"))}

    def test_invented_l4_values_fail_legality(self):
        from photon_tuner.agents.evaluate import structure_legality_check

        out = _TopoOut(circuit_recommendations=[
            {"parameter": "sharing_policy", "value": "full_shared", "rationale": "x"},
            {"parameter": "laser_distribution", "value": "H", "rationale": "x"},
        ])
        row = structure_legality_check("topology", out, self._ctx())
        self.assertEqual(row["status"], "fail")
        self.assertIn("full_shared", row["detail"])
        self.assertIn("H", row["detail"])

    def test_legal_l4_values_pass_legality(self):
        from photon_tuner.agents.evaluate import structure_legality_check

        out = _TopoOut(circuit_recommendations=[
            {"parameter": "sharing_policy", "value": "adc_tia_shared", "rationale": "x"},
            {"parameter": "temporal_accum", "value": "deep", "rationale": "x"},
        ])
        row = structure_legality_check("topology", out, self._ctx())
        self.assertEqual(row["status"], "pass")


class TestInsertionLossDetailNormalization(unittest.TestCase):
    """真实模式与解析模式的 IL 报告必须以同一种形态到达智能体。"""

    def test_real_component_loss_map_expands_to_detail_list(self):
        from photon_tuner.toolbox.device_tools.metrics import _normalize_il_detail

        real = {"LT": {
            "core_path": ["i_laser", "i_mzi_0"],
            "core_insertion_loss": 28.56,
            "core_component_loss": {"i_laser": 2.0, "i_mzi": 26.5, "i_dac": 0.06},
            "node_component_loss": {"i0": 0.5, "i1": 1.16},
            "node_insertion_loss": 1.16,
        }}
        out = _normalize_il_detail(real)["LT"]
        detail = out["core_path_detail"]
        # 按贡献降序排列，且总和回算到报告的总损耗。
        self.assertEqual([e["instance"] for e in detail], ["i_mzi", "i_laser", "i_dac"])
        self.assertAlmostEqual(
            sum(e["insertion_loss_dB"] for e in detail),
            out["core_insertion_loss"],
            places=6,
        )
        self.assertTrue(out["node_path_detail"])

    def test_real_breakdown_resolves_device_names_from_by_device(self):
        from photon_tuner.toolbox.device_tools.metrics import _normalize_il_detail

        report = {
            "energy": {"by_device": {"LT": {
                "core_off_chip_laser_0_i_laser-LT_Laser": {},
                "core_photodetector_0_i_pd-LT_PD": {},
                # 节点与核心复用实例名；查找不能发生冲突。
                "core_mzm_0_i0-LT_MZM": {},
                "node_coupler_0_i0-LT_DC_2x2": {},
            }}},
            "insertion_loss": {"LT": {
                "node_insertion_loss": 1.66,
                "core_component_loss": {"i_laser": 2.0, "i_mzi": 26.56},
                "node_component_loss": {"i0": 1.66},
            }},
        }
        out = _normalize_il_detail(report["insertion_loss"], report)["LT"]
        core = {e["instance"]: e for e in out["core_path_detail"]}
        self.assertEqual(core["i_laser"]["device_name"], "LT_Laser")
        self.assertEqual(core["i_laser"]["device_type"], "off_chip_laser")
        # 损耗是节点 IL 整数倍的无法解析实例，即被折叠的节点子电路。
        self.assertEqual(core["i_mzi"]["device_name"], "node_subcircuit")
        node = {e["instance"]: e for e in out["node_path_detail"]}
        self.assertEqual(node["i0"]["device_name"], "LT_DC_2x2")

    def test_analytic_detail_list_passes_through_unchanged(self):
        from photon_tuner.toolbox.device_tools.metrics import _normalize_il_detail

        analytic = {"LT": {
            "core_path_detail": [{"instance": "i4", "insertion_loss_dB": 2.0}],
        }}
        out = _normalize_il_detail(analytic)["LT"]
        self.assertEqual(out["core_path_detail"], [{"instance": "i4", "insertion_loss_dB": 2.0}])


class TestEvidenceCompletenessCheck(unittest.TestCase):
    """第 5.6 行——针对「证据不可得」说法的确定性兜底检查。"""

    def _out(self, **kw):
        from photon_tuner.agents.outputs import TopologyOutput

        base = {
            "loss_budget_diagnosis": "The MZI chain dominates.",
            "dominant_loss_contributors": [
                {"instance": "i_mzi", "device_type": "node",
                 "device_name": "node_subcircuit", "insertion_loss_dB": 26.56},
            ],
            "power_budget_margin": "laser 23.5 dBm - 28.56 dB loss vs -25 dBm => +19.94 dB",
            "power_budget": {
                "laser_output_dbm": 23.5, "path_loss_db": 28.56,
                "pd_sensitivity_dbm": -25.0, "link_margin_db": 19.94,
            },
        }
        base.update(kw)
        return TopologyOutput(**base)

    def test_complete_evidence_passes(self):
        from photon_tuner.agents.evaluate import evidence_completeness_check

        row = evidence_completeness_check("topology", self._out())
        self.assertEqual(row["status"], "pass")

    def test_placeholder_loss_contributors_fail(self):
        from photon_tuner.agents.evaluate import evidence_completeness_check

        row = evidence_completeness_check(
            "topology", self._out(dominant_loss_contributors=[])
        )
        self.assertEqual(row["status"], "fail")
        self.assertIn("summarize_insertion_loss_detail", row["detail"])

    def test_not_exposed_claim_fails(self):
        from photon_tuner.agents.evaluate import evidence_completeness_check

        row = evidence_completeness_check(
            "topology", self._out(power_budget_margin="absolute dBm margin cannot be stated")
        )
        self.assertEqual(row["status"], "fail")
        self.assertIn("power_budget_margin", row["detail"])

    def test_missing_power_budget_keys_fail(self):
        from photon_tuner.agents.evaluate import evidence_completeness_check

        row = evidence_completeness_check("topology", self._out(power_budget={}))
        self.assertEqual(row["status"], "fail")
        self.assertIn("laser_output_dbm", row["detail"])

    def test_inconsistent_margin_fails(self):
        from photon_tuner.agents.evaluate import evidence_completeness_check

        row = evidence_completeness_check("topology", self._out(power_budget={
            "laser_output_dbm": 23.5, "path_loss_db": 28.56,
            "pd_sensitivity_dbm": -25.0, "link_margin_db": -5.06,  # 符号错误
        }))
        self.assertEqual(row["status"], "fail")
        self.assertIn("inconsistent", row["detail"])

    def test_other_roles_are_not_gated(self):
        from photon_tuner.agents.evaluate import evidence_completeness_check

        row = evidence_completeness_check("architecture", object())
        self.assertEqual(row["status"], "pass")

    def test_row_5_6_reason_reaches_rework_feedback(self):
        # 门控的失败原因是返工看到的 gate_feedback，因此必须是能指明缺失证据的
        # 可操作文本。
        from photon_tuner.agents.evaluate import decide, evidence_completeness_check
        from photon_tuner.agents.outputs import TopologyOutput

        out = TopologyOutput(
            loss_budget_diagnosis="laser dominates",
            dominant_loss_contributors=[
                {"instance": "i_mzi", "device_type": "node",
                 "device_name": "node_subcircuit", "insertion_loss_dB": 26.56}],
            power_budget_margin="cannot be stated",
            power_budget={},
            summary="x",
        )
        checks = [evidence_completeness_check("topology", out)]
        decision = decide(checks)
        self.assertEqual(decision["decision"], "re-execute")
        joined = " ".join(decision["reasons"])
        self.assertIn("power_budget", joined)
        self.assertIn("inspect_device_config", joined)
        # 反馈必须引用违规措辞，使返工可见。
        self.assertIn("cannot be stated", joined)

    def test_row_5_6_is_hard_in_decide(self):
        from photon_tuner.agents.evaluate import decide

        checks = [{"row": 5.6, "name": "evidence_completeness", "status": "fail",
                   "detail": "power_budget is missing [...]"}]
        self.assertEqual(decide(checks)["decision"], "re-execute")


class TestToolErrorsAreRecoverable(unittest.TestCase):
    """坏的工具参数绝不能中止运行。

    本测试验证域错误会被当作工具输出返回，而不是从工具节点抛出并杀死运行中的真实运行。
    """

    def test_domain_error_is_returned_as_tool_output(self):
        from photon_tuner.common.tool_types import ToolSpec
        from photon_tuner.toolbox.bindings import _wrap

        class _State:
            context = object()

        def _boom(toolbox, context, arguments):
            raise ValueError("Device '' not found for type 'mzm'")

        spec = ToolSpec("inspect_device_config", "x", {
            "type": "object",
            "properties": {"device_type": {"type": "string"}, "device_name": {"type": "string"}},
        })
        tool = _wrap(None, _State(), spec, _boom)
        out = tool.invoke({"device_type": "mzm", "device_name": ""})
        self.assertIsInstance(out, dict)
        self.assertIn("error", out)
        self.assertIn("mzm", out["error"])

    def test_empty_device_name_lists_candidates_instead_of_raising(self):
        from pathlib import Path
        from photon_tuner.toolbox.device_tools import DeviceToolbox
        from photon_tuner.toolbox.device_tools.catalog import inspect_device_config
        from photon_tuner.toolbox.common_tool.context import ToolContext
        from photon_tuner.core.search_space import default_hardware

        root = Path(__file__).resolve().parents[2]
        tb = DeviceToolbox(str(root / "photon_tuner/simulators/simphony/configs/devices"))
        ctx = ToolContext(output_dir=root, current_hardware=default_hardware())
        payload = inspect_device_config(tb, ctx, {"device_type": "mzm", "device_name": ""}).payload
        self.assertIn("error", payload)
        self.assertIn("LT_MZM", payload["candidates"])


class TestTopologyEvidenceBundle(unittest.TestCase):
    """单次调用必须提供损耗/功率任务所需的全部内容。

    该捆绑包还携带 laser/PD 配置，使「未暴露」占位符永远不会被当作借口。
    """

    def _payload(self):
        from pathlib import Path

        from photon_tuner.common.io import read_json
        from photon_tuner.core.schemas import EvaluationResult
        from photon_tuner.core.search_space import normalize_hardware_to_design_space
        from photon_tuner.toolbox.common_tool.context import ToolContext
        from photon_tuner.toolbox.device_tools import DeviceToolbox
        from photon_tuner.toolbox.device_tools.metrics import summarize_topology_evidence

        root = Path(__file__).resolve().parents[2]
        # 录制基线随仓库提交在 tests/fixtures/ 下；原先的 runs/run_agent/ 被
        # .gitignore 忽略，CI 全新检出时不存在，会直接让本组测试失效。
        run = root / "tests/fixtures/recorded_runs/run_iteration_013/iter_000"
        self.assertTrue(run.is_dir(), "needs the recorded mesh baseline (tests/fixtures/recorded_runs)")
        hw = normalize_hardware_to_design_space(
            read_json(root / "configs" / "initial_hardware_mesh.json"),
            read_json(root / "configs" / "search_space.json"),
        )
        ev = EvaluationResult(
            run_dir=run,
            hardware=json.loads((run / "hardware.json").read_text()),
            metrics=json.loads((run / "metrics.json").read_text()),
            detail_files={"report": run / "report.json"},
        )
        tb = DeviceToolbox(str(root / "photon_tuner/simulators/simphony/configs/devices"))
        return summarize_topology_evidence(
            tb, ToolContext(output_dir=run, evaluation=ev, current_hardware=hw), {}
        ).payload

    def test_bundle_carries_loss_attribution(self):
        p = self._payload()
        self.assertGreater(p["insertion_loss"]["core_insertion_loss"], 0)
        self.assertTrue(p["core_component_loss"])
        self.assertTrue(p["core_path_detail"])
        self.assertEqual(len(p["core_path_detail"]), 4)  # dac、laser、mzi、pd

    def test_bundle_carries_power_budget_inputs(self):
        p = self._payload()
        inputs = p["power_budget_inputs"]
        self.assertIn("laser", inputs)
        self.assertIn("photodetector", inputs)
        # 链路裕度计算所需的精确 cfgs。
        self.assertIn("power", inputs["laser"]["cfgs"])
        self.assertIn("sensitivity", inputs["photodetector"]["cfgs"])

    def test_bundle_numbers_reproduce_a_positive_link_margin(self):
        # run_011 中评判的要点：裕度是可计算的。
        p = self._payload()
        il = p["insertion_loss"]
        margin = (
            float(p["power_budget_inputs"]["laser"]["cfgs"]["power"])
            - (float(il["core_insertion_loss"]) + float(il["node_insertion_loss"]))
            - float(p["power_budget_inputs"]["photodetector"]["cfgs"]["sensitivity"])
        )
        self.assertGreater(margin, 0.0)
        self.assertAlmostEqual(margin, 18.28, places=2)


class TestStageDegradationIsGatedNotFatal(unittest.TestCase):
    """从不作答的智能体必须使阶段降级，而不是杀死运行。"""

    def _orchestrator(self, **kw):
        import inspect

        from photon_tuner.runtime import orchestrator

        src = inspect.getsource(orchestrator)
        self.assertIn("_last_invoke_error", src)
        self.assertNotIn("produced no final text answer\"\n            )", src)

    def test_invoke_does_not_raise_on_missing_text(self):
        from pathlib import Path
        from unittest import mock

        from photon_tuner.agents.outputs import TopologyOutput
        from photon_tuner.runtime.orchestrator import DeepAgentTuner

        class _Agent:
            name = "topology_agent"

            def invoke(self, _input, config=None):
                # 这正是轮次预算耗尽的表现：没有助手文本，图在达到限制时抛错。
                raise RuntimeError("Recursion limit of 80 reached")

        tuner = object.__new__(DeepAgentTuner)
        tuner._last_transcript = None
        tuner._last_invoke_error = None
        tuner.recursion_limit = 80
        tuner.model = mock.MagicMock()

        root = Path(__file__).resolve().parents[2]
        out = tuner._invoke(_Agent(), root, {"task": "topology_analysis"}, TopologyOutput)
        self.assertIsInstance(out, TopologyOutput)
        self.assertIsNotNone(tuner._last_invoke_error)
        self.assertIn("Recursion limit", tuner._last_invoke_error)

    def test_degraded_error_fails_row_5_7_and_forces_rework(self):
        from photon_tuner.agents.evaluate import decide

        checks = [{"row": 5.7, "name": "agent_completion", "status": "fail",
                   "detail": "GraphRecursionError: recursion limit of 40 reached"}]
        self.assertEqual(decide(checks)["decision"], "re-execute")

    def test_default_recursion_limit_leaves_room_for_tool_turns(self):
        import inspect

        from photon_tuner.runtime.orchestrator import DeepAgentTuner

        sig = inspect.signature(DeepAgentTuner.__init__)
        default = sig.parameters["recursion_limit"].default
        self.assertGreaterEqual(default, 40)


class TestPromptToolsAreOnTheRoleSurface(unittest.TestCase):
    """提示不得指示智能体调用它看不到的工具。

    校验架构提示所要求的成本证据工具与拓扑提示所要求的器件检查工具，
    都必须出现在对应的工具面上。
    """

    def _tool_names(self):
        from pathlib import Path
        from photon_tuner.common.io import read_json
        from photon_tuner.core.search_space import default_hardware
        from photon_tuner.toolbox.bindings import RunState, build_analysis_tools, build_device_tools
        from photon_tuner.toolbox.common_tool.context import ToolContext

        root = Path(__file__).resolve().parents[2]
        space = read_json(root / "configs" / "search_space.json")
        state = RunState(context=ToolContext(
            output_dir=root, current_hardware=default_hardware(), search_space=space))
        names = {t.name for t in build_analysis_tools(space, state)}
        names |= {
            t.name for t in build_device_tools(
                str(root / "photon_tuner/simulators/simphony/configs/devices"), space, state
            )
        }
        return names

    def test_every_prompt_tool_reference_exists_and_is_exposed(self):
        import re

        from photon_tuner.agents import builders, prompts

        known = self._tool_names()
        # SOLUTION_PROMPT 较特殊：它把编辑工具命名为其输出动作的 *schema*
        # （由编排器确定性应用），因此这些名字有意地从 SOLUTION_TOOLS 中省略。
        action_schema_tools = {"modify_hardware_parameter", "step_hardware_parameter", "modify_topology_template"}
        pairs = [
            ("MODEL_TOOLS", prompts.MODEL_PROMPT),
            ("ARCHITECTURE_TOOLS", prompts.ARCHITECTURE_PROMPT),
            ("TOPOLOGY_TOOLS", prompts.TOPOLOGY_PROMPT),
            ("DEVICE_TOOLS", prompts.DEVICE_PROMPT),
            ("SOLUTION_TOOLS", prompts.SOLUTION_PROMPT),
        ]
        for attr, prompt in pairs:
            surface = set(getattr(builders, attr))
            for name in sorted(known):
                if name in action_schema_tools:
                    continue
                # 仅标记实际*指示*调用的提示，即反引号包裹的工具引用。
                if re.search(rf"``{re.escape(name)}\b", prompt):
                    self.assertIn(
                        name, surface,
                        f"{attr} omits {name!r} but its prompt tells the agent to use it",
                    )

    def test_architecture_surface_has_cost_evidence_tools(self):
        from photon_tuner.agents import builders

        surface = set(builders.ARCHITECTURE_TOOLS)
        # 该阶段诊断计算/存储/器件根因，因此必须能获取实测的成本分解。
        for tool in ("summarize_energy_breakdown", "summarize_area_breakdown",
                     "summarize_device_energy_breakdown", "summarize_device_area_breakdown"):
            self.assertIn(tool, surface)

    def test_topology_surface_can_inspect_devices(self):
        from photon_tuner.agents import builders

        self.assertIn("inspect_device_config", set(builders.TOPOLOGY_TOOLS))


class TestOpticalAnalysisPropagatesPowerBudget(unittest.TestCase):
    """结构化的功率预算必须到达下游 device/solution 阶段，而不仅是产生它的 topology 阶段。"""

    def test_orchestrator_forwards_power_budget(self):
        import inspect

        from photon_tuner.runtime import orchestrator

        src = inspect.getsource(orchestrator)
        # 两个交接点（device 与 solution）都必须携带数值字典。
        self.assertGreaterEqual(src.count('"power_budget": dict('), 2)
        self.assertIn('getattr(topo_out, "power_budget"', src)


class TestOpticalStructureExposesDeviceVariants(unittest.TestCase):
    """智能体必须能够核实实际使用的 laser/node 器件变体。

    结构工具必须携带器件表，使 laser 类型等可被核实。
    """

    def test_device_tables_present(self):
        from pathlib import Path
        from photon_tuner.common.io import read_json
        from photon_tuner.core.schemas import EvaluationResult
        from photon_tuner.core.search_space import normalize_hardware_to_design_space
        from photon_tuner.toolbox.common_tool.context import ToolContext
        from photon_tuner.toolbox.device_tools.metrics import summarize_optical_structure

        root = Path(__file__).resolve().parents[2]
        space = read_json(root / "configs" / "search_space.json")
        hw = normalize_hardware_to_design_space(
            read_json(root / "configs" / "initial_hardware_mesh.json"), space
        )
        ctx = ToolContext(output_dir=root, evaluation=None, current_hardware=hw)
        payload = summarize_optical_structure(None, ctx, {}).payload["optical_structure"]
        self.assertEqual(payload["core_kind"], "mzi_mesh_clements")
        self.assertIn("mesh_core_devices", payload)
        self.assertIn("mesh_node_devices", payload)
        # laser 类型必须能从工具结果本身核实。
        self.assertEqual(payload["mesh_core_devices"]["off_chip_laser"], "LT_Laser")
        self.assertEqual(payload["mesh_node_devices"]["phase_shifter"], "IHO_PS")


class TestDeviceRecordEnrichment(unittest.TestCase):
    """器件排名必须指名组件及其占比，而不是裸键名。"""

    def test_device_key_is_parsed_and_share_computed(self):
        from photon_tuner.toolbox.common_tool._report_utils import _summarize_device_records

        rows = _summarize_device_records(
            {
                "core_off_chip_laser_0_i_laser-LT_Laser": {"count": 12, "total_energy": 96.0},
                "node_phase_shifter_0_i1-IHO_PS": {"count": 4, "total_energy": 4.0},
            },
            metric="total_energy",
        )
        top = rows[0]
        self.assertEqual(top["level"], "core")
        self.assertEqual(top["device_type"], "off_chip_laser")
        self.assertEqual(top["instance"], "i_laser")
        self.assertEqual(top["device_name"], "LT_Laser")
        self.assertAlmostEqual(top["total_energy_share"], 0.96, places=6)
        self.assertAlmostEqual(rows[1]["total_energy_share"], 0.04, places=6)

    def test_unparseable_key_still_returns_identity(self):
        from photon_tuner.toolbox.common_tool._report_utils import _summarize_device_records

        rows = _summarize_device_records({"LT": {"count": 1, "total_energy": 1.0}},
                                         metric="total_energy")
        self.assertEqual(rows[0]["device"], "LT")
        self.assertAlmostEqual(rows[0]["total_energy_share"], 1.0, places=6)


class TestNoToolCanAbortTheRun(unittest.TestCase):
    """每个已注册的工具都必须能经受住畸形调用，不能因未处理的域异常而中止运行。"""

    def _run_state(self):
        from pathlib import Path
        from photon_tuner.common.io import read_json
        from photon_tuner.core.search_space import default_hardware
        from photon_tuner.toolbox.bindings import RunState
        from photon_tuner.toolbox.common_tool.context import ToolContext

        root = Path(__file__).resolve().parents[2]
        space = read_json(root / "configs" / "search_space.json")
        ctx = ToolContext(
            output_dir=root,
            current_hardware=default_hardware(),
            search_space=space,
        )
        return RunState(context=ctx), space, root

    def test_every_device_tool_survives_bad_arguments(self):
        from photon_tuner.toolbox.bindings import build_device_tools

        state, space, root = self._run_state()
        tools = build_device_tools(
            str(root / "photon_tuner/simulators/simphony/configs/devices"), space, state
        )
        self.assertTrue(tools)
        for tool in tools:
            # 完全不带参数的调用会走各处理器自己的校验；它必须返回错误载荷，绝不抛出。
            try:
                out = tool.invoke({})
            except Exception as exc:  # noqa: BLE001
                self.fail(f"{tool.name} raised {type(exc).__name__}: {exc}")
            self.assertIsNotNone(out, tool.name)

    def test_every_analysis_tool_survives_bad_arguments(self):
        from photon_tuner.toolbox.bindings import build_analysis_tools

        state, space, _root = self._run_state()
        tools = build_analysis_tools(space, state)
        self.assertTrue(tools)
        for tool in tools:
            try:
                out = tool.invoke({})
            except Exception as exc:  # noqa: BLE001
                self.fail(f"{tool.name} raised {type(exc).__name__}: {exc}")
            self.assertIsNotNone(out, tool.name)

    def test_bad_edit_arguments_return_error_not_exception(self):
        from photon_tuner.toolbox.bindings import build_analysis_tools

        state, space, _root = self._run_state()
        tools = {t.name: t for t in build_analysis_tools(space, state)}
        tool = tools["modify_hardware_parameter"]
        # 缺少必需的 'updates' 列表 -> 可恢复的错误载荷。
        out = tool.invoke({})
        self.assertIsInstance(out, dict)
        self.assertIn("error", out)


class TestArchitectureBoundaryGuard(unittest.TestCase):
    """第 5.4 行——架构阶段不得越界到 device/topology 工作。

    提示仅把「器件级」列为*成本归因类别*；具体的器件变体选择属于 device 阶段，
    拓扑 L4 杠杆属于 topology 阶段。circuit_recommendations 必须停留在
    带枚举值的架构级参数上。
    """

    def _out(self, recs):
        return ArchitectureOutput(
            primary_impact="energy", dominant_root_cause="device",
            layer_diagnoses=[], retrieval_description="x",
            root_cause_summary={"device": 1.0}, recommended_focus=["x"],
            circuit_recommendations=recs,
        )

    def test_device_variant_leakage_fails(self):
        from photon_tuner.agents.evaluate import architecture_legality_check

        row = architecture_legality_check(self._out([
            {"parameter": "phase_shifter", "value": "ITO_PS", "rationale": "lower loss"}
        ]))
        self.assertEqual(row["status"], "fail")
        self.assertIn("architecture stage's authority", row["detail"])

    def test_topology_l4_leakage_fails(self):
        from photon_tuner.agents.evaluate import architecture_legality_check

        row = architecture_legality_check(self._out([
            {"parameter": "sharing_policy", "value": "adc_tia_shared", "rationale": "share"}
        ]))
        self.assertEqual(row["status"], "fail")

    def test_legal_architecture_params_pass(self):
        from photon_tuner.agents.evaluate import architecture_legality_check

        row = architecture_legality_check(self._out([
            {"parameter": "tiles", "value": 4, "rationale": "fewer tiles"},
            {"parameter": "work_freq", "value": 10, "rationale": "clock"},
        ]))
        self.assertEqual(row["status"], "pass")

    def test_non_enumerated_value_fails(self):
        from pathlib import Path

        from photon_tuner.common.io import read_json
        from photon_tuner.core.search_space import build_design_space, default_hardware
        from photon_tuner.agents.evaluate import architecture_legality_check

        root = Path(__file__).resolve().parents[2]
        ds = build_design_space(
            read_json(root / "configs" / "search_space.json"), default_hardware()
        )
        row = architecture_legality_check(
            self._out([{"parameter": "tiles", "value": 9, "rationale": "invented"}]),
            {"design_space": ds},
        )
        self.assertEqual(row["status"], "fail")
        self.assertIn("not an enumerated candidate", row["detail"])

    def test_row_5_4_is_hard_in_decide(self):
        from photon_tuner.agents.evaluate import decide

        checks = [{"row": 5.4, "name": "architecture_legality", "status": "fail",
                   "detail": "phase_shifter is outside the architecture stage's authority"}]
        self.assertEqual(decide(checks)["decision"], "re-execute")


class _TopoOut(BaseModel):
    recommended_template: object = None
    circuit_recommendations: list = []


def _sol(actions):
    return SolutionOutput(strategy="s", actions=actions, link_edits=[], rationale="r")


class TestTopologyBinding(unittest.TestCase):
    def test_pass_without_recommendation(self):
        check = topology_binding_check("solution", _sol([]), {})
        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["row"], 5.8)

    def test_pass_when_switch_proposed(self):
        out = _sol([
            {"name": "modify_topology_template",
             "arguments": {"core_kind": "mzi_mesh_clements"}},
        ])
        check = topology_binding_check("solution", out, {
            "optical_analysis": {"recommended_template": "mzi_mesh_clements"},
            "current_core_kind": "dptc_crossbar",
            "mode": "analytic",
        })
        self.assertEqual(check["status"], "pass")

    def test_pass_when_template_already_current(self):
        check = topology_binding_check("solution", _sol([]), {
            "optical_analysis": {"recommended_template": "mzi_mesh_clements"},
            "current_core_kind": "mzi_mesh_clements",
            "mode": "analytic",
        })
        self.assertEqual(check["status"], "pass")

    def test_pass_for_illegal_recommendation(self):
        # 非法的推荐（不存在的模板名）不具约束力：topology 阶段自身的结构行已判定
        # 其失败，solution 不受其约束。
        check = topology_binding_check("solution", _sol([]), {
            "optical_analysis": {"recommended_template": "bogus_template"},
            "current_core_kind": "dptc_crossbar",
            "mode": "real",
        })
        self.assertEqual(check["status"], "pass")

    def test_fail_when_switch_omitted(self):
        check = topology_binding_check("solution", _sol([]), {
            "optical_analysis": {"recommended_template": "mzi_mesh_clements"},
            "current_core_kind": "dptc_crossbar",
            "mode": "analytic",
        })
        self.assertEqual(check["status"], "fail")
        self.assertIn("no core_kind edit", check["detail"])

    def test_binding_row_is_hard_in_decide(self):
        check = topology_binding_check("solution", _sol([]), {
            "optical_analysis": {"recommended_template": "mzi_mesh_clements"},
            "current_core_kind": "dptc_crossbar",
            "mode": "analytic",
        })
        self.assertEqual(decide([check])["decision"], "re-execute")


if __name__ == "__main__":
    unittest.main()
