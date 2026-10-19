"""Progress-reporter tests: live step output is emitted, quiet stays silent."""

import io
import unittest

from photon_tuner.runtime.react_loop import run_react_tool_loop
from photon_tuner.common.progress import ProgressReporter, summarize_payload
from photon_tuner.common.tool_types import ToolResult, ToolSpec


class _ScriptedLLM:
    retry_attempts = 0

    def __init__(self, responses):
        self._responses = list(responses)

    def complete_json_messages(self, messages):
        return self._responses.pop(0)


class _FakeToolbox:
    def __init__(self, payload=None):
        self._payload = payload or {"layers": list(range(25))}
        self.calls = []

    def specs(self):
        return [ToolSpec("fake_probe", "probe tool", {"type": "object"})]

    def run(self, name, context, arguments=None):
        self.calls.append((name, arguments))
        return ToolResult(name=name, payload=self._payload)


def _scripted_llm():
    return _ScriptedLLM(
        [
            {"action": "tool", "tool": {"name": "fake_probe", "arguments": {"top": 3}}},
            {"action": "finish"},
        ]
    )


class TestLiveStepReporting(unittest.TestCase):
    def test_tool_step_is_rendered_with_size_summary(self):
        stream = io.StringIO()
        progress = ProgressReporter(enabled=True, stream=stream)
        progress.begin_agent(
            component="agent:model_level", iteration=1, total_iterations=2, detail="go"
        )
        run_react_tool_loop(
            llm=_scripted_llm(),
            toolbox=_FakeToolbox(),
            system_prompt="sp",
            task="t",
            agent_name="model_level",
            agent_goal="g",
            context_payload={},
            context=object(),
            max_steps=4,
            progress=progress,
        )
        progress.end_agent(detail="done")
        out = stream.getvalue()
        self.assertIn("[START]", out)
        self.assertIn("[step 1/4]", out)
        self.assertIn("deciding next tool", out)
        self.assertIn("[TOOL]", out)
        self.assertIn("fake_probe(top=3)", out)
        self.assertIn("25 layers", out)  # size-aware payload summary
        self.assertIn("[END]", out)
        self.assertIn("[DONE]", out)

    def test_quiet_reporter_emits_nothing(self):
        stream = io.StringIO()
        progress = ProgressReporter(enabled=False, stream=stream)
        progress.begin_agent(component="agent:model_level", iteration=1, total_iterations=1)
        run_react_tool_loop(
            llm=_scripted_llm(),
            toolbox=_FakeToolbox(),
            system_prompt="sp",
            task="t",
            agent_name="a",
            agent_goal="g",
            context_payload={},
            context=object(),
            max_steps=2,
            progress=progress,
        )
        progress.end_agent()
        self.assertEqual(stream.getvalue(), "")

    def test_tool_error_is_reported(self):
        class _BoomToolbox(_FakeToolbox):
            def run(self, name, context, arguments=None):
                raise ValueError("bad argument")

        stream = io.StringIO()
        progress = ProgressReporter(enabled=True, stream=stream)
        progress.begin_agent(component="agent:optical_level", iteration=1, total_iterations=1)
        run_react_tool_loop(
            llm=_scripted_llm(),
            toolbox=_BoomToolbox(),
            system_prompt="sp",
            task="t",
            agent_name="d",
            agent_goal="g",
            context_payload={},
            context=object(),
            max_steps=4,
            progress=progress,
        )
        out = stream.getvalue()
        self.assertIn("[ERROR]", out)
        self.assertIn("bad argument", out)

    def test_decision_and_metric_delta_lines(self):
        stream = io.StringIO()
        progress = ProgressReporter(enabled=True, stream=stream)
        progress.decision(
            iteration=1,
            total_iterations=1,
            component="agent:model_level",
            message="energy dominates",
        )
        progress.metric_delta(
            iteration=1,
            total_iterations=1,
            component="simulator",
            before={"energy_pj": 100.0, "objective": 10.0},
            after={"energy_pj": 50.0, "objective": 5.0},
        )
        out = stream.getvalue()
        self.assertIn("[DECISION] energy dominates", out)
        self.assertIn("-50.0%", out)

    def test_summarize_payload(self):
        self.assertIn("25 layers", summarize_payload({"layers": list(range(25))}))
        self.assertIn("3 keys", summarize_payload({"a": 1, "b": 2, "c": 3}))
        self.assertIn("5 items", summarize_payload([1, 2, 3, 4, 5]))


class TestEditReporting(unittest.TestCase):
    def test_edit_events_are_rendered(self):
        stream = io.StringIO()
        progress = ProgressReporter(enabled=True, stream=stream)
        progress.begin_agent(
            component="agent:solution_generation", iteration=1, total_iterations=1
        )
        progress.edit(
            name="modify_device_parameter",
            arguments={"device_name": "LT_MZM", "parameter": "efficiency", "value": 50},
            status="ok",
        )
        progress.edit(
            name="step_hardware_parameter",
            arguments={"parameter": "core_height", "value": "increase"},
            status="error",
            error="invalid literal for int() with base 10: 'increase'",
        )
        out = stream.getvalue()
        self.assertIn("[EDIT][ok]", out)
        self.assertIn("LT_MZM", out)
        self.assertIn("[EDIT][FAIL]", out)
        self.assertIn("invalid literal", out)

    def test_solution_agent_surfaces_rejected_edits(self):
        from photon_tuner.agents.solution import SolutionGenerationAgent

        class _RejectingToolbox:
            def run(self, name, context, arguments=None):
                raise TypeError("bad value for parameter")

        stream = io.StringIO()
        progress = ProgressReporter(enabled=True, stream=stream)
        progress.begin_agent(
            component="agent:solution_generation", iteration=1, total_iterations=1
        )

        agent = SolutionGenerationAgent.__new__(SolutionGenerationAgent)
        agent.progress = progress
        applied, failed = [], []
        agent._run_edit(
            _RejectingToolbox(),
            "step_hardware_parameter",
            {"parameter": "core_height", "value": "increase"},
            object(),
            applied,
            failed,
        )

        out = stream.getvalue()
        self.assertIn("[EDIT][FAIL]", out)
        self.assertIn("bad value for parameter", out)
        self.assertEqual(applied, [])
        self.assertEqual(
            failed,
            [{"name": "step_hardware_parameter", "error": "bad value for parameter"}],
        )


if __name__ == "__main__":
    unittest.main()
