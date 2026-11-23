import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photon_tuner.runtime.scheduler import (
    ANALYSIS_LAYERS, MODES, ScheduleSignals, SchedulerConfig, decide,
    fixed_mode_decision, summarize_history,
)


class SchedulerTest(unittest.TestCase):
    def mature(self, **kwargs):
        values = dict(
            iteration=4,
            improved_case_count=3,
            improved_case_layers=["architecture", "topology", "device"],
            pipeline_evidence_ages={"architecture": 1, "topology": 2, "device": 1},
            pipeline_count=1,
            mode_history=["pipeline"],
        )
        values.update(kwargs)
        return ScheduleSignals(**values)

    def test_exactly_three_modes(self):
        self.assertEqual(set(MODES), {"serial", "pipeline", "focus"})

    def test_immature_memory_runs_full_serial_chain(self):
        d = decide(ScheduleSignals(iteration=1))
        self.assertEqual(d.mode, "serial")
        self.assertEqual(d.stages_to_run, list(ANALYSIS_LAYERS))

    def test_mature_nonconverged_runs_three_layer_pipeline(self):
        d = decide(self.mature(objective_history=[100, 70, 49, 34.3]))
        self.assertEqual(d.mode, "pipeline")
        self.assertEqual(d.stages_to_run, list(ANALYSIS_LAYERS))
        self.assertEqual(d.evidence_offsets, {"architecture": 0, "topology": 1, "device": 2})

    def test_mature_converged_focuses_single_layer(self):
        history = [100.0]
        for _ in range(10):
            history.append(history[-1] * 0.95)
        d = decide(self.mature(iteration=12, objective_history=history))
        self.assertEqual(d.mode, "focus")
        self.assertEqual(len(d.stages_to_run), 1)
        self.assertTrue(d.signals["converged"])

    def test_first_pipeline_must_precede_focus(self):
        d = decide(self.mature(
            iteration=5,
            objective_history=[100, 90, 81, 72.9],
            pipeline_count=0,
            mode_history=["serial", "serial", "serial", "serial"],
        ))
        self.assertEqual(d.mode, "pipeline")
        self.assertIn("before entering focus", d.reason)

    def test_focus_budget_is_a_mode_cap_not_only_a_layer_rotation(self):
        cfg = SchedulerConfig(
            convergence_window=3, convergence_delta=0.15,
            max_consecutive_focus=2,
        )
        d = decide(self.mature(
            iteration=8,
            objective_history=[100, 90, 81, 72.9, 72.9, 72.9],
            pipeline_count=1,
            mode_history=["serial", "pipeline", "focus", "focus"],
            previous_mode="focus",
        ), cfg)
        self.assertEqual(d.mode, "pipeline")
        self.assertEqual(d.signals["focus_rounds_since_pipeline"], 2)

    def test_stale_evidence_after_focus_cap_refreshes_then_pipelines(self):
        cfg = SchedulerConfig(
            convergence_window=3, convergence_delta=0.15,
            max_consecutive_focus=2,
        )
        history = ["serial", "pipeline", "focus", "focus"]
        stale = decide(self.mature(
            objective_history=[100, 90, 81, 72.9, 72.9, 72.9],
            pipeline_count=1,
            mode_history=history,
            previous_mode="focus",
            pipeline_evidence_ages={"architecture": 2, "topology": 3, "device": 3},
        ), cfg)
        self.assertEqual(stale.mode, "serial")
        refreshed = decide(self.mature(
            objective_history=[100, 90, 81, 72.9, 72.9, 72.9, 72.9],
            pipeline_count=1,
            mode_history=history + ["serial"],
            previous_mode="serial",
            pipeline_evidence_ages={"architecture": 1, "topology": 2, "device": 1},
        ), cfg)
        self.assertEqual(refreshed.mode, "pipeline")

    def test_main_phase_trend_is_serial_pipeline_focus(self):
        cold = decide(ScheduleSignals(iteration=1))
        pipeline = decide(self.mature(
            objective_history=[100, 90, 81, 72.9],
            pipeline_count=0,
            mode_history=["serial"],
        ))
        low_gain_history = [100.0]
        for _ in range(10):
            low_gain_history.append(low_gain_history[-1] * 0.95)
        focus = decide(self.mature(
            objective_history=low_gain_history,
            pipeline_count=1,
            mode_history=["serial", "pipeline"],
            previous_mode="pipeline",
        ))
        self.assertEqual([cold.mode, pipeline.mode, focus.mode],
                         ["serial", "pipeline", "focus"])

    def test_no_convergence_before_full_window(self):
        d = decide(self.mature(iteration=2, objective_history=[10, 10]))
        self.assertNotEqual(d.mode, "focus")

    def test_default_focus_requires_ten_complete_low_gain_rounds(self):
        nine = [100.0]
        for _ in range(9):
            nine.append(nine[-1] * 0.95)
        ten = nine + [nine[-1] * 0.95]
        self.assertEqual(decide(self.mature(objective_history=nine)).mode, "pipeline")
        self.assertEqual(decide(self.mature(objective_history=ten)).mode, "focus")

    def test_gain_above_ten_percent_exits_focus_to_pipeline(self):
        history = [100.0]
        for _ in range(9):
            history.append(history[-1] * 0.95)
        history.append(history[-1] * 0.89)
        decision = decide(self.mature(
            objective_history=history,
            previous_mode="focus",
            mode_history=["pipeline", "focus"],
        ))
        self.assertEqual(decision.mode, "pipeline")
        self.assertFalse(decision.signals["converged"])

    def test_default_focus_has_no_hard_round_cap(self):
        history = [100.0]
        for _ in range(10):
            history.append(history[-1] * 0.95)
        decision = decide(self.mature(
            objective_history=history,
            previous_mode="focus",
            mode_history=["pipeline"] + ["focus"] * 20,
        ))
        self.assertEqual(decision.mode, "focus")

    def test_improvement_window_uses_best_so_far_not_regression(self):
        s = self.mature(iteration=5, objective_history=[10, 8, 9, 9])
        d = decide(s, SchedulerConfig(convergence_window=2, convergence_delta=0.01))
        self.assertTrue(d.signals["converged"])

    def test_maturity_requires_case_count_and_layer_coverage(self):
        one_layer = self.mature(improved_case_layers=["architecture"])
        self.assertEqual(one_layer.maturity(SchedulerConfig()), 1 / 3)
        d = decide(one_layer)
        self.assertEqual(d.mode, "serial")

    def test_pipeline_requires_evidence_from_every_layer(self):
        for layer in ANALYSIS_LAYERS:
            for missing_key in (True, False):
                with self.subTest(layer=layer, missing_key=missing_key):
                    ages = dict.fromkeys(ANALYSIS_LAYERS, 1)
                    if missing_key:
                        del ages[layer]
                    else:
                        ages[layer] = None
                    decision = decide(self.mature(
                        objective_history=[100, 70, 49, 34.3],
                        pipeline_evidence_ages=ages,
                    ))
                    self.assertEqual(decision.mode, "serial")
                    self.assertFalse(decision.signals["pipeline_evidence_ready"])

    def test_pipeline_evidence_age_boundary_applies_to_every_layer(self):
        for layer in ANALYSIS_LAYERS:
            for age, expected in ((2, "pipeline"), (3, "serial")):
                with self.subTest(layer=layer, age=age):
                    ages = dict.fromkeys(ANALYSIS_LAYERS, 1)
                    ages[layer] = age
                    decision = decide(self.mature(
                        objective_history=[100, 70, 49, 34.3],
                        pipeline_evidence_ages=ages,
                    ))
                    self.assertEqual(decision.mode, expected)
                    self.assertEqual(decision.signals["pipeline_evidence_ready"], expected == "pipeline")

    def test_focus_range_uses_legacy_single_threshold_when_exit_is_none(self):
        cfg = SchedulerConfig(
            convergence_window=3, convergence_delta=0.15, focus_exit_delta=None,
        )
        for previous in (None, "serial", "pipeline", "focus"):
            with self.subTest(previous_mode=previous):
                decision = decide(self.mature(
                    previous_mode=previous, objective_history=[100, 90, 81, 72.9],
                ), cfg)
                self.assertEqual(decision.mode, "focus")
                self.assertEqual(decision.signals["convergence_threshold"], 0.15)

    def test_focus_hysteresis_uses_wider_exit_threshold(self):
        cfg = SchedulerConfig(
            convergence_window=3,
            convergence_delta=0.0025,
            focus_exit_delta=0.01,
        )
        enter = decide(self.mature(
            previous_mode="pipeline",
            objective_history=[100, 99.9, 99.8, 99.7],
        ), cfg)
        stay = decide(self.mature(
            previous_mode="focus",
            objective_history=[100, 99.5, 99.0, 98.5],
        ), cfg)
        leave = decide(self.mature(
            previous_mode="focus",
            objective_history=[100, 98.5, 98.5, 98.5],
        ), cfg)
        self.assertEqual(enter.mode, "focus")
        self.assertEqual(stay.mode, "focus")
        self.assertEqual(leave.mode, "pipeline")
        self.assertEqual(stay.signals["convergence_threshold"], 0.01)

    def test_single_focus_range_boundaries_for_all_previous_modes(self):
        cfg = SchedulerConfig(
            convergence_window=3, convergence_delta=0.15, focus_exit_delta=None,
        )
        for previous in (None, "serial", "pipeline", "focus"):
            for after, expected in ((100, "focus"), (85.1, "focus"), (85, "focus"), (84.9, "pipeline")):
                with self.subTest(previous_mode=previous, after=after):
                    decision = decide(self.mature(
                        previous_mode=previous, objective_history=[100, after, after, after],
                    ), cfg)
                    self.assertEqual(decision.mode, expected)

    def test_one_gain_outside_the_range_prevents_focus(self):
        # Two zero gains do not hide the 20% gain inside the full window.
        cfg = SchedulerConfig(
            convergence_window=3, convergence_delta=0.15, focus_exit_delta=None,
        )
        for history in ([100, 80, 80, 80], [100, 100, 80, 80], [100, 100, 100, 80]):
            with self.subTest(history=history):
                self.assertEqual(decide(self.mature(objective_history=history), cfg).mode, "pipeline")

    def test_custom_thresholds_remain_supported(self):
        cfg = SchedulerConfig(convergence_window=3, convergence_delta=0.05)
        s = self.mature(objective_history=[100, 90, 81, 72.9])
        self.assertEqual(decide(s, cfg).mode, "pipeline")
        self.assertEqual(decide(
            s, SchedulerConfig(convergence_window=3, convergence_delta=0.10)
        ).mode, "focus")

    def test_fixed_mode_validation_and_plans(self):
        self.assertEqual(fixed_mode_decision("serial").stages_to_run, list(ANALYSIS_LAYERS))
        pipeline = fixed_mode_decision("pipeline", "device")
        self.assertEqual(pipeline.stages_to_run, list(ANALYSIS_LAYERS))
        self.assertEqual(pipeline.focus_layer, "device")
        with self.assertRaises(ValueError):
            fixed_mode_decision("other")

    def test_summary(self):
        out = summarize_history([{"mode": "serial"}, {"mode": "focus"}])
        self.assertEqual(out["mode_counts"], {"serial": 1, "focus": 1})


if __name__ == "__main__":
    unittest.main()
