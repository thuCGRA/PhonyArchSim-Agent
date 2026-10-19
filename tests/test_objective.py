"""Validate the photonic objective and insertion-loss budget penalty."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.core.objective import (
    base_objective,
    il_budget_penalty,
    metrics_with_objective,
    objective_weights,
)


class TestObjective(unittest.TestCase):
    def test_base_objective_is_product(self):
        metrics = {"latency_s": 2.0, "pJ_per_mac": 3.0, "area_um2": 4.0}
        self.assertAlmostEqual(base_objective(metrics), 24.0)

    def test_missing_metric_raises(self):
        with self.assertRaises(KeyError):
            base_objective({"latency_s": 1.0})

    def test_metrics_with_objective_enriches(self):
        metrics = {"latency_s": 2.0, "pJ_per_mac": 3.0, "area_um2": 4.0}
        enriched = metrics_with_objective(metrics)
        self.assertAlmostEqual(enriched["objective"], 24.0)
        self.assertAlmostEqual(enriched["base_objective"], 24.0)
        self.assertAlmostEqual(enriched["il_budget_penalty"], 1.0)

    def test_energy_alias_maps_to_pj_per_mac(self):
        weights = objective_weights({"objective_weights": {"energy": 2.0}})
        self.assertAlmostEqual(weights["pJ_per_mac"], 2.0)
        self.assertAlmostEqual(weights["latency_s"], 1.0)

    def test_il_budget_penalty_neutral_without_target(self):
        self.assertEqual(
            il_budget_penalty({"core_insertion_loss": 10.0}, None)["il_budget_penalty"],
            1.0,
        )

    def test_il_budget_penalty_neutral_when_il_missing(self):
        self.assertEqual(
            il_budget_penalty({}, {"il_budget": 20.0})["il_budget_penalty"], 1.0
        )

    def test_il_budget_penalty_one_sided_under_budget(self):
        penalty = il_budget_penalty(
            {"core_insertion_loss": 10.0}, {"il_budget": 20.0}
        )["il_budget_penalty"]
        self.assertAlmostEqual(penalty, 1.0)

    def test_il_budget_penalty_grows_over_budget(self):
        # actual 30 / target 20 -> ratio 1.5 -> penalty = 1 + 1 * 0.5^2 = 1.25
        penalty = il_budget_penalty(
            {"core_insertion_loss": 30.0}, {"il_budget": 20.0}
        )["il_budget_penalty"]
        self.assertAlmostEqual(penalty, 1.25)


if __name__ == "__main__":
    unittest.main()
