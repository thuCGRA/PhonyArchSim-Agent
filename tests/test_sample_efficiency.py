import unittest

from baselines.sample_efficiency import analyse, first_hit


def _run(method, workload, curve):
    return {
        "method": method,
        "workload": workload,
        "curve": curve,
        "initial_metrics": {"objective": curve[0]},
    }


class SampleEfficiencyTest(unittest.TestCase):
    def test_first_hit_uses_round_zero_as_initial_design(self):
        self.assertEqual(first_hit([1.0, 0.99, 0.95], 0.95), 2)
        self.assertIsNone(first_hit([1.0, 0.99], 0.95))

    def test_best_traditional_target_and_iteration_savings(self):
        result = analyse([
            _run("rs", "toy", [100, 98, 96, 94, 92, 90]),
            _run("sa", "toy", [100, 97, 94, 91, 89, 88]),
            _run("agent", "toy", [100, 95, 91, 88, 87, 86]),
        ], max_round=5, checkpoints=(2, 5), improvement_targets=(5,))
        row = next(r for r in result["matched_quality"]
                   if r["target_name"] == "best_traditional_at_50")
        self.assertAlmostEqual(row["target_normalized_objective"], 0.88)
        self.assertEqual(row["traditional_method"], "sa")
        self.assertEqual(row["traditional_round"], 5)
        self.assertEqual(row["agent_round"], 3)
        self.assertEqual(row["iterations_saved"], 2)
        self.assertAlmostEqual(row["iterations_saved_pct"], 40.0)

    def test_unreached_agent_is_reported_without_redefining_target(self):
        result = analyse([
            _run("rs", "toy", [100, 95, 90]),
            _run("agent", "toy", [100, 98, 96]),
        ], max_round=2, checkpoints=(2,), improvement_targets=())
        row = result["matched_quality"][0]
        self.assertFalse(row["agent_reached"])
        self.assertIsNone(row["agent_round"])
        self.assertIsNone(row["iterations_saved"])


if __name__ == "__main__":
    unittest.main()
