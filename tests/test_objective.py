import unittest

from gnn_amg.objective import RewardBaseline, importance_weight, time_reward


class ObjectiveTests(unittest.TestCase):
    def test_seconds_do_not_average_speedup_percentages(self):
        simple = time_reward(1, .5, 50.5, 200, True)
        hard = time_reward(100, 110, 50.5, 200, True)
        self.assertLess(simple+hard, 0)
        self.assertAlmostEqual(simple+hard, -9.5/50.5)

    def test_sampling_correction_recovers_event_mean(self):
        masses, probabilities, rewards = [2, 8], [.8, .2], [4, -1]
        expected = sum(p*importance_weight(m, p, 10)*r for m, p, r in zip(masses, probabilities, rewards))
        self.assertAlmostEqual(expected, 0)

    def test_failure_is_worse_than_slowest_valid_result(self):
        failure = time_reward(.01, float("nan"), 10, 30, False)
        valid = time_reward(.01, 30, 10, 30, True)
        self.assertLess(failure, valid)
        self.assertEqual(failure, time_reward(.01, 30.1, 10, 30, True))

    def test_baseline_has_no_current_candidate_leakage(self):
        baseline = RewardBaseline()
        self.assertEqual(baseline.get("a"), 0)
        baseline.update([("a", 2), ("a", 4)])
        self.assertEqual(baseline.get("a"), 3)
        self.assertEqual(baseline.get("unseen"), 0)


if __name__ == "__main__":
    unittest.main()
