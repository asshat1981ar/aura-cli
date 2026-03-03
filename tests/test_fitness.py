import unittest
from core.fitness import FitnessFunction

class TestFitnessFunction(unittest.TestCase):
    def setUp(self):
        self.ff = FitnessFunction()

    def test_calculate_perfect_score(self):
        metrics = {
            "success_rate": 1.0,
            "tokens_per_action": 5000,
            "baseline_tokens": 5000,
            "complexity_delta": 0.0
        }
        score = self.ff.calculate(metrics)
        # Expected: 0.5*1.0 + 0.3*exp(0) + 0.2*(1.0-0.0) = 0.5 + 0.3 + 0.2 = 1.0
        self.assertEqual(score, 1.0)

    def test_calculate_fail_score(self):
        metrics = {
            "success_rate": 0.0,
            "tokens_per_action": 10000,
            "baseline_tokens": 5000,
            "complexity_delta": 1.0
        }
        score = self.ff.calculate(metrics)
        # Expected: 
        # sr: 0.5 * 0.0 = 0.0
        # eff: 0.3 * exp(-(10000-5000)/5000) = 0.3 * exp(-1) = 0.3 * 0.3678 = 0.11034
        # qual: 0.2 * (1.0 - 1.0) = 0.0
        # total: ~0.1103
        self.assertLess(score, 0.2)
        self.assertGreater(score, 0.1)

    def test_custom_weights(self):
        ff = FitnessFunction(weights={"success_rate": 1.0, "token_efficiency": 0.0, "complexity_penalty": 0.0})
        metrics = {"success_rate": 0.8}
        score = ff.calculate(metrics)
        self.assertEqual(score, 0.8)

if __name__ == "__main__":
    unittest.main()
