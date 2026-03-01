"""Unit tests for core/explain.py."""
import unittest
from core.explain import format_decision_log

class TestExplain(unittest.TestCase):
    def test_format_decision_log_empty(self):
        result = format_decision_log([])
        self.assertEqual(result, "")

    def test_format_decision_log_minimal(self):
        history = [
            {
                "cycle_id": "c1",
                "phase_outputs": {
                    "plan": {"steps": [1, 2]},
                    "verification": {"status": "pass"}
                }
            }
        ]
        result = format_decision_log(history)
        self.assertIn("Cycle: c1", result)
        self.assertIn("Plan steps: 2", result)
        self.assertIn("Verification: pass", result)

if __name__ == "__main__":
    unittest.main()
