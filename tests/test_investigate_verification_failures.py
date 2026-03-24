import unittest

from core.investigate_verification_failures import investigate_verification_failure


class TestInvestigateVerificationFailures(unittest.TestCase):
    def test_extracts_signals_and_signature(self):
        result = investigate_verification_failure(
            {
                "failures": ["SyntaxError: invalid syntax"],
                "logs": "SyntaxError: invalid syntax\npytest run failed",
            },
            root_cause_analysis={"patterns": ["syntax_error"]},
            context={"goal": "Fix parser", "phase": "verify", "route": "plan"},
        )

        self.assertEqual(result["failure_signature"], "syntaxerror: invalid syntax")
        self.assertIn("syntax_error", result["signals"])
        self.assertEqual(result["context"]["phase"], "verify")

    def test_detects_repeated_failures_from_history(self):
        history = [
            {
                "phase_outputs": {
                    "verification": {
                        "status": "fail",
                        "failures": ["ModuleNotFoundError: No module named 'foo'"],
                    }
                }
            }
        ]

        result = investigate_verification_failure(
            {"failures": ["ModuleNotFoundError: No module named 'foo'"], "logs": ""},
            history=history,
        )

        self.assertTrue(result["repeated_failure_detected"])
        self.assertEqual(result["history_matches"], 1)

    def test_handles_empty_inputs(self):
        result = investigate_verification_failure(None)

        self.assertEqual(result["signals"], ["unknown_failure"])
        self.assertFalse(result["repeated_failure_detected"])


if __name__ == "__main__":
    unittest.main()
