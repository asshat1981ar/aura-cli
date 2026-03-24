import unittest

from core.remediation_plan import build_remediation_plan


class TestRemediationPlan(unittest.TestCase):
    def test_builds_retry_plan_with_deduped_hints(self):
        plan = build_remediation_plan(
            {"failures": ["AssertionError: expected 2 got 1"], "logs": ""},
            route="act",
            analysis_suggestion="Inspect the failing assertion.",
            root_cause_analysis={
                "recommended_actions": [
                    "Inspect the failing assertion.",
                    "Update the implementation or test deliberately.",
                ]
            },
            investigation={"signals": ["assertion_failure"], "repeated_failure_detected": False},
            context={"goal": "Fix assertion", "phase": "verify", "route": "act"},
        )

        self.assertEqual(plan["route"], "retry")
        self.assertIn("AssertionError: expected 2 got 1", plan["fix_hints"])
        self.assertEqual(plan["fix_hints"].count("Inspect the failing assertion."), 1)
        self.assertIn("Compare the failing assertion against current behavior and update code or tests deliberately.", plan["next_checks"])

    def test_builds_replan_for_repeated_failure(self):
        plan = build_remediation_plan(
            {"failures": ["SyntaxError: invalid syntax"], "logs": "SyntaxError: invalid syntax"},
            route="plan",
            root_cause_analysis={"recommended_actions": ["Review the generated file structure."]},
            investigation={"signals": ["syntax_error"], "repeated_failure_detected": True},
            context={"goal": "Fix parser", "phase": "verify", "route": "plan"},
        )

        self.assertEqual(plan["route"], "replan")
        self.assertTrue(plan["repeated_failure_detected"])
        self.assertTrue(any("repeating" in item.lower() for item in plan["fix_hints"]))

    def test_builds_skip_plan_for_environment_failure(self):
        plan = build_remediation_plan(
            {"failures": ["Permission denied writing /tmp/foo"], "logs": ""},
            route="skip",
            investigation={"signals": ["permission"], "repeated_failure_detected": False},
        )

        self.assertEqual(plan["route"], "skip")
        self.assertEqual(plan["operator_action"], "review_environment")


if __name__ == "__main__":
    unittest.main()
