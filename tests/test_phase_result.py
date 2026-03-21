"""Tests for phase confidence scoring and routing."""
import unittest

from core.phase_result import PhaseResult, NextAction, ConfidenceRouter


class TestPhaseResult(unittest.TestCase):
    def test_defaults(self):
        r = PhaseResult(phase="plan")
        self.assertEqual(r.phase, "plan")
        self.assertEqual(r.confidence, 0.5)
        self.assertEqual(r.suggested_next, NextAction.CONTINUE)
        self.assertFalse(r.is_high_confidence)
        self.assertFalse(r.is_low_confidence)

    def test_high_confidence(self):
        r = PhaseResult(phase="act", confidence=0.9)
        self.assertTrue(r.is_high_confidence)
        self.assertFalse(r.is_low_confidence)

    def test_low_confidence(self):
        r = PhaseResult(phase="act", confidence=0.1)
        self.assertFalse(r.is_high_confidence)
        self.assertTrue(r.is_low_confidence)

    def test_to_dict(self):
        r = PhaseResult(phase="verify", confidence=0.75,
                        suggested_next=NextAction.RETRY,
                        reasoning="test flaky")
        d = r.to_dict()
        self.assertEqual(d["phase"], "verify")
        self.assertEqual(d["confidence"], 0.75)
        self.assertEqual(d["suggested_next"], "retry")
        self.assertEqual(d["reasoning"], "test flaky")

    def test_with_output(self):
        r = PhaseResult(phase="plan", output={"steps": [1, 2, 3]})
        self.assertEqual(r.output["steps"], [1, 2, 3])


class TestConfidenceRouter(unittest.TestCase):
    def setUp(self):
        self.router = ConfidenceRouter()

    def test_defaults(self):
        self.assertIn("replan_below", self.router.thresholds)
        self.assertEqual(len(self.router.phase_history), 0)

    def test_record(self):
        r = PhaseResult(phase="plan", confidence=0.8)
        self.router.record(r)
        self.assertEqual(len(self.router.phase_history), 1)

    def test_should_replan_from_suggestion(self):
        r = PhaseResult(phase="act", suggested_next=NextAction.REPLAN)
        self.assertTrue(self.router.should_replan(r))

    def test_should_replan_from_low_plan_confidence(self):
        r = PhaseResult(phase="plan", confidence=0.1)
        self.assertTrue(self.router.should_replan(r))

    def test_should_not_replan_high_confidence(self):
        r = PhaseResult(phase="plan", confidence=0.8)
        self.assertFalse(self.router.should_replan(r))

    def test_should_escalate_from_suggestion(self):
        r = PhaseResult(phase="act", suggested_next=NextAction.ESCALATE)
        self.assertTrue(self.router.should_escalate(r))

    def test_should_escalate_very_low_confidence(self):
        r = PhaseResult(phase="act", confidence=0.1)
        self.assertTrue(self.router.should_escalate(r))

    def test_should_escalate_declining_trend(self):
        self.router.record(PhaseResult(phase="plan", confidence=0.7))
        self.router.record(PhaseResult(phase="critique", confidence=0.5))
        r = PhaseResult(phase="act", confidence=0.3)
        self.router.record(r)
        self.assertTrue(self.router.should_escalate(r))

    def test_should_not_escalate_stable(self):
        self.router.record(PhaseResult(phase="plan", confidence=0.7))
        self.router.record(PhaseResult(phase="critique", confidence=0.7))
        r = PhaseResult(phase="act", confidence=0.7)
        self.router.record(r)
        self.assertFalse(self.router.should_escalate(r))

    def test_should_retry(self):
        r = PhaseResult(phase="act", confidence=0.2)
        self.assertTrue(self.router.should_retry(r))

    def test_should_not_retry_high_confidence(self):
        r = PhaseResult(phase="act", confidence=0.8)
        self.assertFalse(self.router.should_retry(r))

    def test_should_skip_optional(self):
        r = PhaseResult(phase="plan", confidence=0.95)
        self.assertTrue(self.router.should_skip_optional(r, "critique"))
        self.assertFalse(self.router.should_skip_optional(r, "act"))

    def test_cycle_confidence_empty(self):
        self.assertEqual(self.router.get_cycle_confidence(), 0.5)

    def test_cycle_confidence_with_history(self):
        self.router.record(PhaseResult(phase="plan", confidence=0.8))
        self.router.record(PhaseResult(phase="act", confidence=0.6))
        self.assertAlmostEqual(self.router.get_cycle_confidence(), 0.7)

    def test_reset(self):
        self.router.record(PhaseResult(phase="plan", confidence=0.8))
        self.router.reset()
        self.assertEqual(len(self.router.phase_history), 0)

    def test_custom_thresholds(self):
        router = ConfidenceRouter(thresholds={
            "replan_below": 0.5,
            "escalate_below": 0.3,
            "retry_below": 0.6,
            "skip_above": 0.95,
        })
        r = PhaseResult(phase="plan", confidence=0.4)
        self.assertTrue(router.should_replan(r))


if __name__ == "__main__":
    unittest.main()
