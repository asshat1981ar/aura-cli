"""Comprehensive tests for core.reflection_loop.DeepReflectionLoop."""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, call, patch


from core.reflection_loop import DeepReflectionLoop


# ── Helpers ───────────────────────────────────────────────────────────────────


def _entry(
    *,
    goal_type: str = "refactor",
    verif_status: str = "pass",
    phases: dict | None = None,
    skill_context: dict | None = None,
) -> dict:
    """Build a minimal history entry with configurable shape."""
    po: dict = {
        "plan": {"status": "pass"},
        "act": {"status": "pass"},
        "verification": {"status": verif_status},
    }
    if phases:
        po.update(phases)
    if skill_context is not None:
        po["skill_context"] = skill_context
    return {"goal_type": goal_type, "phase_outputs": po}


def _make_loop(entries: list) -> DeepReflectionLoop:
    memory = MagicMock()
    memory.read_log.return_value = entries
    brain = MagicMock()
    return DeepReflectionLoop(memory, brain)


# ── Trigger behaviour ─────────────────────────────────────────────────────────


class TestOnCycleCompleteTrigger(unittest.TestCase):
    def test_does_not_trigger_before_n_cycles(self):
        loop = _make_loop([_entry() for _ in range(10)])
        for _ in range(loop.TRIGGER_EVERY_N - 1):
            loop.on_cycle_complete({})
        loop.memory.read_log.assert_not_called()

    def test_triggers_exactly_on_nth_cycle(self):
        loop = _make_loop([_entry() for _ in range(10)])
        for _ in range(loop.TRIGGER_EVERY_N):
            loop.on_cycle_complete({})
        loop.memory.read_log.assert_called_once()

    def test_triggers_again_at_2n(self):
        loop = _make_loop([_entry() for _ in range(10)])
        for _ in range(loop.TRIGGER_EVERY_N * 2):
            loop.on_cycle_complete({})
        self.assertEqual(loop.memory.read_log.call_count, 2)

    def test_does_not_trigger_at_2n_minus_1(self):
        loop = _make_loop([_entry() for _ in range(10)])
        for _ in range(loop.TRIGGER_EVERY_N * 2 - 1):
            loop.on_cycle_complete({})
        self.assertEqual(loop.memory.read_log.call_count, 1)


# ── Insufficient history guard ────────────────────────────────────────────────


class TestInsufficientHistory(unittest.TestCase):
    def test_skipped_when_zero_entries(self):
        loop = _make_loop([])
        result = loop.run()
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "insufficient_history")

    def test_skipped_when_below_min_cycles(self):
        loop = _make_loop([_entry()] * (DeepReflectionLoop.MIN_CYCLES - 1))
        result = loop.run()
        self.assertTrue(result.get("skipped"))

    def test_not_skipped_when_at_min_cycles(self):
        loop = _make_loop([_entry()] * DeepReflectionLoop.MIN_CYCLES)
        result = loop.run()
        self.assertNotIn("skipped", result)


# ── Phase failure rate detection ──────────────────────────────────────────────


class TestPhaseFailureRateDetection(unittest.TestCase):
    def _run_with_fail_rate(self, fail_count: int, total: int) -> dict:
        """Build entries so that 'act' fails fail_count/total times."""
        entries = []
        for i in range(total):
            status = "fail" if i < fail_count else "pass"
            entries.append(_entry(phases={"act": {"status": status}}))
        return _make_loop(entries).run()

    def test_no_insight_below_warn_threshold(self):
        # 30% failure rate is below PHASE_FAIL_RATE_WARN (40%)
        result = self._run_with_fail_rate(3, 10)
        phase_insights = [
            i for i in result.get("insights", [])
            if i["type"] == "phase_failure" and i["phase"] == "act"
        ]
        self.assertEqual(phase_insights, [])

    def test_medium_severity_at_warn_threshold(self):
        # 50% failure rate → MEDIUM (>= 40%, < 65%)
        result = self._run_with_fail_rate(5, 10)
        phase_insights = [
            i for i in result.get("insights", [])
            if i["type"] == "phase_failure" and i["phase"] == "act"
        ]
        self.assertEqual(len(phase_insights), 1)
        self.assertEqual(phase_insights[0]["severity"], "MEDIUM")

    def test_high_severity_at_high_threshold(self):
        # 70% failure rate → HIGH (>= 65%)
        result = self._run_with_fail_rate(7, 10)
        phase_insights = [
            i for i in result.get("insights", [])
            if i["type"] == "phase_failure" and i["phase"] == "act"
        ]
        self.assertEqual(len(phase_insights), 1)
        self.assertEqual(phase_insights[0]["severity"], "HIGH")

    def test_phase_insight_includes_failure_rate(self):
        result = self._run_with_fail_rate(8, 10)
        insight = next(
            (i for i in result.get("insights", []) if i["type"] == "phase_failure" and i["phase"] == "act"),
            None,
        )
        self.assertIsNotNone(insight)
        self.assertAlmostEqual(insight["failure_rate"], 0.8, places=1)

    def test_phases_with_underscore_prefix_are_skipped(self):
        entries = [
            _entry(phases={"_internal": {"status": "fail"}}) for _ in range(10)
        ]
        result = _make_loop(entries).run()
        internal_insights = [
            i for i in result.get("insights", [])
            if i.get("phase") == "_internal"
        ]
        self.assertEqual(internal_insights, [])

    def test_apply_result_phase_is_skipped(self):
        entries = [
            _entry(phases={"apply_result": {"status": "fail"}}) for _ in range(10)
        ]
        result = _make_loop(entries).run()
        apply_insights = [
            i for i in result.get("insights", [])
            if i.get("phase") == "apply_result"
        ]
        self.assertEqual(apply_insights, [])


# ── Skill low-signal detection ────────────────────────────────────────────────


class TestSkillLowSignalDetection(unittest.TestCase):
    def _entries_with_skill(self, skill_name: str, actionable_count: int, total: int) -> list:
        entries = []
        for i in range(total):
            if i < actionable_count:
                skill_result = {"findings": ["issue1"]}
            else:
                skill_result = {}  # empty → not actionable
            entries.append(_entry(skill_context={skill_name: skill_result}))
        return entries

    def test_no_insight_when_skill_run_count_below_min(self):
        # Only 2 runs < SKILL_MIN_RUNS (3)
        entries = self._entries_with_skill("linter", 0, 2)
        result = _make_loop(entries).run()
        skill_insights = [i for i in result.get("insights", []) if i["type"] == "low_value_skill"]
        self.assertEqual(skill_insights, [])

    def test_low_signal_insight_when_actionable_rate_below_threshold(self):
        # 0/5 actionable → 0% < 25% threshold
        entries = self._entries_with_skill("linter", 0, 5)
        result = _make_loop(entries).run()
        skill_insights = [
            i for i in result.get("insights", [])
            if i["type"] == "low_value_skill" and i["skill"] == "linter"
        ]
        self.assertEqual(len(skill_insights), 1)
        self.assertEqual(skill_insights[0]["severity"], "LOW")

    def test_no_insight_when_skill_produces_sufficient_signal(self):
        # 4/5 actionable → 80% >> 25% threshold
        entries = self._entries_with_skill("linter", 4, 5)
        result = _make_loop(entries).run()
        skill_insights = [
            i for i in result.get("insights", [])
            if i["type"] == "low_value_skill" and i["skill"] == "linter"
        ]
        self.assertEqual(skill_insights, [])

    def test_skill_insight_contains_runs_and_rate(self):
        entries = self._entries_with_skill("complexity_scorer", 0, 5)
        result = _make_loop(entries).run()
        insight = next(
            (i for i in result.get("insights", [])
             if i["type"] == "low_value_skill" and i["skill"] == "complexity_scorer"),
            None,
        )
        self.assertIsNotNone(insight)
        self.assertEqual(insight["runs"], 5)
        self.assertAlmostEqual(insight["actionable_rate"], 0.0)

    def test_skill_with_error_key_is_not_counted_as_actionable(self):
        # result with only "error" key should not be actionable
        entries = [
            _entry(skill_context={"linter": {"error": "timeout"}}) for _ in range(5)
        ]
        result = _make_loop(entries).run()
        skill_insights = [
            i for i in result.get("insights", [])
            if i["type"] == "low_value_skill" and i["skill"] == "linter"
        ]
        self.assertEqual(len(skill_insights), 1)


# ── Goal-type outcome tracking ────────────────────────────────────────────────


class TestGoalTypeOutcomeTracking(unittest.TestCase):
    def _entries_for_goal_type(self, goal_type: str, pass_count: int, total: int) -> list:
        entries = []
        for i in range(total):
            status = "pass" if i < pass_count else "fail"
            entries.append(_entry(goal_type=goal_type, verif_status=status))
        return entries

    def test_goal_type_stats_present_in_report(self):
        entries = self._entries_for_goal_type("feature", 4, 5)
        result = _make_loop(entries).run()
        self.assertIn("goal_type_outcomes", result)
        self.assertIn("feature", result["goal_type_outcomes"])

    def test_goal_type_struggling_insight_when_low_success_rate(self):
        # 1/5 pass → 20% < 40% threshold
        entries = self._entries_for_goal_type("bugfix", 1, 5)
        result = _make_loop(entries).run()
        insights = [
            i for i in result.get("insights", [])
            if i["type"] == "goal_type_struggling" and i["goal_type"] == "bugfix"
        ]
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0]["severity"], "HIGH")

    def test_no_insight_when_success_rate_is_adequate(self):
        # 4/5 pass → 80%
        entries = self._entries_for_goal_type("refactor", 4, 5)
        result = _make_loop(entries).run()
        insights = [
            i for i in result.get("insights", [])
            if i["type"] == "goal_type_struggling" and i["goal_type"] == "refactor"
        ]
        self.assertEqual(insights, [])

    def test_no_insight_when_fewer_than_3_samples(self):
        # Only 2 entries — minimum is 3
        entries = self._entries_for_goal_type("feature", 0, 2)
        result = _make_loop(entries).run()
        insights = [
            i for i in result.get("insights", [])
            if i["type"] == "goal_type_struggling" and i["goal_type"] == "feature"
        ]
        self.assertEqual(insights, [])

    def test_skip_status_counts_as_pass(self):
        entries = [
            _entry(goal_type="migration", verif_status="skip") for _ in range(5)
        ]
        result = _make_loop(entries).run()
        outcomes = result["goal_type_outcomes"]["migration"]
        self.assertEqual(outcomes["pass"], 5)

    def test_multiple_goal_types_tracked_independently(self):
        entries = (
            self._entries_for_goal_type("refactor", 5, 5) +
            self._entries_for_goal_type("bugfix", 0, 5)
        )
        result = _make_loop(entries).run()
        self.assertEqual(result["goal_type_outcomes"]["refactor"]["pass"], 5)
        self.assertEqual(result["goal_type_outcomes"]["bugfix"]["pass"], 0)


# ── Insight extraction ────────────────────────────────────────────────────────


class TestInsightExtraction(unittest.TestCase):
    def test_insights_written_to_brain(self):
        entries = [
            _entry(phases={"plan": {"status": "fail"}}) for _ in range(10)
        ]
        loop = _make_loop(entries)
        loop.run()
        # At minimum, add_weakness should be called (even zero times is valid
        # if no threshold is breached) — but we explicitly forced 100% failure
        # on 'plan' which exceeds the HIGH threshold.
        loop.brain.add_weakness.assert_called()

    def test_insight_written_as_json_string(self):
        entries = [
            _entry(phases={"act": {"status": "fail"}}) for _ in range(10)
        ]
        loop = _make_loop(entries)
        loop.run()
        # Each call to add_weakness should pass a valid JSON string
        for call_args in loop.brain.add_weakness.call_args_list:
            arg = call_args[0][0]
            parsed = json.loads(arg)
            self.assertIn("type", parsed)

    def test_report_stored_in_memory(self):
        entries = [_entry() for _ in range(5)]
        loop = _make_loop(entries)
        loop.run()
        loop.memory.put.assert_called_once()
        args = loop.memory.put.call_args[0]
        self.assertEqual(args[0], "reflection_reports")

    def test_report_contains_cycle_count(self):
        loop = _make_loop([_entry() for _ in range(5)])
        loop._cycle_count = 42
        result = loop.run()
        self.assertEqual(result["cycle_count"], 42)


# ── Error resilience ──────────────────────────────────────────────────────────


class TestErrorResilience(unittest.TestCase):
    def test_never_raises_on_read_log_failure(self):
        memory = MagicMock()
        memory.read_log.side_effect = RuntimeError("db gone")
        loop = DeepReflectionLoop(memory, MagicMock())
        result = loop.run()
        self.assertIn("error", result)

    def test_never_raises_on_brain_failure(self):
        entries = [_entry(phases={"act": {"status": "fail"}}) for _ in range(10)]
        loop = _make_loop(entries)
        loop.brain.add_weakness.side_effect = RuntimeError("brain exploded")
        # Should not propagate
        result = loop.run()
        self.assertIn("error", result)

    def test_returns_dict_on_all_failures(self):
        memory = MagicMock()
        memory.read_log.side_effect = Exception("unexpected")
        loop = DeepReflectionLoop(memory, MagicMock())
        result = loop.run()
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
