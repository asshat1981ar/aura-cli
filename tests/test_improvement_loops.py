"""Tests for DeepReflectionLoop, HealthMonitor, WeaknessRemediator."""
import unittest
from unittest.mock import MagicMock, call, patch

from core.reflection_loop import DeepReflectionLoop
from core.health_monitor import HealthMonitor
from core.weakness_remediator import WeaknessRemediator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cycle_entry(phase_status="pass", skill_name="linter", skill_ok=True):
    """Build a minimal cycle entry dict compatible with reflection loop."""
    return {
        "goal_type": "refactor",
        "phase_outputs": {
            "plan": {"status": "pass"},
            "act": {"status": "pass"},
            "verification": {"status": phase_status},
            "skill_context": {
                skill_name: {"findings": ["thing"]} if skill_ok else {"error": "boom"},
            },
        },
    }


# ── DeepReflectionLoop ────────────────────────────────────────────────────────

class TestDeepReflectionLoopTrigger(unittest.TestCase):
    def _make_loop(self, entries_count: int) -> DeepReflectionLoop:
        memory = MagicMock()
        memory.read_log.return_value = [_make_cycle_entry() for _ in range(entries_count)]
        brain = MagicMock()
        brain.recall_weaknesses.return_value = []
        return DeepReflectionLoop(memory, brain)

    def test_does_not_trigger_before_n(self):
        loop = self._make_loop(10)
        for _ in range(loop.TRIGGER_EVERY_N - 1):
            loop.on_cycle_complete({})
        loop.memory.read_log.assert_not_called()

    def test_triggers_on_nth_cycle(self):
        loop = self._make_loop(10)
        for _ in range(loop.TRIGGER_EVERY_N):
            loop.on_cycle_complete({})
        loop.memory.read_log.assert_called_once()

    def test_triggers_again_at_2n(self):
        loop = self._make_loop(10)
        for _ in range(loop.TRIGGER_EVERY_N * 2):
            loop.on_cycle_complete({})
        self.assertEqual(loop.memory.read_log.call_count, 2)


class TestDeepReflectionLoopInsights(unittest.TestCase):
    def _make_loop_with_entries(self, entries):
        memory = MagicMock()
        memory.read_log.return_value = entries
        brain = MagicMock()
        brain.recall_weaknesses.return_value = []
        loop = DeepReflectionLoop(memory, brain)
        return loop

    def test_skipped_when_insufficient_history(self):
        loop = self._make_loop_with_entries([_make_cycle_entry()])
        result = loop.run()
        self.assertTrue(result.get("skipped"))

    def test_report_contains_phase_stats(self):
        entries = [_make_cycle_entry() for _ in range(5)]
        loop = self._make_loop_with_entries(entries)
        result = loop.run()
        self.assertIn("phase_stats", result)

    def test_phase_failure_triggers_weakness(self):
        # All cycles fail 'verification' to exceed the HIGH threshold
        entries = [_make_cycle_entry(phase_status="fail") for _ in range(10)]
        loop = self._make_loop_with_entries(entries)
        loop.run()
        # Brain.add_weakness should be called at least once
        loop.brain.add_weakness.assert_called()

    def test_never_raises_on_exception(self):
        memory = MagicMock()
        memory.read_log.side_effect = RuntimeError("boom")
        loop = DeepReflectionLoop(memory, MagicMock())
        result = loop.run()
        self.assertIn("error", result)


# ── HealthMonitor ─────────────────────────────────────────────────────────────

class TestHealthMonitorTrigger(unittest.TestCase):
    def _make_monitor(self) -> HealthMonitor:
        skills = {}
        queue = MagicMock()
        memory = MagicMock()
        memory.query.return_value = []
        return HealthMonitor(skills, queue, memory, project_root="/tmp")

    def test_does_not_trigger_before_n(self):
        m = self._make_monitor()
        for _ in range(m.TRIGGER_EVERY_N - 1):
            m.on_cycle_complete({})
        m.memory.query.assert_not_called()

    def test_triggers_on_nth_cycle(self):
        m = self._make_monitor()
        with patch.object(m, "run_scan", wraps=m.run_scan) as mock_scan:
            for _ in range(m.TRIGGER_EVERY_N):
                m.on_cycle_complete({})
        mock_scan.assert_called_once()


class TestHealthMonitorScan(unittest.TestCase):
    def test_skipped_when_no_skills_available(self):
        m = HealthMonitor({}, MagicMock(), MagicMock())
        result = m.run_scan()
        self.assertTrue(result.get("skipped"))

    def test_never_raises_on_skill_exception(self):
        with patch("core.skill_dispatcher.dispatch_skills", side_effect=RuntimeError("boom")):
            skill_stub = MagicMock()
            skills = {"complexity_scorer": skill_stub}
            m = HealthMonitor(skills, MagicMock(), MagicMock())
            result = m.run_scan()
        self.assertIn("error", result)

    def test_queues_goal_on_threshold_breach(self):
        queue = MagicMock()
        memory = MagicMock()
        memory.query.return_value = []  # no previous snapshot

        skill_results = {
            "complexity_scorer": {"high_risk_count": 20},  # exceeds max=10
        }
        with patch("core.skill_dispatcher.dispatch_skills", return_value=skill_results):
            skills = {"complexity_scorer": MagicMock()}
            m = HealthMonitor(skills, queue, memory)
            m.run_scan()

        queue.add.assert_called_once()
        goal_text = queue.add.call_args[0][0]
        self.assertIn("20", goal_text)


# ── WeaknessRemediator ────────────────────────────────────────────────────────

class TestWeaknessRemediatorRun(unittest.TestCase):
    def _make_brain(self, weaknesses=None):
        brain = MagicMock()
        brain.recall_weaknesses.return_value = weaknesses or []
        brain.recall_queued_weakness_hashes.return_value = []
        return brain

    def test_no_op_when_no_weaknesses(self):
        brain = self._make_brain()
        queue = MagicMock()
        result = WeaknessRemediator().run(brain, queue, limit=3)
        self.assertEqual(result["goals_generated"], 0)
        queue.add.assert_not_called()

    def test_generates_goal_for_structured_weakness(self):
        import json
        weakness = json.dumps({
            "type": "phase_failure",
            "phase": "act",
            "failure_rate": 0.80,
            "severity": "HIGH",
        })
        brain = self._make_brain([weakness])
        queue = MagicMock()
        result = WeaknessRemediator().run(brain, queue, limit=3)
        self.assertEqual(result["goals_generated"], 1)
        queue.add.assert_called_once()

    def test_respects_limit(self):
        import json
        weaknesses = [
            json.dumps({"type": "phase_failure", "phase": f"p{i}",
                        "failure_rate": 0.9, "severity": "HIGH"})
            for i in range(5)
        ]
        brain = self._make_brain(weaknesses)
        queue = MagicMock()
        result = WeaknessRemediator().run(brain, queue, limit=2)
        self.assertEqual(result["goals_generated"], 2)
        self.assertEqual(queue.add.call_count, 2)

    def test_skips_already_queued_weaknesses(self):
        import json, hashlib
        weakness = json.dumps({"type": "phase_failure", "phase": "act",
                               "failure_rate": 0.9, "severity": "HIGH"})
        w_hash = hashlib.sha256(weakness.encode()).hexdigest()[:16]
        brain = self._make_brain([weakness])
        brain.recall_queued_weakness_hashes.return_value = [w_hash]
        queue = MagicMock()
        result = WeaknessRemediator().run(brain, queue, limit=3)
        self.assertEqual(result["goals_generated"], 0)
        queue.add.assert_not_called()

    def test_never_raises_on_exception(self):
        brain = MagicMock()
        brain.recall_weaknesses.side_effect = RuntimeError("db gone")
        result = WeaknessRemediator().run(brain, MagicMock(), limit=3)
        self.assertIn("error", result)
