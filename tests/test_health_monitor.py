"""Unit tests for core/health_monitor.py."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.health_monitor import CheckResult, HealthMonitor, SystemHealthProbe


class _StaticSkill:
    def __init__(self, result):
        self._result = result

    def run(self, _input_data):
        return dict(self._result)


class TestHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.mock_skills = {}
        self.mock_queue = MagicMock()
        self.mock_store = MagicMock()
        self.monitor = HealthMonitor(
            skills=self.mock_skills,
            goal_queue=self.mock_queue,
            memory_store=self.mock_store,
            project_root=Path(".")
        )

    def test_instantiation(self):
        self.assertIsNotNone(self.monitor)

    def test_run_scan_returns_dict(self):
        # We need to mock _run_scan to avoid running real skills
        with patch.object(self.monitor, "_run_scan") as mock_run:
            mock_run.return_value = {"status": "ok", "metrics": {}}
            result = self.monitor.run_scan()
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)

    def test_on_cycle_complete_triggers_scan(self):
        # Trigger every 1 cycle for testing
        self.monitor.TRIGGER_EVERY_N = 1
        with patch.object(self.monitor, "run_scan") as mock_scan:
            self.monitor.on_cycle_complete({"goal": "test"})
            assert mock_scan.called

    def test_run_scan_dispatches_health_skills_and_generates_goal(self):
        queue = MagicMock()
        store = MagicMock()
        store.query.return_value = []
        monitor = HealthMonitor(
            skills={"complexity_scorer": _StaticSkill({"high_risk_count": 20})},
            goal_queue=queue,
            memory_store=store,
            project_root=Path("."),
        )

        result = monitor.run_scan()

        queue.add.assert_called_once()
        self.assertEqual(result["skills_ran"], ["complexity_scorer"])
        self.assertEqual(result["breaches"][0]["skill"], "complexity_scorer")
        store.put.assert_called_once()

    def test_system_health_probe_defaults_to_runtime_agnostic_checks(self):
        with patch.object(
            SystemHealthProbe,
            "check_skill_registry",
            return_value=CheckResult(name="skill_registry", ok=True, latency_ms=1.0, detail="35 skills loaded"),
        ):
            report = SystemHealthProbe().run_all()

        self.assertTrue(report.all_ok)
        self.assertEqual([check.name for check in report.checks], ["skill_registry"])

    def test_system_health_probe_includes_configured_runtime_checks(self):
        brain = MagicMock()
        brain.db.execute.return_value.fetchone.return_value = 1
        brain.SCHEMA_VERSION = 3
        model = MagicMock()
        model.respond = MagicMock()
        model.provider = "test"
        queue = MagicMock()
        queue.queue = ["goal-1"]
        memory_controller = MagicMock()
        memory_controller.persistent_store = object()

        with patch.object(
            SystemHealthProbe,
            "check_skill_registry",
            return_value=CheckResult(name="skill_registry", ok=True, latency_ms=1.0, detail="35 skills loaded"),
        ):
            report = SystemHealthProbe(
                brain=brain,
                model=model,
                goal_queue=queue,
                memory_controller=memory_controller,
            ).run_all()

        self.assertTrue(report.all_ok)
        self.assertEqual(
            [check.name for check in report.checks],
            ["brain_db", "memory_controller", "model_adapter", "skill_registry", "goal_queue"],
        )

if __name__ == "__main__":
    unittest.main()
