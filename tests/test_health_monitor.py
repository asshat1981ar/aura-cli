"""Unit tests for core/health_monitor.py."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.health_monitor import HealthMonitor

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

if __name__ == "__main__":
    unittest.main()
