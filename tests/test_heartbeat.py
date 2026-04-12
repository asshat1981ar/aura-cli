"""Tests for core.heartbeat — periodic maintenance manager."""

import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from core.heartbeat import HeartbeatCheck, HeartbeatManager, HeartbeatResult


class TestHeartbeatRegister(unittest.TestCase):
    """Test check registration."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state_path = Path(self.tmp.name)
        self.state_path.unlink(missing_ok=True)

    def tearDown(self):
        self.state_path.unlink(missing_ok=True)

    def test_register_check(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        mgr.register("test_check", lambda: "ok", interval_seconds=3600)
        status = mgr.status()
        self.assertIn("test_check", status)
        self.assertTrue(status["test_check"]["enabled"])
        self.assertEqual(status["test_check"]["interval_seconds"], 3600)

    def test_register_disabled_check(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        mgr.register("disabled_check", lambda: "ok", interval_seconds=3600, enabled=False)
        status = mgr.status()
        self.assertFalse(status["disabled_check"]["enabled"])


class TestHeartbeatTick(unittest.TestCase):
    """Test the tick() method."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state_path = Path(self.tmp.name)
        self.state_path.unlink(missing_ok=True)

    def tearDown(self):
        self.state_path.unlink(missing_ok=True)

    def test_tick_runs_due_checks(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        callback = MagicMock(return_value="ok")
        mgr.register("check_a", callback, interval_seconds=3600)
        # Force it to be due by using force_run
        result = mgr.force_run("check_a")
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        callback.assert_called()

    def test_tick_skips_not_due_checks(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        callback = MagicMock(return_value="ok")
        mgr.register("check_b", callback, interval_seconds=3600)
        # Force run so last_run is set to now
        mgr.force_run("check_b")
        callback.reset_mock()

        # Tick should NOT run it again (it's not due yet)
        results = mgr.tick()
        callback.assert_not_called()
        self.assertEqual(len(results), 0)

    def test_tick_max_per_tick(self):
        mgr = HeartbeatManager(state_path=self.state_path, max_checks_per_tick=2)
        callbacks = {}
        for name in ["a", "b", "c"]:
            cb = MagicMock(return_value=f"{name}_ok")
            callbacks[name] = cb
            mgr.register(name, cb, interval_seconds=3600)
            # All are due (last_run=0)

        results = mgr.tick()
        # Should only run 2 out of 3
        self.assertEqual(len(results), 2)

    def test_tick_disabled_checks_skipped(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        callback = MagicMock(return_value="ok")
        mgr.register("disabled", callback, interval_seconds=3600, enabled=False)
        results = mgr.tick()
        callback.assert_not_called()
        self.assertEqual(len(results), 0)


class TestHeartbeatErrorHandling(unittest.TestCase):
    """Test error handling in check execution."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state_path = Path(self.tmp.name)
        self.state_path.unlink(missing_ok=True)

    def tearDown(self):
        self.state_path.unlink(missing_ok=True)

    def test_check_failure_captured(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        mgr.register("failing", lambda: 1 / 0, interval_seconds=3600)
        result = mgr.force_run("failing")
        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertIn("division by zero", result.error)

    def test_force_run_unknown_check(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        result = mgr.force_run("nonexistent")
        self.assertIsNone(result)


class TestHeartbeatForceRun(unittest.TestCase):
    """Test force_run bypass."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state_path = Path(self.tmp.name)
        self.state_path.unlink(missing_ok=True)

    def tearDown(self):
        self.state_path.unlink(missing_ok=True)

    def test_force_run_executes_immediately(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        callback = MagicMock(return_value="forced_result")
        mgr.register("forceable", callback, interval_seconds=3600)
        # Force-run even though it was just registered
        mgr.force_run("forceable")
        callback.reset_mock()
        # It should not be due via tick
        results = mgr.tick()
        callback.assert_not_called()
        # But force_run always works
        result = mgr.force_run("forceable")
        self.assertTrue(result.success)
        self.assertEqual(result.detail, "forced_result")
        callback.assert_called_once()


class TestHeartbeatStatePersistence(unittest.TestCase):
    """Test state persistence across instances."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state_path = Path(self.tmp.name)
        self.state_path.unlink(missing_ok=True)

    def tearDown(self):
        self.state_path.unlink(missing_ok=True)

    def test_state_persisted_and_reloaded(self):
        # First manager — run a check
        mgr1 = HeartbeatManager(state_path=self.state_path)
        mgr1.register("persist_check", lambda: "ok", interval_seconds=3600)
        mgr1.force_run("persist_check")

        # Verify state file was created
        self.assertTrue(self.state_path.exists())
        state_data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertIn("persist_check", state_data)

        # Second manager — should load state and know the check ran
        mgr2 = HeartbeatManager(state_path=self.state_path)
        mgr2.register("persist_check", lambda: "ok", interval_seconds=3600)
        # Check should NOT be due since it was just run
        self.assertFalse(mgr2.is_due("persist_check"))

    def test_is_due_without_registration(self):
        mgr = HeartbeatManager(state_path=self.state_path)
        self.assertFalse(mgr.is_due("unknown_check"))


if __name__ == "__main__":
    unittest.main()
