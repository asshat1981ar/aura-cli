"""Tests for orchestrator failure routing and act-loop retry/replan/stash."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from core.orchestrator import LoopOrchestrator
from memory.store import MemoryStore


def _make_orchestrator(tmpdir="/tmp"):
    agents = {}
    memory = MagicMock(spec=MemoryStore)
    memory.read_log.return_value = []
    return LoopOrchestrator(agents=agents, memory_store=memory,
                            project_root=Path(tmpdir))


# ── _route_failure ─────────────────────────────────────────────────────────────

class TestRouteFailure(unittest.TestCase):
    def setUp(self):
        self.orc = _make_orchestrator()

    def test_defaults_to_act(self):
        result = self.orc._route_failure({"failures": ["assertion error"]})
        self.assertEqual(result, "act")

    def test_structural_signals_return_plan(self):
        for signal in ("architecture", "circular", "api_breaking", "design"):
            with self.subTest(signal=signal):
                result = self.orc._route_failure({"failures": [f"detected {signal} issue"]})
                self.assertEqual(result, "plan")

    def test_external_signals_return_skip(self):
        for signal in ("no module", "import error", "network", "permission"):
            with self.subTest(signal=signal):
                result = self.orc._route_failure({"failures": [f"fatal: {signal}"]})
                self.assertEqual(result, "skip")

    def test_logs_field_also_inspected(self):
        result = self.orc._route_failure({
            "failures": [],
            "logs": "ModuleNotFoundError: No module named 'requests'"
        })
        self.assertEqual(result, "skip")

    def test_empty_verification_returns_act(self):
        result = self.orc._route_failure({})
        self.assertEqual(result, "act")


# ── _restore_applied_changes ──────────────────────────────────────────────────

class TestRestoreAppliedChanges(unittest.TestCase):
    def setUp(self):
        self.orc = _make_orchestrator()

    def test_restores_existing_file_content_and_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file_path = root / "script.sh"
            file_path.write_text("echo old\n", encoding="utf-8")
            file_path.chmod(0o755)

            snapshot = {
                "file": "script.sh",
                "target": str(file_path),
                "existed": True,
                "content": "echo old\n",
                "mode": 0o755,
            }

            file_path.write_text("echo new\n", encoding="utf-8")
            file_path.chmod(0o644)

            self.orc._restore_applied_changes([snapshot])

            self.assertEqual(file_path.read_text(encoding="utf-8"), "echo old\n")
            self.assertEqual(file_path.stat().st_mode & 0o777, 0o755)

    def test_removes_new_file_created_by_loop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file_path = root / "tests" / "test_generated.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("assert True\n", encoding="utf-8")

            snapshot = {
                "file": "tests/test_generated.py",
                "target": str(file_path),
                "existed": False,
                "content": None,
                "mode": None,
            }

            self.orc._restore_applied_changes([snapshot])

            self.assertFalse(file_path.exists())


# ── _run_act_loop retry backoff ───────────────────────────────────────────────

class TestActLoopRetryBackoff(unittest.TestCase):
    def _make_pipeline_cfg(self, max_attempts: int):
        cfg = MagicMock()
        cfg.max_act_attempts = max_attempts
        cfg.phases = ["act", "verify"]
        cfg.skill_set = []
        return cfg

    def _make_agents_with_verify_fail(self):
        """Agents where act returns a valid change_set but verify always fails."""
        act_agent = MagicMock()
        act_agent.run.return_value = {
            "changes": [{"file_path": "f.py", "new_code": "x = 1", "old_code": "", "overwrite_file": True}]
        }
        verify_agent = MagicMock()
        verify_agent.run.return_value = {
            "status": "fail",
            "failures": ["assertion error"],
            "passed": [],
            "logs": "",
        }
        return {"act": act_agent, "verify": verify_agent}

    def test_backoff_called_between_retries(self):
        orc = _make_orchestrator()
        orc.agents = self._make_agents_with_verify_fail()
        cfg = self._make_pipeline_cfg(max_attempts=3)

        with patch("time.sleep") as mock_sleep, \
             patch("core.orchestrator.validate_phase_output", return_value=[]), \
             patch.object(orc, "_run_phase") as mock_phase, \
             patch.object(orc, "_run_sandbox_loop", return_value=({}, {}, 0)):
            mock_phase.side_effect = [
                # Alternating act / verify outputs
                {"changes": []},
                {"status": "fail", "failures": ["err"], "passed": [], "logs": ""},
                {"changes": []},
                {"status": "fail", "failures": ["err"], "passed": [], "logs": ""},
                {"changes": []},
                {"status": "fail", "failures": ["err"], "passed": [], "logs": ""},
            ]
            orc._run_act_loop(
                goal="test", plan={}, task_bundle={}, pipeline_cfg=cfg,
                cycle_id="c1", phase_outputs={}, dry_run=True,
                plan_attempt=0, max_plan_retries=1, skill_context={},
            )
        # sleep should have been called for attempts 2 and 3
        self.assertGreaterEqual(mock_sleep.call_count, 1)

    def test_replan_when_route_returns_plan(self):
        orc = _make_orchestrator()
        cfg = self._make_pipeline_cfg(max_attempts=2)

        with patch("time.sleep"), \
             patch("core.orchestrator.validate_phase_output", return_value=[]), \
             patch.object(orc, "_run_phase") as mock_phase, \
             patch.object(orc, "_run_sandbox_loop", return_value=({}, {}, 0)), \
             patch.object(orc, "_route_failure", return_value="plan"):
            mock_phase.side_effect = [
                {"changes": []},
                {"status": "fail", "failures": ["design error"], "passed": [], "logs": ""},
            ]
            _, replan, _ = orc._run_act_loop(
                goal="test", plan={}, task_bundle={}, pipeline_cfg=cfg,
                cycle_id="c1", phase_outputs={}, dry_run=True,
                plan_attempt=0, max_plan_retries=1, skill_context={},
            )
        self.assertTrue(replan)
