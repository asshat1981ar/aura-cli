"""Tests for SandboxAgent security violation logging.

Verifies that ``agents.sandbox`` emits structured ``sandbox_violation_attempt``
log entries via :func:`core.logging_utils.log_json` whenever a security
violation is detected in subprocess output or via a raised
:class:`core.sanitizer.SecurityError`.
"""
import unittest
from unittest.mock import MagicMock, patch

from agents.sandbox import SandboxAgent, SandboxResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> SandboxAgent:
    """Return a SandboxAgent backed by a lightweight mock brain."""
    brain = MagicMock()
    brain.remember = MagicMock()
    return SandboxAgent(brain, timeout=5)


def _make_proc_mock(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Return a mock ``subprocess.Popen`` instance whose communicate returns fixed data."""
    proc = MagicMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestSandboxViolationLogging(unittest.TestCase):
    """Sandbox security violation detection and structured log emission."""

    # ------------------------------------------------------------------
    # test_blocked_import_logs_violation
    # ------------------------------------------------------------------

    def test_blocked_import_logs_violation(self):
        """A subprocess stderr containing an ImportError triggers sandbox_violation_attempt."""
        agent = _make_agent()

        stderr_with_violation = (
            "Traceback (most recent call last):\n"
            "  File 'aura_exec.py', line 1, in <module>\n"
            "ImportError: import of 'subprocess' halted; restricted module not allowed\n"
        )
        mock_proc = _make_proc_mock(stdout="", stderr=stderr_with_violation, returncode=1)

        with (
            patch("agents.sandbox.log_json") as mock_log,
            patch("agents.sandbox.subprocess.Popen", return_value=mock_proc),
        ):
            result = agent.run_code("import subprocess")

        # Exactly one sandbox_violation_attempt call should have been emitted
        violation_calls = [
            c
            for c in mock_log.call_args_list
            if len(c.args) > 1 and c.args[1] == "sandbox_violation_attempt"
        ]
        self.assertGreaterEqual(
            len(violation_calls), 1, "Expected at least one sandbox_violation_attempt log"
        )

        details = violation_calls[0].kwargs.get("details", {})
        self.assertEqual(details.get("violation_type"), "blocked_import")
        self.assertEqual(details.get("agent"), "sandbox")
        self.assertFalse(result.passed, "Failed subprocess should produce a non-passing result")

    # ------------------------------------------------------------------
    # test_clean_execution_no_violation_log
    # ------------------------------------------------------------------

    def test_clean_execution_no_violation_log(self):
        """A subprocess that exits cleanly must not emit any sandbox_violation_attempt log."""
        agent = _make_agent()

        mock_proc = _make_proc_mock(stdout="Hello, World!\n", stderr="", returncode=0)

        with (
            patch("agents.sandbox.log_json") as mock_log,
            patch("agents.sandbox.subprocess.Popen", return_value=mock_proc),
        ):
            result = agent.run_code("print('Hello, World!')")

        violation_calls = [
            c
            for c in mock_log.call_args_list
            if len(c.args) > 1 and c.args[1] == "sandbox_violation_attempt"
        ]
        self.assertEqual(
            len(violation_calls),
            0,
            f"Expected no violation logs for clean run, got: {violation_calls}",
        )
        self.assertTrue(result.passed, "Clean subprocess should produce a passing result")

    # ------------------------------------------------------------------
    # test_permission_error_logs_restricted_path
    # ------------------------------------------------------------------

    def test_permission_error_logs_restricted_path(self):
        """PermissionError in subprocess stderr is logged as restricted_path_access."""
        agent = _make_agent()

        stderr_with_perm = "PermissionError: [Errno 13] Permission denied: '/etc/shadow'\n"
        mock_proc = _make_proc_mock(stdout="", stderr=stderr_with_perm, returncode=1)

        with (
            patch("agents.sandbox.log_json") as mock_log,
            patch("agents.sandbox.subprocess.Popen", return_value=mock_proc),
        ):
            agent.run_code("open('/etc/shadow')")

        violation_calls = [
            c
            for c in mock_log.call_args_list
            if len(c.args) > 1 and c.args[1] == "sandbox_violation_attempt"
        ]
        self.assertGreaterEqual(len(violation_calls), 1)
        details = violation_calls[0].kwargs.get("details", {})
        self.assertEqual(details.get("violation_type"), "restricted_path_access")

    # ------------------------------------------------------------------
    # test_security_error_on_blocked_command_logs_violation
    # ------------------------------------------------------------------

    def test_security_error_on_blocked_command_logs_violation(self):
        """SecurityError raised by sanitize_command is logged as blocked_command violation."""
        from core.sanitizer import SecurityError

        agent = _make_agent()

        with (
            patch("agents.sandbox.log_json") as mock_log,
            patch("agents.sandbox.subprocess.Popen") as mock_popen,
            patch(
                "core.sanitizer.sanitize_command",
                side_effect=SecurityError("Access denied: Command 'rm' is not in the allowlist."),
            ),
        ):
            result = agent.run_code("import os; os.system('rm -rf /')")

        # Popen should NOT have been called because sanitize_command raised first
        mock_popen.assert_not_called()

        violation_calls = [
            c
            for c in mock_log.call_args_list
            if len(c.args) > 1 and c.args[1] == "sandbox_violation_attempt"
        ]
        self.assertGreaterEqual(len(violation_calls), 1)
        details = violation_calls[0].kwargs.get("details", {})
        self.assertEqual(details.get("violation_type"), "blocked_command")
        self.assertEqual(details.get("agent"), "sandbox")
        self.assertFalse(result.passed)

    # ------------------------------------------------------------------
    # test_check_and_log_violations_directly
    # ------------------------------------------------------------------

    def test_check_and_log_violations_directly(self):
        """_check_and_log_violations emits a warning log for each matched violation."""
        agent = _make_agent()

        result = SandboxResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="BlockingIOError: Operation not permitted",
            execution_path="/tmp/test.py",
        )

        with patch("agents.sandbox.log_json") as mock_log:
            agent._check_and_log_violations(result, goal="test-goal")

        violation_calls = [
            c
            for c in mock_log.call_args_list
            if len(c.args) > 1 and c.args[1] == "sandbox_violation_attempt"
        ]
        self.assertGreaterEqual(len(violation_calls), 1)
        call_args = violation_calls[0]
        self.assertEqual(call_args.args[0].lower(), "warning")
        details = call_args.kwargs.get("details", {})
        self.assertEqual(details.get("agent"), "sandbox")
        self.assertEqual(details.get("goal"), "test-goal")


if __name__ == "__main__":
    unittest.main()
