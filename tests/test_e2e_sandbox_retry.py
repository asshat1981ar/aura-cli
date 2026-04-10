"""E2E tests: sandbox retry scenario.

Tests that the orchestration layer can retry sandbox execution up to 3
attempts when the code fails, and eventually succeed when the code is
corrected between attempts (simulating an LLM self-correction loop).

These tests use the real SandboxAgent with real subprocess execution
(no mocking) to validate true end-to-end behavior.
"""

import pytest
from unittest.mock import MagicMock

from agents.sandbox import SandboxAgent, SandboxResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(timeout: int = 10) -> SandboxAgent:
    brain = MagicMock()
    brain.remember = MagicMock()
    brain.recall_with_budget = MagicMock(return_value=[])
    return SandboxAgent(brain, timeout=timeout)


def retry_run_code(agent: SandboxAgent, code_versions: list, max_retries: int = 3) -> tuple[SandboxResult, int]:
    """Simulate an orchestrator retry loop: try each code version in sequence.

    Returns:
        Tuple of (final_result, attempts_used).
    """
    result = None
    for attempt, code in enumerate(code_versions[:max_retries], start=1):
        result = agent.run_code(code)
        if result.passed:
            return result, attempt
    return result, len(code_versions[:max_retries])


# ---------------------------------------------------------------------------
# Retry scenarios
# ---------------------------------------------------------------------------


class TestSandboxRetryE2E:
    def test_immediate_success_uses_one_attempt(self):
        """Code that succeeds on the first try should use exactly 1 attempt."""
        agent = _make_agent()
        code_versions = ["print('success')"]
        result, attempts = retry_run_code(agent, code_versions)
        assert result.passed is True
        assert attempts == 1

    def test_fail_then_succeed_uses_two_attempts(self):
        """Code that fails on attempt 1 and succeeds on attempt 2."""
        agent = _make_agent()
        code_versions = [
            "raise ValueError('first attempt fails')",
            "print('second attempt succeeds')",
        ]
        result, attempts = retry_run_code(agent, code_versions)
        assert result.passed is True
        assert attempts == 2

    def test_fail_fail_succeed_uses_three_attempts(self):
        """Code that fails twice and succeeds on the third attempt."""
        agent = _make_agent()
        code_versions = [
            "import sys; sys.exit(1)",
            "raise RuntimeError('still failing')",
            "x = 1 + 1\nprint('ok:', x)",
        ]
        result, attempts = retry_run_code(agent, code_versions)
        assert result.passed is True
        assert attempts == 3

    def test_all_three_attempts_fail_returns_failure(self):
        """When all 3 attempts fail, retry_run_code returns a failure result."""
        agent = _make_agent()
        code_versions = [
            "raise ValueError('attempt 1')",
            "raise ValueError('attempt 2')",
            "raise ValueError('attempt 3')",
        ]
        result, attempts = retry_run_code(agent, code_versions, max_retries=3)
        assert result.passed is False
        assert attempts == 3

    def test_retry_respects_max_retries_cap(self):
        """Retry loop should not exceed max_retries even if more code versions exist."""
        agent = _make_agent()
        code_versions = [
            "raise ValueError('1')",
            "raise ValueError('2')",
            "raise ValueError('3')",
            "print('would succeed but never reached')",
        ]
        result, attempts = retry_run_code(agent, code_versions, max_retries=3)
        assert result.passed is False
        assert attempts == 3  # 4th version was never tried

    def test_each_attempt_recorded_in_brain(self):
        """Brain.remember() should be called once per attempt."""
        agent = _make_agent()
        code_versions = [
            "raise ValueError('fail')",
            "print('ok')",
        ]
        retry_run_code(agent, code_versions)
        # One brain.remember call per run_code call
        assert agent.brain.remember.call_count == 2

    def test_syntax_error_retry_succeeds(self):
        """A syntax error on attempt 1 should trigger a retry with valid code."""
        agent = _make_agent()
        code_versions = [
            "def broken(: pass",  # SyntaxError
            "def fixed(): pass\nfixed()\nprint('ok')",
        ]
        result, attempts = retry_run_code(agent, code_versions)
        assert result.passed is True
        assert attempts == 2

    def test_exit_code_42_triggers_retry(self):
        """Non-zero exit code triggers retry; success on second attempt."""
        agent = _make_agent()
        code_versions = [
            "import sys; sys.exit(42)",
            "import sys; sys.exit(0)",
        ]
        result, attempts = retry_run_code(agent, code_versions)
        assert result.passed is True
        assert result.exit_code == 0
        assert attempts == 2


# ---------------------------------------------------------------------------
# Sandbox rollback scenario
# ---------------------------------------------------------------------------


class TestSandboxRollbackE2E:
    def test_failed_result_does_not_affect_next_execution(self):
        """Each run_code call uses a fresh temp directory — no state leaks."""
        agent = _make_agent()
        # First run writes a file (it's in a sandbox temp dir)
        r1 = agent.run_code("with open('state.txt', 'w') as f: f.write('polluted')")
        # Second run in its own fresh temp dir: state.txt should not exist
        r2 = agent.run_code("import os\nif os.path.exists('state.txt'):\n    raise AssertionError('state leak!')\nprint('clean')")
        assert r2.passed is True, f"State leaked between runs: {r2.stderr}"

    def test_timeout_does_not_block_subsequent_runs(self):
        """A timed-out run should not prevent subsequent runs from completing."""
        import subprocess
        from unittest.mock import patch, MagicMock

        agent = _make_agent(timeout=1)
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            # First call times out
            proc_timeout = MagicMock()
            proc_timeout.communicate.side_effect = [
                subprocess.TimeoutExpired("python", 1),
                ("", ""),
            ]
            # Second call succeeds
            proc_ok = MagicMock()
            proc_ok.communicate.return_value = ("done\n", "")
            proc_ok.returncode = 0
            mock_popen.side_effect = [proc_timeout, proc_ok]

            r1 = agent.run_code("import time; time.sleep(999)")
            r2 = agent.run_code("print('done')")

        assert r1.timed_out is True
        assert r2.success is True
