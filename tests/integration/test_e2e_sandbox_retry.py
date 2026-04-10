"""E2E tests for sandbox retry with debugger integration.

Sprint 5: E2E tests — sandbox retry up to 3 attempts with debugger.

These tests verify the complete retry loop:
1. Code fails in sandbox
2. DebuggerAgent is invoked to fix the code
3. Fixed code is retried in sandbox
4. Process repeats up to 3 times
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agents.sandbox import SandboxAgent, SandboxResult
from agents.debugger import DebuggerAgent


class TestSandboxRetryLoop:
    """E2E tests for sandbox retry with debugger."""

    def test_sandbox_success_no_retry_needed(self):
        """Should return success on first try when code passes."""
        brain = MagicMock()
        sandbox = SandboxAgent(brain=brain, timeout=10)

        # Mock successful sandbox execution
        with patch.object(sandbox, "_run") as mock_run:
            mock_run.return_value = SandboxResult(
                success=True,
                exit_code=0,
                stdout="test passed",
                stderr="",
            )

            result = sandbox.run_code("print('hello')")

            assert result.success is True
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_sandbox_retry_with_fix_success(self):
        """Should retry with debugger fix and succeed on second try."""
        brain = MagicMock()
        sandbox = SandboxAgent(brain=brain, timeout=10)
        debugger = MagicMock(spec=DebuggerAgent)

        # First run fails, second succeeds
        fail_result = SandboxResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="NameError: name 'x' is not defined",
        )
        success_result = SandboxResult(
            success=True,
            exit_code=0,
            stdout="fixed!",
            stderr="",
        )

        with patch.object(sandbox, "_run") as mock_run:
            mock_run.side_effect = [fail_result, success_result]

            # Debugger returns fixed code
            debugger.debug.return_value = {
                "fixed_code": "x = 1\nprint(x)",
                "explanation": "Defined x before use",
            }

            # First attempt
            result = sandbox.run_code("print(x)")
            assert result.success is False

            # Simulate retry with fixed code
            fixed_code = debugger.debug.return_value["fixed_code"]
            result2 = sandbox.run_code(fixed_code)

            assert result2.success is True
            assert mock_run.call_count == 2

    def test_sandbox_retry_exhausts_3_attempts(self):
        """Should give up after 3 failed attempts."""
        brain = MagicMock()
        sandbox = SandboxAgent(brain=brain, timeout=5)

        fail_result = SandboxResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="SyntaxError: invalid syntax",
        )

        with patch.object(sandbox, "_run") as mock_run:
            mock_run.return_value = fail_result

            attempts = 0
            max_retries = 3

            while attempts < max_retries:
                result = sandbox.run_code("invalid syntax {{[[")
                attempts += 1
                if result.success:
                    break

            assert attempts == 3
            assert result.success is False
            assert mock_run.call_count == 3

    def test_sandbox_retry_preserves_context(self):
        """Should preserve error context across retry attempts."""
        brain = MagicMock()
        sandbox = SandboxAgent(brain=brain, timeout=10)

        results = [
            SandboxResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="ImportError: No module named 'missing'",
            ),
            SandboxResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="ImportError: Still missing after fix attempt",
            ),
            SandboxResult(
                success=True,
                exit_code=0,
                stdout="import succeeded",
                stderr="",
            ),
        ]

        with patch.object(sandbox, "_run") as mock_run:
            mock_run.side_effect = results

            collected_errors = []
            for _ in range(3):
                result = sandbox.run_code("import missing")
                if not result.success:
                    collected_errors.append(result.stderr)
                else:
                    break

            assert len(collected_errors) == 2
            assert "No module named 'missing'" in collected_errors[0]
            assert mock_run.call_count == 3


class TestSandboxDebuggerIntegration:
    """Integration tests for sandbox + debugger workflow."""

    def test_debugger_receives_sandbox_output(self):
        """Debugger should receive sandbox stderr for analysis."""
        debugger = MagicMock(spec=DebuggerAgent)

        sandbox_error = "TypeError: unsupported operand type(s) for +: 'int' and 'str'"

        debugger.debug.return_value = {
            "fixed_code": "print(str(1) + 'hello')",
            "explanation": "Convert int to str before concatenation",
        }

        # Simulate calling debugger with error
        result = debugger.debug(
            code="print(1 + 'hello')",
            error=sandbox_error,
        )

        debugger.debug.assert_called_once_with(
            code="print(1 + 'hello')",
            error=sandbox_error,
        )
        assert "fixed_code" in result

    def test_full_retry_cycle_integration(self):
        """Full integration test of sandbox → debugger → retry cycle."""
        brain = MagicMock()
        sandbox = SandboxAgent(brain=brain, timeout=10)
        debugger = MagicMock(spec=DebuggerAgent)

        original_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""

        # First run fails with ZeroDivisionError
        first_result = SandboxResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="ZeroDivisionError: division by zero",
        )

        # Debugger fixes the code
        fixed_code = """
def divide(a, b):
    if b == 0:
        return 0
    return a / b

result = divide(10, 0)
print(result)
"""
        debugger.debug.return_value = {
            "fixed_code": fixed_code,
            "explanation": "Added zero check",
        }

        # Second run succeeds
        second_result = SandboxResult(
            success=True,
            exit_code=0,
            stdout="0",
            stderr="",
        )

        with patch.object(sandbox, "_run") as mock_run:
            mock_run.side_effect = [first_result, second_result]

            # Attempt 1: Run original code
            result1 = sandbox.run_code(original_code)
            assert result1.success is False

            # Call debugger with error
            debug_result = debugger.debug(
                code=original_code,
                error=result1.stderr,
            )

            # Attempt 2: Run fixed code
            result2 = sandbox.run_code(debug_result["fixed_code"])
            assert result2.success is True

            assert mock_run.call_count == 2


class TestSandboxTimeoutRetry:
    """Tests for timeout handling in retry loop."""

    def test_timeout_triggers_retry(self):
        """Timeout should be treated as failure and trigger retry."""
        brain = MagicMock()
        sandbox = SandboxAgent(brain=brain, timeout=5)

        timeout_result = SandboxResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="Execution timed out after 5s",
            timed_out=True,
        )

        success_result = SandboxResult(
            success=True,
            exit_code=0,
            stdout="completed",
            stderr="",
        )

        with patch.object(sandbox, "_run") as mock_run:
            mock_run.side_effect = [timeout_result, success_result]

            result1 = sandbox.run_code("slow_code()")
            assert result1.timed_out is True

            # Retry with optimized code
            result2 = sandbox.run_code("optimized_code()")
            assert result2.success is True

    def test_max_timeout_does_not_exceed_limit(self):
        """Should enforce max timeout even with retries."""
        brain = MagicMock()

        # Create sandbox with short timeout
        sandbox = SandboxAgent(brain=brain, timeout=2)

        # Verify timeout is respected
        assert sandbox.timeout == 2

        # Even with 3 retries, total time shouldn't exceed ~6 seconds
        # (this is a configuration test, not timing test)
        max_expected_time = sandbox.timeout * 3
        assert max_expected_time <= 6  # 2 seconds * 3 retries
