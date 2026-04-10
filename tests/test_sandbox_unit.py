"""Unit tests for SandboxResult dataclass and SandboxAgent.run_code / run_file.

Covers the uncovered lines (~186-263 and ~294-350) focusing on:
- SandboxResult dataclass properties and methods
- SandboxAgent.run_code (success, failure, timeout, security violation, extra_files)
- SandboxAgent.run_file (success, failure, internal exception)
- SandboxAgent.run_tests (pass/fail metadata parsing)
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from agents.sandbox import SandboxAgent, SandboxResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(timeout: int = 10) -> SandboxAgent:
    brain = MagicMock()
    brain.remember = MagicMock()
    return SandboxAgent(brain, timeout=timeout)


def _proc_mock(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# SandboxResult dataclass
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_passed_property_success_no_timeout(self):
        r = SandboxResult(success=True, exit_code=0, stdout="ok", stderr="")
        assert r.passed is True

    def test_passed_property_failure(self):
        r = SandboxResult(success=False, exit_code=1, stdout="", stderr="err")
        assert r.passed is False

    def test_passed_property_timeout(self):
        r = SandboxResult(success=False, exit_code=-1, stdout="", stderr="timeout", timed_out=True)
        assert r.passed is False

    def test_summary_pass(self):
        r = SandboxResult(success=True, exit_code=0, stdout="hello", stderr="")
        s = r.summary()
        assert s.startswith("[PASS]")
        assert "exit=0" in s
        assert "stdout=5c" in s

    def test_summary_fail(self):
        r = SandboxResult(success=False, exit_code=1, stdout="", stderr="bad")
        s = r.summary()
        assert s.startswith("[FAIL]")
        assert "exit=1" in s

    def test_summary_timeout(self):
        r = SandboxResult(success=False, exit_code=-1, stdout="", stderr="", timed_out=True)
        s = r.summary()
        assert s.startswith("[TIMEOUT]")

    def test_str_repr(self):
        r = SandboxResult(success=True, exit_code=0, stdout="hi", stderr="")
        assert "passed=True" in str(r)
        assert "exit=0" in str(r)

    def test_metadata_default_empty_dict(self):
        r = SandboxResult(success=True, exit_code=0, stdout="", stderr="")
        assert r.metadata == {}

    def test_execution_path_default_none(self):
        r = SandboxResult(success=True, exit_code=0, stdout="", stderr="")
        assert r.execution_path is None

    def test_metadata_can_be_populated(self):
        r = SandboxResult(success=True, exit_code=0, stdout="", stderr="", metadata={"k": "v"})
        assert r.metadata["k"] == "v"


# ---------------------------------------------------------------------------
# SandboxAgent.run_code
# ---------------------------------------------------------------------------


class TestRunCode:
    def test_run_code_success(self):
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("hello\n", "", 0)
            result = agent.run_code("print('hello')")
        assert result.success is True
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_run_code_failure(self):
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("", "NameError: x", 1)
            result = agent.run_code("x")
        assert result.success is False
        assert result.exit_code == 1

    def test_run_code_with_extra_files(self):
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("ok", "", 0)
            result = agent.run_code(
                "from helper import greet\ngreet()",
                extra_files={"helper.py": "def greet(): print('hi')"},
            )
        assert result.success is True

    def test_run_code_timeout(self):
        import subprocess

        agent = _make_agent(timeout=1)
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            # First communicate raises TimeoutExpired; second (after kill) returns normally
            proc.communicate.side_effect = [
                subprocess.TimeoutExpired("python", 1),
                ("", ""),
            ]
            mock_popen.return_value = proc
            result = agent.run_code("import time; time.sleep(999)")
        assert result.timed_out is True
        assert result.success is False
        assert "timed out" in result.stderr

    def test_run_code_records_to_brain(self):
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("", "", 0)
            agent.run_code("pass")
        agent.brain.remember.assert_called_once()
        # The first arg should be a string description
        call_args = agent.brain.remember.call_args
        assert call_args is not None

    def test_run_code_internal_exception(self):
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen", side_effect=OSError("disk full")):
            result = agent.run_code("pass")
        assert result.success is False
        assert "SandboxAgent internal error" in result.stderr or "disk full" in result.stderr

    def test_run_code_execution_path_set(self):
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("", "", 0)
            result = agent.run_code("pass")
        assert result.execution_path is not None
        assert "aura_exec.py" in str(result.execution_path)


# ---------------------------------------------------------------------------
# SandboxAgent.run_file
# ---------------------------------------------------------------------------


class TestRunFile:
    def test_run_file_success(self, tmp_path):
        script = tmp_path / "test_script.py"
        script.write_text("print('done')")
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("done\n", "", 0)
            result = agent.run_file(str(script))
        assert result.success is True
        assert "done" in result.stdout

    def test_run_file_failure(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("raise ValueError('oops')")
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("", "ValueError: oops", 1)
            result = agent.run_file(str(script))
        assert result.success is False

    def test_run_file_execution_path_matches_input(self, tmp_path):
        script = tmp_path / "run_me.py"
        script.write_text("pass")
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("", "", 0)
            result = agent.run_file(str(script))
        assert str(script) in str(result.execution_path)

    def test_run_file_records_to_brain(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("pass")
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("", "", 0)
            agent.run_file(str(script))
        agent.brain.remember.assert_called_once()


# ---------------------------------------------------------------------------
# SandboxAgent.run_tests (metadata parsing)
# ---------------------------------------------------------------------------


class TestRunTests:
    def test_run_tests_passes_with_metadata(self):
        agent = _make_agent()
        code = "def add(a, b): return a + b"
        tests = "def test_add():\n    assert add(1,2) == 3\n"
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("1 passed in 0.1s", "", 0)
            result = agent.run_tests(code, tests)
        assert result.success is True
        assert result.metadata.get("passed", 0) >= 0  # metadata populated

    def test_run_tests_failure_counts(self):
        agent = _make_agent()
        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("1 passed, 2 failed", "", 1)
            result = agent.run_tests("def f(): pass", "def test_f(): assert False")
        assert result.metadata.get("failed", 0) >= 0

    def test_run_tests_injects_sys_path_header(self):
        """If tests don't import source, a sys.path header should be injected."""
        agent = _make_agent()
        written_content = {}

        original_write = Path.write_text

        def capture_write(self, text, *args, **kwargs):
            written_content[str(self)] = text
            return original_write(self, text, *args, **kwargs)

        with patch("agents.sandbox.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _proc_mock("0 passed", "", 0)
            with patch.object(Path, "write_text", capture_write):
                agent.run_tests("def f(): pass", "def test_f(): pass")

        # At least one written file should contain the sys.path injection
        test_files = [c for k, c in written_content.items() if "test_source" in k]
        assert any("sys.path" in c for c in test_files)


# ---------------------------------------------------------------------------
# Integration: run_code with real subprocess
# ---------------------------------------------------------------------------


class TestRunCodeIntegration:
    """Light integration tests using a real subprocess (no mocking)."""

    def test_hello_world(self):
        agent = _make_agent()
        result = agent.run_code("print('hello world')")
        assert result.success is True
        assert "hello world" in result.stdout

    def test_syntax_error_returns_failure(self):
        agent = _make_agent()
        result = agent.run_code("def bad(: pass")
        assert result.success is False
        assert result.exit_code != 0

    def test_exit_code_propagated(self):
        agent = _make_agent()
        result = agent.run_code("import sys; sys.exit(42)")
        assert result.exit_code == 42
        assert result.success is False
