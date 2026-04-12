"""Unit tests for agents.verifier.VerifierAgent."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.verifier import VerifierAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed(returncode: int = 0, stdout: str = "1 passed", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["python3", "-m", "pytest", "-q"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# dry_run shortcut
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_skip_without_running_subprocess(self):
        agent = VerifierAgent(timeout=5)
        result = agent.run({"dry_run": True})
        assert result["status"] == "skip"
        assert result["failures"] == []


# ---------------------------------------------------------------------------
# Happy path — pass
# ---------------------------------------------------------------------------


class TestVerifierPass:
    def test_explicit_targeted_command_is_used_verbatim(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text("def test_ok(): pass\n")

        agent = VerifierAgent(timeout=5)
        with patch("agents.verifier.subprocess.run", return_value=_completed(0)) as mock_run:
            result = agent.run(
                {
                    "project_root": str(tmp_path),
                    "tests": ["python3 -m pytest -q tests/test_foo.py"],
                }
            )

        assert result["status"] == "pass"
        assert result["failures"] == []
        assert mock_run.call_args.args[0] == ["python3", "-m", "pytest", "-q", "tests/test_foo.py"]

    def test_incremental_tests_chosen_when_change_set_provided(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_bar.py").write_text("def test_ok(): pass\n")

        agent = VerifierAgent(timeout=5)
        with patch("agents.verifier.subprocess.run", return_value=_completed(0)) as mock_run:
            result = agent.run(
                {
                    "project_root": str(tmp_path),
                    "change_set": {"changes": [{"file_path": "bar.py"}]},
                    "tests": ["python3 -m pytest -q"],  # repo-wide, should be overridden
                }
            )

        assert result["status"] == "pass"
        called_cmd = mock_run.call_args.args[0]
        assert "tests/test_bar.py" in called_cmd

    def test_aura_skip_chdir_set_in_env(self, tmp_path):
        agent = VerifierAgent(timeout=5)
        with patch("agents.verifier.subprocess.run", return_value=_completed(0)) as mock_run, patch.dict("os.environ", {}, clear=True):
            agent.run({"project_root": str(tmp_path)})
        env_used = mock_run.call_args.kwargs["env"]
        assert env_used.get("AURA_SKIP_CHDIR") == "1"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestVerifierFail:
    def test_nonzero_returncode_gives_fail_status(self, tmp_path):
        agent = VerifierAgent(timeout=5)
        with patch("agents.verifier.subprocess.run", return_value=_completed(1, stdout="", stderr="FAILED")):
            result = agent.run({"project_root": str(tmp_path)})

        assert result["status"] == "fail"
        assert "pytest_failed" in result["failures"]

    def test_logs_combine_stdout_and_stderr(self, tmp_path):
        agent = VerifierAgent(timeout=5)
        with patch("agents.verifier.subprocess.run", return_value=_completed(1, stdout="out", stderr="err")):
            result = agent.run({"project_root": str(tmp_path)})

        assert "out" in result["logs"]
        assert "err" in result["logs"]


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestVerifierTimeout:
    def test_timeout_returns_fail_status(self, tmp_path):
        agent = VerifierAgent(timeout=1)
        with patch(
            "agents.verifier.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=1),
        ):
            result = agent.run({"project_root": str(tmp_path)})

        assert result["status"] == "fail"
        assert "pytest_timeout" in result["failures"]

    def test_timeout_logs_contain_command(self, tmp_path):
        agent = VerifierAgent(timeout=1)
        with patch(
            "agents.verifier.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=1),
        ):
            result = agent.run({"project_root": str(tmp_path), "tests": ["pytest -q some_test.py"]})

        assert "timeout" in result["logs"].lower()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_available_test_files_empty_when_no_tests_dir(self, tmp_path):
        agent = VerifierAgent()
        files = agent._available_test_files(tmp_path)
        assert files == []

    def test_available_test_files_finds_test_py(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_something.py").write_text("")
        agent = VerifierAgent()
        files = agent._available_test_files(tmp_path)
        assert any("test_something.py" in f for f in files)

    def test_is_repo_wide_pytest_command_true(self):
        agent = VerifierAgent()
        assert agent._is_repo_wide_pytest_command(["pytest"]) is True
        assert agent._is_repo_wide_pytest_command(["pytest", "-q"]) is True

    def test_is_repo_wide_pytest_command_false_for_targeted(self):
        agent = VerifierAgent()
        assert agent._is_repo_wide_pytest_command(["pytest", "tests/test_foo.py"]) is False

    def test_normalize_test_command_from_list(self):
        agent = VerifierAgent()
        result = agent._normalize_test_command(["pytest -q tests/test_foo.py"])
        assert result == ["pytest", "-q", "tests/test_foo.py"]

    def test_normalize_test_command_from_string(self):
        agent = VerifierAgent()
        result = agent._normalize_test_command("pytest tests/test_foo.py")
        assert result == ["pytest", "tests/test_foo.py"]

    def test_normalize_test_command_empty_returns_empty(self):
        agent = VerifierAgent()
        assert agent._normalize_test_command([]) == []
        assert agent._normalize_test_command("") == []

    def test_related_test_files_prefers_exact_match(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text("")
        (tmp_path / "tests" / "test_bar.py").write_text("")
        agent = VerifierAgent()
        available = ["tests/test_foo.py", "tests/test_bar.py"]
        matches = agent._related_test_files(tmp_path, "foo.py", available)
        assert matches[0] == "tests/test_foo.py"
