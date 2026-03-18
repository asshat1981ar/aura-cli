"""Tests for LintSkill."""
from unittest.mock import MagicMock, patch
import subprocess

from agents.skills.lint import LintSkill
from agents.skills.registry import all_skills


def _make_proc(returncode=0, stdout="", stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# Basic output contracts
# ---------------------------------------------------------------------------

def test_lint_skill_clean_files_returns_success():
    skill = LintSkill()
    with patch("agents.skills.lint.subprocess.run", return_value=_make_proc(0)) as mock_run:
        result = skill.run({"files": ["src/app.py"]})

    assert result["status"] == "success"
    assert result["files_checked"] == 1
    assert result["violation_count"] == 0
    assert result["violations"] == []
    mock_run.assert_called_once()


def test_lint_skill_violations_returns_violations_found():
    stdout = "src/app.py:4:1: E302 expected 2 blank lines, found 1\n"
    skill = LintSkill()
    with patch("agents.skills.lint.subprocess.run", return_value=_make_proc(1, stdout=stdout)):
        result = skill.run({"files": ["src/app.py"]})

    assert result["status"] == "violations_found"
    assert result["violation_count"] == 1
    v = result["violations"][0]
    assert v["file"] == "src/app.py"
    assert v["line"] == 4
    assert v["col"] == 1
    assert "E302" in v["message"]
    assert v["severity"] == "error"


def test_lint_skill_warning_code_produces_warning_severity():
    stdout = "utils.py:10:5: W291 trailing whitespace\n"
    skill = LintSkill()
    with patch("agents.skills.lint.subprocess.run", return_value=_make_proc(1, stdout=stdout)):
        result = skill.run({"files": ["utils.py"]})

    assert result["violations"][0]["severity"] == "warning"


def test_lint_skill_violation_includes_suggested_commands():
    stdout = "a.py:1:1: E401 multiple imports\n"
    skill = LintSkill()
    with patch("agents.skills.lint.subprocess.run", return_value=_make_proc(1, stdout=stdout)):
        result = skill.run({"files": ["a.py"]})

    cmds = result["violations"][0]["suggested_next_commands"]
    assert any("E401" in " ".join(cmd) for cmd in cmds)


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------

def test_lint_skill_no_files_and_no_staged_returns_success_without_calling_flake8():
    skill = LintSkill()
    with patch.object(skill, "_get_staged_files", return_value=[]):
        with patch("agents.skills.lint.subprocess.run") as mock_run:
            result = skill.run({})

    assert result["status"] == "success"
    assert result["files_checked"] == 0
    mock_run.assert_not_called()


def test_lint_skill_explicit_empty_files_list_returns_success():
    skill = LintSkill()
    with patch("agents.skills.lint.subprocess.run") as mock_run:
        result = skill.run({"files": []})

    assert result["status"] == "success"
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Staged-file fallback
# ---------------------------------------------------------------------------

def test_lint_skill_falls_back_to_staged_files_when_files_omitted():
    skill = LintSkill()
    with patch.object(skill, "_get_staged_files", return_value=["changed.py"]) as mock_staged:
        with patch("agents.skills.lint.subprocess.run", return_value=_make_proc(0)) as mock_run:
            result = skill.run({})

    mock_staged.assert_called_once()
    assert result["files_checked"] == 1
    # Confirm "changed.py" was in the command
    called_cmd = mock_run.call_args[0][0]
    assert "changed.py" in called_cmd


def test_get_staged_files_returns_only_python_files(tmp_path):
    skill = LintSkill()
    git_output = b"src/app.py\nREADME.md\ntests/test_x.py\n"
    with patch("agents.skills.lint.subprocess.check_output", return_value=git_output):
        result = skill._get_staged_files()

    assert result == ["src/app.py", "tests/test_x.py"]


def test_get_staged_files_returns_empty_on_git_error():
    skill = LintSkill()
    with patch(
        "agents.skills.lint.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(128, "git"),
    ):
        result = skill._get_staged_files()

    assert result == []


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------

def test_lint_skill_handles_missing_flake8_gracefully():
    skill = LintSkill()
    with patch(
        "agents.skills.lint.subprocess.run",
        side_effect=FileNotFoundError("flake8 not found"),
    ):
        result = skill.run({"files": ["app.py"]})

    assert result["status"] == "error"
    assert "flake8 not installed" in result["error"]
    assert result["violations"] == []


def test_lint_skill_handles_timeout_gracefully():
    skill = LintSkill()
    with patch(
        "agents.skills.lint.subprocess.run",
        side_effect=subprocess.TimeoutExpired("flake8", 60),
    ):
        result = skill.run({"files": ["app.py"]})

    assert result["status"] == "error"
    assert "timed out" in result["error"]


def test_lint_skill_config_path_passed_to_flake8():
    skill = LintSkill()
    with patch("agents.skills.lint.subprocess.run", return_value=_make_proc(0)) as mock_run:
        skill.run({"files": ["app.py"], "config": ".flake8"})

    called_cmd = mock_run.call_args[0][0]
    assert "--config=.flake8" in called_cmd


def test_lint_skill_raw_stdout_included_in_result():
    raw = "app.py:1:1: E302 missing blank lines\n"
    skill = LintSkill()
    with patch("agents.skills.lint.subprocess.run", return_value=_make_proc(1, stdout=raw)):
        result = skill.run({"files": ["app.py"]})

    assert result["raw"] == raw


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_lint_skill_registered_in_all_skills():
    skills = all_skills()

    assert "lint" in skills
    assert skills["lint"].name == "lint"
    assert isinstance(skills["lint"], LintSkill)
