"""Integration tests: lint and test_and_observe skill pipeline.

Exercises the two new skills end-to-end with mocked subprocesses,
and verifies they are selected by the dispatcher for appropriate goal types.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.skills.lint import LintSkill
from agents.skills.test_and_observe import (
    CommandResult,
    TestAndObserveSkill,
    parse_flake8_output,
)
from core.skill_dispatcher import SKILL_MAP, dispatch_skills


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(returncode=0, stdout="", stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _make_command_result(exit_code=0, stdout="", stderr="", timed_out=False):
    return CommandResult(
        id="test",
        exit_code=exit_code,
        duration_sec=0.1,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# LintSkill end-to-end
# ---------------------------------------------------------------------------

class TestLintSkillEndToEnd:
    def test_clean_file_pipeline(self):
        """
        Arrange: flake8 exits 0 with no output.
        Assert:  skill returns status=success, zero violations, raw is empty.
        """
        skill = LintSkill()
        with patch("agents.skills.lint.subprocess.run",
                   return_value=_make_proc(0, stdout="")):
            result = skill.run({"files": ["clean.py"]})

        assert result["status"] == "success"
        assert result["violation_count"] == 0
        assert result["violations"] == []
        assert result["raw"] == ""

    def test_violations_pipeline(self):
        """
        Arrange: flake8 reports two violations.
        Assert:  skill parses both, returns status=violations_found.
        """
        stdout = (
            "src/a.py:3:1: E302 expected 2 blank lines, found 1\n"
            "src/a.py:7:5: W291 trailing whitespace\n"
        )
        skill = LintSkill()
        with patch("agents.skills.lint.subprocess.run",
                   return_value=_make_proc(1, stdout=stdout)):
            result = skill.run({"files": ["src/a.py"]})

        assert result["status"] == "violations_found"
        assert result["violation_count"] == 2
        assert result["violations"][0]["severity"] == "error"    # E302
        assert result["violations"][1]["severity"] == "warning"  # W291

    def test_violation_carries_suggested_commands(self):
        """Every violation must include at least one suggested remediation command."""
        stdout = "module.py:1:1: F401 'os' imported but unused\n"
        skill = LintSkill()
        with patch("agents.skills.lint.subprocess.run",
                   return_value=_make_proc(1, stdout=stdout)):
            result = skill.run({"files": ["module.py"]})

        v = result["violations"][0]
        assert len(v["suggested_next_commands"]) >= 1
        assert any("F401" in " ".join(cmd) for cmd in v["suggested_next_commands"])

    def test_missing_flake8_returns_error_not_exception(self):
        """Missing flake8 binary must return an error dict, not raise."""
        skill = LintSkill()
        with patch("agents.skills.lint.subprocess.run",
                   side_effect=FileNotFoundError("no flake8")):
            result = skill.run({"files": ["x.py"]})

        assert result["status"] == "error"
        assert "flake8" in result["error"].lower()

    def test_staged_file_fallback_filters_python_only(self):
        """
        When files= is omitted, only .py files from git staging are linted.
        Non-Python staged files must be silently ignored.
        """
        git_output = b"src/app.py\nREADME.md\nstatic/main.js\n"
        skill = LintSkill()
        with patch("agents.skills.lint.subprocess.check_output",
                   return_value=git_output):
            with patch("agents.skills.lint.subprocess.run",
                       return_value=_make_proc(0)) as mock_run:
                result = skill.run({})

        cmd = mock_run.call_args[0][0]
        assert "src/app.py" in cmd
        assert "README.md" not in cmd
        assert "main.js" not in cmd
        assert result["files_checked"] == 1


# ---------------------------------------------------------------------------
# TestAndObserveSkill end-to-end
# ---------------------------------------------------------------------------

class TestAndObserveSkillEndToEnd:
    def test_passing_command_returns_success(self):
        """A zero-exit command produces status=success with no diagnostics."""
        skill = TestAndObserveSkill()
        success = _make_command_result(exit_code=0, stdout="all tests passed")
        with patch("agents.skills.test_and_observe.execute_command",
                   return_value=success):
            result = skill.run({"runs": [{"id": "r1", "cmd": ["pytest"]}]})

        assert result["status"] == "success"
        assert result["diagnostics"] == []
        assert result["summary"]["runs_failed"] == 0

    def test_python_traceback_produces_diagnostic(self):
        """A Python traceback in stderr yields a python_traceback diagnostic."""
        skill = TestAndObserveSkill()
        failure = _make_command_result(
            exit_code=1,
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                '  File "/app/main.py", line 12, in run\n'
                "    do_thing()\n"
                "ValueError: invalid literal\n"
            ),
        )
        with patch("agents.skills.test_and_observe.execute_command",
                   return_value=failure):
            result = skill.run({"runs": [{"id": "r1", "cmd": ["python3", "main.py"]}]})

        assert result["status"] == "failure"
        diag = next(d for d in result["diagnostics"] if d["kind"] == "python_traceback")
        assert diag["primary_location"]["file"] == "/app/main.py"
        assert diag["primary_location"]["line"] == 12
        assert len(diag["suggested_next_commands"]) >= 1

    def test_pytest_failure_produces_diagnostic(self):
        """pytest FAILED lines in stdout yield pytest_failure diagnostics."""
        skill = TestAndObserveSkill()
        failure = _make_command_result(
            exit_code=1,
            stdout=(
                "FAILED tests/test_auth.py::TestLogin::test_bad_token"
                " - AssertionError: 401 != 200\n"
            ),
            stderr="",
        )
        with patch("agents.skills.test_and_observe.execute_command",
                   return_value=failure):
            result = skill.run({"runs": [{"id": "r1", "cmd": ["pytest"]}]})

        diag = next(
            (d for d in result["diagnostics"] if d["kind"] == "pytest_failure"), None
        )
        assert diag is not None
        assert diag["primary_location"]["file"] == "tests/test_auth.py"

    def test_node_stacktrace_produces_diagnostic(self):
        """A Node.js stack trace produces a node_stacktrace diagnostic."""
        skill = TestAndObserveSkill()
        failure = _make_command_result(
            exit_code=1,
            stdout="",
            stderr=(
                "TypeError: Cannot read properties of undefined\n"
                "    at getUser (/srv/users.js:42:15)\n"
                "    at Module._compile (node:internal/loader:100:10)\n"
            ),
        )
        with patch("agents.skills.test_and_observe.execute_command",
                   return_value=failure):
            result = skill.run({"runs": [{"id": "r1", "cmd": ["node", "index.js"]}]})

        diag = next(
            (d for d in result["diagnostics"] if d["kind"] == "node_stacktrace"), None
        )
        assert diag is not None
        assert diag["primary_location"]["file"] == "/srv/users.js"
        assert diag["primary_location"]["line"] == 42

    def test_flake8_output_in_stdout_produces_lint_diagnostic(self):
        """flake8 violations in stdout are captured by the lint_violation parser."""
        skill = TestAndObserveSkill()
        failure = _make_command_result(
            exit_code=1,
            stdout="src/utils.py:5:1: E302 expected 2 blank lines\n",
            stderr="",
        )
        with patch("agents.skills.test_and_observe.execute_command",
                   return_value=failure):
            result = skill.run({"runs": [{"id": "r1", "cmd": ["flake8", "src/"]}]})

        diag = next(
            (d for d in result["diagnostics"] if d["kind"] == "lint_violation"), None
        )
        assert diag is not None
        assert diag["primary_location"]["file"] == "src/utils.py"

    def test_multiple_parsers_fire_on_combined_output(self):
        """
        When a run produces both a Python traceback and a flake8 violation,
        both diagnostic kinds appear in the output.
        """
        skill = TestAndObserveSkill()
        failure = _make_command_result(
            exit_code=1,
            stdout="bad.py:2:1: E302 missing blank lines\n",
            stderr=(
                "Traceback (most recent call last):\n"
                '  File "/bad.py", line 5, in <module>\n'
                "    crash()\n"
                "RuntimeError: oops\n"
            ),
        )
        with patch("agents.skills.test_and_observe.execute_command",
                   return_value=failure):
            result = skill.run({"runs": [{"id": "r1", "cmd": ["python3", "bad.py"]}]})

        kinds = {d["kind"] for d in result["diagnostics"]}
        assert "python_traceback" in kinds
        assert "lint_violation" in kinds

    def test_timed_out_run_status_is_failure(self):
        """A timed-out command reports failure and timed_out=True."""
        skill = TestAndObserveSkill()
        timeout_result = _make_command_result(exit_code=-1, timed_out=True)
        with patch("agents.skills.test_and_observe.execute_command",
                   return_value=timeout_result):
            result = skill.run({"runs": [{"id": "slow", "cmd": ["sleep", "9999"]}]})

        assert result["status"] == "failure"
        assert result["runs"][0]["timed_out"] is True

    def test_multiple_runs_all_reported(self):
        """Multiple run configs are each executed and summarised."""
        skill = TestAndObserveSkill()
        results_seq = [
            _make_command_result(exit_code=0, stdout="ok"),
            _make_command_result(exit_code=1, stderr=""),
        ]
        with patch("agents.skills.test_and_observe.execute_command",
                   side_effect=results_seq):
            result = skill.run({
                "runs": [
                    {"id": "a", "cmd": ["echo", "ok"]},
                    {"id": "b", "cmd": ["false"]},
                ]
            })

        assert result["summary"]["runs_total"] == 2
        assert result["summary"]["runs_failed"] == 1
        assert result["status"] == "failure"


# ---------------------------------------------------------------------------
# Skill dispatcher — new skills selected per goal type
# ---------------------------------------------------------------------------

class TestDispatcherSkillSelection:
    def _skill(self, name):
        s = MagicMock()
        s.name = name
        s.run.return_value = {"status": "ok", "skill": name}
        return s

    def test_bug_fix_selects_lint_and_test_and_observe(self):
        skills = {
            "lint": self._skill("lint"),
            "test_and_observe": self._skill("test_and_observe"),
            "symbol_indexer": self._skill("symbol_indexer"),
        }
        results = dispatch_skills("bug_fix", skills, ".")
        assert "lint" in results
        assert "test_and_observe" in results

    def test_refactor_selects_lint_and_test_and_observe(self):
        skills = {
            "lint": self._skill("lint"),
            "test_and_observe": self._skill("test_and_observe"),
        }
        results = dispatch_skills("refactor", skills, ".")
        assert "lint" in results
        assert "test_and_observe" in results

    def test_feature_selects_lint_but_not_test_and_observe(self):
        skills = {
            "lint": self._skill("lint"),
            "test_and_observe": self._skill("test_and_observe"),
        }
        results = dispatch_skills("feature", skills, ".")
        assert "lint" in results
        assert "test_and_observe" not in results

    def test_default_selects_lint(self):
        skills = {"lint": self._skill("lint")}
        results = dispatch_skills("default", skills, ".")
        assert "lint" in results

    def test_lint_skill_run_called_with_project_root(self):
        """dispatch_skills passes project_root into each skill.run() call."""
        lint_skill = self._skill("lint")
        dispatch_skills("default", {"lint": lint_skill}, "/my/project")
        lint_skill.run.assert_called_once_with({"project_root": "/my/project"})

    def test_test_and_observe_in_bug_fix_before_lint(self):
        """test_and_observe appears before lint in SKILL_MAP['bug_fix']."""
        skills_list = SKILL_MAP["bug_fix"]
        assert skills_list.index("test_and_observe") < skills_list.index("lint")

    def test_skill_failure_does_not_prevent_other_skills_running(self):
        lint_skill = self._skill("lint")
        lint_skill.run.side_effect = RuntimeError("flake8 not found")
        tao_skill = self._skill("test_and_observe")

        results = dispatch_skills(
            "bug_fix",
            {"lint": lint_skill, "test_and_observe": tao_skill},
            ".",
        )
        assert "lint" in results
        assert results["lint"].get("error") is not None
        assert "test_and_observe" in results
        assert results["test_and_observe"]["status"] == "ok"
