from unittest.mock import patch

from agents.skills.registry import all_skills
from agents.skills.test_and_observe import (
    CommandResult,
    TestAndObserveSkill,
    parse_flake8_output,
    parse_node_stacktrace,
    parse_pytest_output,
    parse_python_traceback,
)


def test_parse_python_traceback_uses_last_frame():
    stderr = """Traceback (most recent call last):
  File "/tmp/runner.py", line 4, in <module>
    main()
  File "/workspace/app.py", line 12, in main
    explode()
RuntimeError: boom
"""
    diagnostics = parse_python_traceback(stderr)

    assert len(diagnostics) == 1
    assert diagnostics[0].message == "RuntimeError: boom"
    assert diagnostics[0].primary_location.file == "/workspace/app.py"
    assert diagnostics[0].primary_location.line == 12


def test_skill_run_empty_input_returns_success_dict():
    result = TestAndObserveSkill().run({})

    assert result["status"] == "success"
    assert result["summary"]["runs_total"] == 0
    assert result["runs"] == []
    assert result["diagnostics"] == []


def test_skill_collects_diagnostics_from_failed_run():
    skill = TestAndObserveSkill()
    failure = CommandResult(
        id="unit",
        exit_code=1,
        duration_sec=0.2,
        stdout="",
        stderr="""Traceback (most recent call last):
  File "/repo/tests/test_demo.py", line 9, in <module>
    raise ValueError("bad")
ValueError: bad
""",
    )

    with patch("agents.skills.test_and_observe.execute_command", return_value=failure):
        result = skill.run({"runs": [{"id": "unit", "cmd": ["pytest"]}]})

    assert result["status"] == "failure"
    assert result["summary"]["runs_failed"] == 1
    assert result["runs"][0]["id"] == "unit"
    assert result["diagnostics"][0]["kind"] == "python_traceback"
    assert result["diagnostics"][0]["primary_location"]["file"] == "/repo/tests/test_demo.py"


def test_python_traceback_populates_suggested_next_commands():
    stderr = """Traceback (most recent call last):
  File "/app/main.py", line 7, in run
    do_thing()
AttributeError: 'NoneType' object has no attribute 'x'
"""
    diagnostics = parse_python_traceback(stderr)

    assert len(diagnostics) == 1
    cmds = diagnostics[0].suggested_next_commands
    assert any("pytest" in cmd for cmd in cmds)
    assert any("/app/main.py" in " ".join(cmd) for cmd in cmds)


# ---------------------------------------------------------------------------
# parse_pytest_output
# ---------------------------------------------------------------------------

def test_parse_pytest_output_extracts_failed_test_with_message():
    stdout = """\
collected 3 items

FAILED tests/test_auth.py::TestLogin::test_bad_password - AssertionError: assert 401 == 200
PASSED tests/test_auth.py::TestLogin::test_good_password
"""
    diagnostics = parse_pytest_output(stdout)

    assert len(diagnostics) == 1
    d = diagnostics[0]
    assert d.kind == "pytest_failure"
    assert d.severity == "error"
    assert d.primary_location.file == "tests/test_auth.py"
    assert "AssertionError" in d.message
    assert any("test_auth.py" in " ".join(cmd) for cmd in d.suggested_next_commands)


def test_parse_pytest_output_extracts_multiple_failures():
    stdout = """\
FAILED tests/test_a.py::test_one - ValueError: bad
FAILED tests/test_b.py::TestSuite::test_two - TypeError: oops
"""
    diagnostics = parse_pytest_output(stdout)

    assert len(diagnostics) == 2
    assert diagnostics[0].primary_location.file == "tests/test_a.py"
    assert diagnostics[1].primary_location.file == "tests/test_b.py"


def test_parse_pytest_output_no_message_uses_fallback():
    stdout = "FAILED tests/test_x.py::test_no_reason\n"
    diagnostics = parse_pytest_output(stdout)

    assert len(diagnostics) == 1
    assert diagnostics[0].message == "test failed"


def test_parse_pytest_output_no_failures_returns_empty():
    stdout = "PASSED tests/test_ok.py::test_good\n1 passed in 0.12s\n"
    assert parse_pytest_output(stdout) == []


# ---------------------------------------------------------------------------
# parse_node_stacktrace
# ---------------------------------------------------------------------------

def test_parse_node_stacktrace_extracts_innermost_app_frame():
    stderr = """\
TypeError: Cannot read properties of undefined (reading 'name')
    at getUser (/workspace/src/users.js:42:15)
    at Object.<anonymous> (/workspace/src/index.js:10:5)
    at Module._compile (node:internal/modules/cjs/loader:1356:14)
"""
    diagnostics = parse_node_stacktrace(stderr)

    assert len(diagnostics) == 1
    d = diagnostics[0]
    assert d.kind == "node_stacktrace"
    assert d.severity == "error"
    assert "TypeError" in d.message
    assert d.primary_location.file == "/workspace/src/users.js"
    assert d.primary_location.line == 42
    assert d.primary_location.col == 15


def test_parse_node_stacktrace_skips_internal_frames():
    stderr = """\
RangeError: Maximum call stack size exceeded
    at node:internal/modules/cjs/loader:200:10
    at node:vm:140:3
"""
    # All frames are internal — no app frame to pin to
    diagnostics = parse_node_stacktrace(stderr)
    assert diagnostics == []


def test_parse_node_stacktrace_no_error_marker_returns_empty():
    assert parse_node_stacktrace("nothing useful here") == []


# ---------------------------------------------------------------------------
# parse_flake8_output
# ---------------------------------------------------------------------------

def test_parse_flake8_output_error_code_is_error_severity():
    stdout = "src/app.py:10:1: E302 expected 2 blank lines, found 1\n"
    diagnostics = parse_flake8_output(stdout)

    assert len(diagnostics) == 1
    d = diagnostics[0]
    assert d.kind == "lint_violation"
    assert d.severity == "error"
    assert d.primary_location.file == "src/app.py"
    assert d.primary_location.line == 10
    assert d.primary_location.col == 1
    assert "E302" in d.message
    assert any("E302" in " ".join(cmd) for cmd in d.suggested_next_commands)


def test_parse_flake8_output_warning_code_is_warning_severity():
    stdout = "src/utils.py:5:20: W291 trailing whitespace\n"
    diagnostics = parse_flake8_output(stdout)

    assert len(diagnostics) == 1
    assert diagnostics[0].severity == "warning"


def test_parse_flake8_output_multiple_violations():
    stdout = (
        "a.py:1:1: E302 expected 2 blank lines\n"
        "a.py:3:5: W291 trailing whitespace\n"
        "b.py:7:1: F401 'os' imported but unused\n"
    )
    diagnostics = parse_flake8_output(stdout)
    assert len(diagnostics) == 3
    assert {d.primary_location.file for d in diagnostics} == {"a.py", "b.py"}


def test_parse_flake8_output_no_violations_returns_empty():
    assert parse_flake8_output("") == []


# ---------------------------------------------------------------------------
# Skill integration — combined output parsing
# ---------------------------------------------------------------------------

def test_skill_runs_all_parsers_on_combined_output():
    """Both a flake8 violation (stdout) and a Python traceback (stderr)
    should be collected from the same failed run."""
    skill = TestAndObserveSkill()
    failure = CommandResult(
        id="combined",
        exit_code=1,
        duration_sec=0.5,
        stdout="src/app.py:2:1: E302 expected 2 blank lines\n",
        stderr="Traceback (most recent call last):\n"
               '  File "/src/app.py", line 5, in <module>\n'
               "    boom()\n"
               "RuntimeError: exploded\n",
    )
    with patch("agents.skills.test_and_observe.execute_command", return_value=failure):
        result = skill.run({"runs": [{"id": "combined", "cmd": ["python3", "src/app.py"]}]})

    kinds = {d["kind"] for d in result["diagnostics"]}
    assert "lint_violation" in kinds
    assert "python_traceback" in kinds


def test_skill_timeout_produces_failure_status():
    """A timed-out run has exit_code=-1 and timed_out=True; skill reports failure."""
    skill = TestAndObserveSkill()
    timeout_result = CommandResult(
        id="slow",
        exit_code=-1,
        duration_sec=60.0,
        stdout="",
        stderr="",
        timed_out=True,
    )
    with patch("agents.skills.test_and_observe.execute_command", return_value=timeout_result):
        result = skill.run({"runs": [{"id": "slow", "cmd": ["sleep", "9999"]}]})

    assert result["status"] == "failure"
    assert result["runs"][0]["timed_out"] is True


def test_registry_includes_test_and_observe_skill():
    skills = all_skills()

    assert "test_and_observe" in skills
    assert skills["test_and_observe"].name == "test_and_observe"
