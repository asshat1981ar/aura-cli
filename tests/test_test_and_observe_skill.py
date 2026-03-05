from unittest.mock import patch

from agents.skills.registry import all_skills
from agents.skills.test_and_observe import (
    CommandResult,
    TestAndObserveSkill,
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


def test_registry_includes_test_and_observe_skill():
    skills = all_skills()

    assert "test_and_observe" in skills
    assert skills["test_and_observe"].name == "test_and_observe"
