"""Tests for agents/skills/linter_enforcer.py."""

import pytest
from pathlib import Path
from unittest.mock import patch

from agents.skills.linter_enforcer import (
    _camel_to_snake,
    _snake_to_pascal,
    _check_naming,
    _run_flake8,
    _scan_project,
    LinterEnforcerSkill,
)


# ---------------------------------------------------------------------------
# _camel_to_snake / _snake_to_pascal
# ---------------------------------------------------------------------------

class TestNameConversions:
    def test_camel_to_snake_basic(self):
        assert _camel_to_snake("MyClass") == "my_class"

    def test_camel_to_snake_multiple_words(self):
        assert _camel_to_snake("MyBigClass") == "my_big_class"

    def test_camel_to_snake_already_snake(self):
        assert _camel_to_snake("my_func") == "my_func"

    def test_snake_to_pascal_basic(self):
        assert _snake_to_pascal("my_class") == "MyClass"

    def test_snake_to_pascal_single_word(self):
        assert _snake_to_pascal("myclass") == "Myclass"

    def test_snake_to_pascal_multiple_parts(self):
        assert _snake_to_pascal("one_two_three") == "OneTwoThree"


# ---------------------------------------------------------------------------
# _check_naming
# ---------------------------------------------------------------------------

class TestCheckNaming:
    def test_valid_snake_function_no_violation(self):
        src = "def my_function(x):\n    pass\n"
        violations = _check_naming(src, "f.py")
        assert not any(v["code"] == "N802" for v in violations)

    def test_camelcase_function_flagged(self):
        src = "def myFunction(x):\n    pass\n"
        violations = _check_naming(src, "f.py")
        assert any(v["code"] == "N802" for v in violations)

    def test_dunder_method_not_flagged(self):
        src = "def __init__(self):\n    pass\n"
        violations = _check_naming(src, "f.py")
        assert not any(v["code"] == "N802" for v in violations)

    def test_valid_pascal_class_no_violation(self):
        src = "class MyClass:\n    pass\n"
        violations = _check_naming(src, "f.py")
        assert not any(v["code"] == "N801" for v in violations)

    def test_snake_class_flagged(self):
        src = "class my_class:\n    pass\n"
        violations = _check_naming(src, "f.py")
        assert any(v["code"] == "N801" for v in violations)

    def test_syntax_error_returns_empty(self):
        violations = _check_naming("def :(", "bad.py")
        assert violations == []

    def test_violation_has_required_keys(self):
        src = "def BadName():\n    pass\n"
        v = _check_naming(src, "f.py")[0]
        for key in ("code", "file", "line", "message", "fix_hint"):
            assert key in v

    def test_fix_hint_suggests_rename(self):
        src = "def myFunc():\n    pass\n"
        v = _check_naming(src, "f.py")[0]
        assert "my_func" in v["fix_hint"]

    def test_empty_source_no_violations(self):
        assert _check_naming("", "f.py") == []

    def test_multiple_violations_all_reported(self):
        src = "class bad_class:\n    def badMethod(self):\n        pass\n"
        violations = _check_naming(src, "f.py")
        codes = {v["code"] for v in violations}
        assert "N801" in codes
        assert "N802" in codes


# ---------------------------------------------------------------------------
# _run_flake8
# ---------------------------------------------------------------------------

class TestRunFlake8:
    def test_returns_none_when_flake8_missing(self, tmp_path):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _run_flake8(".", tmp_path)
        assert result is None

    def test_returns_list_on_success(self, tmp_path):
        mock = type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()
        with patch("subprocess.run", return_value=mock):
            result = _run_flake8(".", tmp_path)
        assert isinstance(result, list)

    def test_parses_flake8_output(self, tmp_path):
        output = "myfile.py:5:1: E302 expected 2 blank lines, found 1\n"
        mock = type("R", (), {"stdout": output, "stderr": "", "returncode": 1})()
        with patch("subprocess.run", return_value=mock):
            result = _run_flake8(".", tmp_path)
        assert len(result) == 1
        assert result[0]["code"] == "E302"
        assert result[0]["line"] == 5

    def test_fix_hint_populated_for_known_code(self, tmp_path):
        output = "f.py:1:1: E302 expected 2 blank lines\n"
        mock = type("R", (), {"stdout": output, "stderr": "", "returncode": 1})()
        with patch("subprocess.run", return_value=mock):
            result = _run_flake8(".", tmp_path)
        assert result[0]["fix_hint"] != ""

    def test_ignore_codes_passed_to_cmd(self, tmp_path):
        mock = type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()
        with patch("subprocess.run", return_value=mock) as mock_run:
            _run_flake8(".", tmp_path, ignore_codes=["E501"])
        cmd = mock_run.call_args[0][0]
        assert any("E501" in arg for arg in cmd)


# ---------------------------------------------------------------------------
# LinterEnforcerSkill — inline code mode
# ---------------------------------------------------------------------------

class TestLinterEnforcerInlineCode:
    @pytest.fixture
    def skill(self):
        return LinterEnforcerSkill()

    def test_returns_dict(self, skill):
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"code": "x = 1\n", "file_path": "f.py"})
        assert isinstance(result, dict)

    def test_required_keys_present(self, skill):
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"code": "x = 1\n", "file_path": "f.py"})
        for key in ("violations", "violation_count", "error_count", "warning_count", "naming_violation_count"):
            assert key in result

    def test_naming_violations_detected_inline(self, skill):
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"code": "def myFunc():\n    pass\n", "file_path": "f.py"})
        assert result["naming_violation_count"] >= 1

    def test_file_path_label_in_result(self, skill):
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"code": "x = 1\n", "file_path": "my_module.py"})
        assert result["file"] == "my_module.py"


# ---------------------------------------------------------------------------
# LinterEnforcerSkill — single project root
# ---------------------------------------------------------------------------

class TestLinterEnforcerProjectRoot:
    @pytest.fixture
    def skill(self):
        return LinterEnforcerSkill()

    def test_returns_dict_for_project_root(self, skill, tmp_path):
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"project_root": str(tmp_path)})
        assert isinstance(result, dict)

    def test_project_root_in_result(self, skill, tmp_path):
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"project_root": str(tmp_path)})
        assert "violation_count" in result

    def test_naming_violations_counted(self, skill, tmp_path):
        (tmp_path / "bad.py").write_text("class bad_name:\n    pass\n")
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["naming_violation_count"] >= 1


# ---------------------------------------------------------------------------
# LinterEnforcerSkill — multi-project paths
# ---------------------------------------------------------------------------

class TestLinterEnforcerMultiProject:
    @pytest.fixture
    def skill(self):
        return LinterEnforcerSkill()

    def test_multi_project_returns_results_list(self, skill, tmp_path):
        p1 = tmp_path / "proj1"
        p2 = tmp_path / "proj2"
        p1.mkdir()
        p2.mkdir()
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"paths": [str(p1), str(p2)]})
        assert "results" in result
        assert result["projects_scanned"] == 2

    def test_missing_path_records_error(self, skill, tmp_path):
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"paths": [str(tmp_path / "nonexistent")]})
        assert "error" in result["results"][0]

    def test_total_violations_summed(self, skill, tmp_path):
        p1 = tmp_path / "p1"
        p1.mkdir()
        with patch("agents.skills.linter_enforcer._run_flake8", return_value=[]):
            result = skill.run({"paths": [str(p1)]})
        assert isinstance(result["total_violations"], int)
