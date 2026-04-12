"""Tests for agents/skills/type_checker.py — _annotation_coverage, TypeCheckerSkill."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.skills.type_checker import _annotation_coverage, TypeCheckerSkill


# ---------------------------------------------------------------------------
# _annotation_coverage
# ---------------------------------------------------------------------------

class TestAnnotationCoverage:
    def test_fully_annotated_returns_100(self):
        src = "def foo(x: int) -> str:\n    return str(x)\n"
        assert _annotation_coverage(src) == 100.0

    def test_unannotated_returns_0(self):
        src = "def foo(x):\n    return x\n"
        assert _annotation_coverage(src) == 0.0

    def test_partial_annotation(self):
        src = (
            "def annotated(x: int) -> int:\n    return x\n"
            "def bare(x):\n    return x\n"
        )
        cov = _annotation_coverage(src)
        assert 0.0 < cov < 100.0

    def test_no_functions_returns_0(self):
        src = "x = 1\ny = 2\n"
        # no functions → total=0, division guard returns 0.0
        assert _annotation_coverage(src) == 0.0

    def test_syntax_error_returns_0(self):
        assert _annotation_coverage("def foo(: bad syntax") == 0.0

    def test_async_function_counted(self):
        src = "async def fetch(url: str) -> bytes:\n    pass\n"
        assert _annotation_coverage(src) == 100.0

    def test_return_annotation_alone_counts(self):
        src = "def foo() -> None:\n    pass\n"
        assert _annotation_coverage(src) == 100.0

    def test_arg_annotation_alone_counts(self):
        src = "def foo(x: int):\n    pass\n"
        assert _annotation_coverage(src) == 100.0


# ---------------------------------------------------------------------------
# TypeCheckerSkill._run — mypy unavailable (heuristic fallback)
# ---------------------------------------------------------------------------

class TestTypeCheckerSkillHeuristicFallback:
    @pytest.fixture
    def skill(self):
        return TypeCheckerSkill()

    def test_returns_dict(self, skill, tmp_path):
        with patch("agents.skills.type_checker._run_mypy", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert isinstance(result, dict)

    def test_mypy_available_false(self, skill, tmp_path):
        with patch("agents.skills.type_checker._run_mypy", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["mypy_available"] is False

    def test_fallback_note_present(self, skill, tmp_path):
        with patch("agents.skills.type_checker._run_mypy", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert "note" in result

    def test_fallback_annotation_coverage_numeric(self, skill, tmp_path):
        (tmp_path / "m.py").write_text("def foo(x: int) -> int:\n    return x\n")
        with patch("agents.skills.type_checker._run_mypy", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert isinstance(result["annotation_coverage_pct"], float)

    def test_fallback_error_count_zero(self, skill, tmp_path):
        with patch("agents.skills.type_checker._run_mypy", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["error_count"] == 0

    def test_fallback_type_errors_empty(self, skill, tmp_path):
        with patch("agents.skills.type_checker._run_mypy", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["type_errors"] == []


# ---------------------------------------------------------------------------
# TypeCheckerSkill._run — mypy available
# ---------------------------------------------------------------------------

class TestTypeCheckerSkillWithMypy:
    @pytest.fixture
    def skill(self):
        return TypeCheckerSkill()

    def test_mypy_available_true(self, skill, tmp_path):
        with patch("agents.skills.type_checker._run_mypy", return_value=[]):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["mypy_available"] is True

    def test_no_errors_returns_zero_count(self, skill, tmp_path):
        with patch("agents.skills.type_checker._run_mypy", return_value=[]):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["error_count"] == 0
        assert result["type_errors"] == []

    def test_errors_counted(self, skill, tmp_path):
        errors = [
            {"file": "x.py", "line": 1, "level": "error", "error": "bad type"},
            {"file": "x.py", "line": 2, "level": "error", "error": "another"},
        ]
        with patch("agents.skills.type_checker._run_mypy", return_value=errors):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["error_count"] == 2

    def test_warnings_not_counted_as_errors(self, skill, tmp_path):
        mixed = [
            {"file": "x.py", "line": 1, "level": "warning", "error": "style"},
            {"file": "x.py", "line": 2, "level": "error", "error": "bad"},
        ]
        with patch("agents.skills.type_checker._run_mypy", return_value=mixed):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["error_count"] == 1

    def test_annotation_coverage_included(self, skill, tmp_path):
        (tmp_path / "f.py").write_text("def foo(x: int) -> int:\n    return x\n")
        with patch("agents.skills.type_checker._run_mypy", return_value=[]):
            result = skill.run({"project_root": str(tmp_path)})
        assert "annotation_coverage_pct" in result

    def test_type_errors_capped_at_100(self, skill, tmp_path):
        errors = [
            {"file": "x.py", "line": i, "level": "error", "error": "e"}
            for i in range(150)
        ]
        with patch("agents.skills.type_checker._run_mypy", return_value=errors):
            result = skill.run({"project_root": str(tmp_path)})
        assert len(result["type_errors"]) == 100

    def test_no_project_root_uses_cwd(self, skill):
        with patch("agents.skills.type_checker._run_mypy", return_value=[]):
            result = skill.run({})
        assert "error_count" in result


# ---------------------------------------------------------------------------
# _run_mypy helper (smoke)
# ---------------------------------------------------------------------------

class TestRunMypy:
    def test_returns_none_when_mypy_missing(self):
        from agents.skills.type_checker import _run_mypy
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _run_mypy(".", Path("."))
        assert result is None

    def test_returns_list_on_success(self):
        from agents.skills.type_checker import _run_mypy
        mock_proc = MagicMock()
        mock_proc.stdout = "module.py:5: error: Incompatible types\n"
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            result = _run_mypy(".", Path("."))
        assert isinstance(result, list)
        assert result[0]["level"] == "error"
