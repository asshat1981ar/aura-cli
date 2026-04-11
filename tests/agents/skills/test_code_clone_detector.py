"""Tests for agents/skills/code_clone_detector.py."""

import pytest
from pathlib import Path
from unittest.mock import patch

from agents.skills.code_clone_detector import (
    _normalize_ast,
    _extract_functions,
    _jaccard,
    CodeCloneDetectorSkill,
)
import ast


# ---------------------------------------------------------------------------
# _normalize_ast
# ---------------------------------------------------------------------------


class TestNormalizeAst:
    def test_returns_string(self):
        tree = ast.parse("x = 1")
        result = _normalize_ast(tree)
        assert isinstance(result, str)

    def test_contains_node_type(self):
        tree = ast.parse("def foo(): pass")
        result = _normalize_ast(tree)
        assert "FunctionDef" in result

    def test_identical_structure_same_hash(self):
        a = ast.parse("def foo(x):\n    return x + 1\n")
        b = ast.parse("def bar(y):\n    return y + 1\n")
        # structural normalization ignores names, so these should match
        na = _normalize_ast(a)
        nb = _normalize_ast(b)
        assert na == nb

    def test_different_structure_different_result(self):
        a = ast.parse("def foo():\n    pass\n")
        b = ast.parse("def foo():\n    return 1\n")
        assert _normalize_ast(a) != _normalize_ast(b)


# ---------------------------------------------------------------------------
# _extract_functions
# ---------------------------------------------------------------------------


class TestExtractFunctions:
    def test_empty_source_returns_empty(self):
        assert _extract_functions("", "f.py", 1) == []

    def test_syntax_error_returns_empty(self):
        assert _extract_functions("def :(", "f.py", 1) == []

    def test_extracts_function(self):
        src = "def foo():\n    pass\n"
        results = _extract_functions(src, "f.py", 1)
        assert len(results) == 1

    def test_result_tuple_shape(self):
        src = "def foo():\n    pass\n"
        h, norm, fpath, line, name = _extract_functions(src, "f.py", 1)[0]
        assert isinstance(h, str) and len(h) == 32  # md5 hex
        assert fpath == "f.py"
        assert name == "foo"
        assert line == 1

    def test_min_lines_filters_short_functions(self):
        src = "def tiny():\n    pass\n"
        # 2-line function, require 10 lines → filtered out
        assert _extract_functions(src, "f.py", 10) == []

    def test_async_function_extracted(self):
        src = "async def fetch():\n    pass\n"
        results = _extract_functions(src, "f.py", 1)
        assert len(results) == 1
        assert results[0][4] == "fetch"

    def test_multiple_functions_all_extracted(self):
        src = "def a():\n    pass\ndef b():\n    pass\n"
        results = _extract_functions(src, "f.py", 1)
        names = [r[4] for r in results]
        assert "a" in names
        assert "b" in names


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_strings_score_1(self):
        assert _jaccard("a,b,c", "a,b,c") == 1.0

    def test_disjoint_strings_score_0(self):
        assert _jaccard("a,b", "c,d") == 0.0

    def test_partial_overlap(self):
        score = _jaccard("a,b,c", "b,c,d")
        assert 0.0 < score < 1.0

    def test_empty_strings_no_crash(self):
        score = _jaccard("", "")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# CodeCloneDetectorSkill._run
# ---------------------------------------------------------------------------


class TestCodeCloneDetectorSkill:
    @pytest.fixture
    def skill(self):
        return CodeCloneDetectorSkill()

    def test_returns_dict(self, skill, tmp_path):
        result = skill.run({"project_root": str(tmp_path)})
        assert isinstance(result, dict)

    def test_required_keys_present(self, skill, tmp_path):
        result = skill.run({"project_root": str(tmp_path)})
        for key in ("exact_clones", "near_duplicates", "clone_count", "functions_analyzed"):
            assert key in result

    def test_empty_dir_no_clones(self, skill, tmp_path):
        result = skill.run({"project_root": str(tmp_path)})
        assert result["clone_count"] == 0
        assert result["functions_analyzed"] == 0

    def test_detects_exact_clone(self, skill, tmp_path):
        body = "def work():\n    x = 1\n    y = 2\n    z = x + y\n    return z\n"
        (tmp_path / "a.py").write_text(body)
        (tmp_path / "b.py").write_text(body)
        result = skill.run({"project_root": str(tmp_path), "min_lines": 3})
        assert result["clone_count"] >= 1
        assert len(result["exact_clones"]) >= 1

    def test_suggestions_generated_for_clones(self, skill, tmp_path):
        body = "def dup():\n    a = 1\n    b = 2\n    c = a + b\n    return c\n"
        (tmp_path / "x.py").write_text(body)
        (tmp_path / "y.py").write_text(body)
        result = skill.run({"project_root": str(tmp_path), "min_lines": 3})
        if result["clone_count"] > 0:
            assert len(result["consolidation_suggestions"]) >= 1

    def test_functions_analyzed_counts_correctly(self, skill, tmp_path):
        (tmp_path / "m.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n")
        result = skill.run({"project_root": str(tmp_path), "min_lines": 1})
        assert result["functions_analyzed"] == 2

    def test_exact_clones_capped_at_50(self, skill, tmp_path):
        result = skill.run({"project_root": str(tmp_path)})
        assert len(result["exact_clones"]) <= 50

    def test_near_duplicates_capped_at_50(self, skill, tmp_path):
        result = skill.run({"project_root": str(tmp_path)})
        assert len(result["near_duplicates"]) <= 50

    def test_default_project_root_no_crash(self, skill):
        # Running with no project_root defaults to "." — should not raise
        result = skill.run({})
        assert "clone_count" in result
