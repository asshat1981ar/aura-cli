"""Comprehensive tests for security_scanner, test_coverage_analyzer,
dependency_analyzer, and multi_file_editor skills (priority day-4 batch).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Security Scanner
# ---------------------------------------------------------------------------
from agents.skills.security_scanner import (
    SecurityScannerSkill,
    _scan_ast,
    _scan_text,
)


class TestSecurityScanner:
    """Tests for SecurityScannerSkill."""

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        skill = SecurityScannerSkill()
        result = skill.run(input_data)
        return result.data if hasattr(result, "data") else dict(result)

    # -- hardcoded secrets --------------------------------------------------

    def test_detect_hardcoded_api_key(self):
        code = 'api_key = "sk-abcdefghijklmnop1234"\n'
        res = self._run({"code": code, "file_path": "config.py"})
        issues = [f["issue"] for f in res["findings"]]
        assert "hardcoded_secret" in issues
        assert res["critical_count"] >= 1

    def test_detect_aws_access_key(self):
        code = 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
        res = self._run({"code": code})
        issues = [f["issue"] for f in res["findings"]]
        assert "aws_access_key" in issues

    def test_detect_bearer_token(self):
        code = 'headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghijk"}\n'
        res = self._run({"code": code})
        issues = [f["issue"] for f in res["findings"]]
        assert "bearer_token_in_code" in issues

    # -- SQL injection ------------------------------------------------------

    def test_detect_sql_injection(self):
        code = 'query = "SELECT * FROM users WHERE id = %s" % user_id\n'
        res = self._run({"code": code})
        issues = [f["issue"] for f in res["findings"]]
        assert "sql_injection_risk" in issues

    # -- unsafe calls -------------------------------------------------------

    def test_detect_unsafe_eval_exec(self):
        code = textwrap.dedent("""\
            result = eval(user_input)
            exec(some_code)
        """)
        res = self._run({"code": code})
        issues = [f["issue"] for f in res["findings"]]
        assert "unsafe_eval" in issues
        assert "unsafe_exec" in issues

    def test_detect_pickle_loads(self):
        code = textwrap.dedent("""\
            import pickle
            obj = pickle.loads(data)
        """)
        res = self._run({"code": code})
        issues = [f["issue"] for f in res["findings"]]
        assert "unsafe_pickle_loads" in issues

    def test_detect_subprocess_shell_true(self):
        code = textwrap.dedent("""\
            import subprocess
            subprocess.run("ls", shell=True)
        """)
        res = self._run({"code": code})
        issues = [f["issue"] for f in res["findings"]]
        assert "subprocess_shell_true" in issues

    def test_detect_os_system(self):
        code = textwrap.dedent("""\
            import os
            os.system("rm -rf /")
        """)
        res = self._run({"code": code})
        issues = [f["issue"] for f in res["findings"]]
        assert "os_system_call" in issues

    # -- clean code ---------------------------------------------------------

    def test_clean_code_no_findings(self):
        code = textwrap.dedent("""\
            def add(a, b):
                return a + b
        """)
        res = self._run({"code": code})
        assert res["findings"] == []
        assert res["critical_count"] == 0
        assert res["high_count"] == 0

    # -- project_root scanning mode -----------------------------------------

    def test_project_root_scanning(self, tmp_path: Path):
        bad = tmp_path / "bad.py"
        bad.write_text('secret_key = "supersecretvalue123"\n')
        clean = tmp_path / "clean.py"
        clean.write_text("x = 1\n")
        res = self._run({"project_root": str(tmp_path)})
        assert len(res["findings"]) >= 1
        files_hit = {f["file"] for f in res["findings"]}
        assert "bad.py" in files_hit

    # -- empty / missing input ----------------------------------------------

    def test_empty_input_returns_error(self):
        res = self._run({})
        assert "error" in res


# ---------------------------------------------------------------------------
# Test Coverage Analyzer
# ---------------------------------------------------------------------------
from agents.skills.test_coverage_analyzer import (
    TestCoverageAnalyzerSkill,
    _heuristic_coverage,
)


class TestTestCoverageAnalyzer:
    """Tests for TestCoverageAnalyzerSkill."""

    def _make_project(self, tmp_path: Path) -> Path:
        """Create a minimal project with source and test files."""
        src = tmp_path / "mylib.py"
        src.write_text(textwrap.dedent("""\
            def add(a, b):
                return a + b

            def subtract(a, b):
                return a - b

            def multiply(a, b):
                return a * b
        """))
        tests = tmp_path / "test_mylib.py"
        tests.write_text(textwrap.dedent("""\
            def test_add():
                assert True

            def test_subtract():
                assert True
        """))
        return tmp_path

    def test_heuristic_coverage_basic(self, tmp_path: Path):
        root = self._make_project(tmp_path)
        res = _heuristic_coverage(root)
        assert res["method"] == "heuristic"
        assert 0 <= res["coverage_pct"] <= 100
        # multiply is not tested
        untested_names = [u["function"] for u in res["untested_functions"]]
        assert "multiply" in untested_names

    def test_untested_function_detection(self, tmp_path: Path):
        root = self._make_project(tmp_path)
        res = _heuristic_coverage(root)
        untested_names = [u["function"] for u in res["untested_functions"]]
        # add and subtract are tested, multiply is not
        assert "add" not in untested_names
        assert "subtract" not in untested_names
        assert "multiply" in untested_names

    def test_coverage_threshold_meets_target(self, tmp_path: Path):
        """All functions tested => meets_target should be True at default 80."""
        src = tmp_path / "lib.py"
        src.write_text("def foo(): pass\n")
        tst = tmp_path / "test_lib.py"
        tst.write_text("def test_foo(): pass\n")
        res = _heuristic_coverage(tmp_path)
        assert res["meets_target"] is True

    def test_coverage_threshold_below_target(self, tmp_path: Path):
        """Many untested functions => meets_target should be False."""
        src = tmp_path / "big.py"
        funcs = "\n".join(f"def func_{i}(): pass" for i in range(20))
        src.write_text(funcs + "\n")
        tst = tmp_path / "test_big.py"
        tst.write_text("def test_func_0(): pass\n")
        res = _heuristic_coverage(tmp_path)
        assert res["coverage_pct"] < 80
        assert res["meets_target"] is False

    @patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=None)
    def test_fallback_when_coverage_not_available(self, mock_cmd, tmp_path: Path):
        """When coverage.py fails (_run_cmd returns None), skill falls back to heuristic."""
        root = self._make_project(tmp_path)
        skill = TestCoverageAnalyzerSkill()
        result = skill.run({"project_root": str(root)})
        data = result.data if hasattr(result, "data") else dict(result)
        assert data["method"] == "heuristic"

    @patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=None)
    def test_incremental_mode_falls_back(self, mock_cmd, tmp_path: Path):
        """Incremental mode with no git falls back to full run."""
        root = self._make_project(tmp_path)
        skill = TestCoverageAnalyzerSkill()
        result = skill.run({"project_root": str(root), "incremental": True})
        data = result.data if hasattr(result, "data") else dict(result)
        # Should still produce a valid result via heuristic fallback
        assert "coverage_pct" in data

    def test_incremental_with_changed_files(self, tmp_path: Path):
        """Incremental with explicit changed_files but no matching test files falls back."""
        root = self._make_project(tmp_path)
        skill = TestCoverageAnalyzerSkill()
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=None):
            data = skill.run_incremental(str(root), changed_files=["nonexistent.py"])
        # No matching test files => falls back to full _run
        assert "coverage_pct" in data

    def test_empty_project(self, tmp_path: Path):
        """Empty project returns 0 coverage."""
        res = _heuristic_coverage(tmp_path)
        assert res["coverage_pct"] == 0.0
        assert res["untested_functions"] == []


# ---------------------------------------------------------------------------
# Dependency Analyzer
# ---------------------------------------------------------------------------
from agents.skills.dependency_analyzer import (
    DependencyAnalyzerSkill,
    _check_conflicts,
    _check_vulns,
    _parse_pyproject_toml,
    _parse_requirements,
)


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzerSkill."""

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        skill = DependencyAnalyzerSkill()
        result = skill.run(input_data)
        return result.data if hasattr(result, "data") else dict(result)

    # -- requirements.txt parsing -------------------------------------------

    def test_parse_pinned_requirements(self, tmp_path: Path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests==2.28.1\nflask==2.2.5\n")
        pkgs = _parse_requirements(req)
        assert len(pkgs) == 2
        assert all(p["pinned"] for p in pkgs)
        names = {p["name"] for p in pkgs}
        assert names == {"requests", "flask"}

    def test_parse_unpinned_requirements(self, tmp_path: Path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests\nnumpy\n")
        pkgs = _parse_requirements(req)
        assert len(pkgs) == 2
        assert all(p["unpinned"] for p in pkgs)

    def test_parse_range_requirements(self, tmp_path: Path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests>=2.20,<3.0\nflask~=2.2\n")
        pkgs = _parse_requirements(req)
        assert len(pkgs) == 2
        assert not any(p["pinned"] for p in pkgs)
        # Range specifiers are not considered "unpinned"
        assert not any(p["unpinned"] for p in pkgs)

    def test_parse_ignores_comments_and_flags(self, tmp_path: Path):
        req = tmp_path / "requirements.txt"
        req.write_text("# comment\n-r other.txt\ngit+https://...\nflask==2.0\n")
        pkgs = _parse_requirements(req)
        assert len(pkgs) == 1
        assert pkgs[0]["name"] == "flask"

    # -- CVE detection ------------------------------------------------------

    def test_cve_detection_known_vulnerable(self):
        pkgs = [{"name": "requests", "raw_name": "requests", "specifier": "==2.18.0", "source": "requirements.txt", "pinned": True, "unpinned": False}]
        vulns = _check_vulns(pkgs)
        assert len(vulns) == 1
        assert vulns[0]["cve"] == "CVE-2018-18074"

    def test_cve_detection_safe_package(self):
        pkgs = [{"name": "some_unknown_pkg", "raw_name": "some-unknown-pkg", "specifier": "==1.0", "source": "requirements.txt", "pinned": True, "unpinned": False}]
        vulns = _check_vulns(pkgs)
        assert vulns == []

    # -- conflict detection -------------------------------------------------

    def test_conflict_detection(self):
        pkgs = [
            {"name": "requests", "raw_name": "requests", "specifier": "==2.28.0", "source": "requirements.txt", "pinned": True, "unpinned": False},
            {"name": "requests", "raw_name": "requests", "specifier": "==2.30.0", "source": "requirements-dev.txt", "pinned": True, "unpinned": False},
        ]
        conflicts = _check_conflicts(pkgs)
        assert len(conflicts) == 1
        assert conflicts[0]["package"] == "requests"

    # -- pyproject.toml parsing ---------------------------------------------

    def test_parse_pyproject_toml(self, tmp_path: Path):
        toml = tmp_path / "pyproject.toml"
        toml.write_text(textwrap.dedent("""\
            [tool.poetry.dependencies]
            python = "^3.8"
            requests = "^2.28"
            flask = "2.2.5"
        """))
        pkgs = _parse_pyproject_toml(toml)
        names = {p["name"] for p in pkgs}
        assert "requests" in names
        assert "flask" in names
        # python should be excluded
        assert "python" not in names

    # -- empty / missing requirements ---------------------------------------

    def test_empty_requirements(self, tmp_path: Path):
        req = tmp_path / "requirements.txt"
        req.write_text("")
        pkgs = _parse_requirements(req)
        assert pkgs == []

    def test_missing_requirements_file(self, tmp_path: Path):
        pkgs = _parse_requirements(tmp_path / "nonexistent.txt")
        assert pkgs == []

    def test_full_scan_no_files(self, tmp_path: Path):
        """Scanning a project with no requirements files produces a recommendation."""
        res = self._run({"project_root": str(tmp_path)})
        assert res["packages_found"] == 0
        assert any("No requirements" in r for r in res["recommendations"])

    def test_full_scan_with_conflicts(self, tmp_path: Path):
        req1 = tmp_path / "requirements.txt"
        req1.write_text("requests==2.28.0\n")
        req2 = tmp_path / "requirements-dev.txt"
        req2.write_text("requests==2.30.0\n")
        res = self._run({"project_root": str(tmp_path)})
        assert len(res["conflicts"]) >= 1


# ---------------------------------------------------------------------------
# Multi-File Editor
# ---------------------------------------------------------------------------
from agents.skills.multi_file_editor import (
    MultiFileEditorSkill,
    _assign_priority,
    _keywords,
)


class TestMultiFileEditor:
    """Tests for MultiFileEditorSkill."""

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        skill = MultiFileEditorSkill()
        result = skill.run(input_data)
        return result.data if hasattr(result, "data") else dict(result)

    # -- keyword extraction -------------------------------------------------

    def test_keyword_extraction_basic(self):
        kws = _keywords("Rename the user authentication module")
        assert "rename" in kws
        assert "user" in kws
        assert "authentication" in kws
        assert "module" in kws
        # stopwords removed
        assert "the" not in kws

    def test_keyword_extraction_strips_stopwords(self):
        kws = _keywords("add a new feature to the system")
        assert "a" not in kws
        assert "the" not in kws
        assert "to" not in kws

    def test_keyword_extraction_empty(self):
        kws = _keywords("")
        assert kws == []

    # -- file scoring and priority ------------------------------------------

    def test_assign_priority_test_file(self):
        assert _assign_priority("tests/test_foo.py", 5, 5) == 3

    def test_assign_priority_max_score(self):
        assert _assign_priority("foo.py", 5, 5) == 1

    def test_assign_priority_lower_score(self):
        assert _assign_priority("bar.py", 2, 5) == 2

    def test_file_scoring_prefers_keyword_match(self, tmp_path: Path):
        (tmp_path / "auth.py").write_text("x = 1\n")
        (tmp_path / "utils.py").write_text("y = 2\n")
        res = self._run({"goal": "Fix auth logic", "project_root": str(tmp_path)})
        plan = res["change_plan"]
        assert len(plan) == 2
        auth_entry = next(e for e in plan if "auth" in e["file"])
        utils_entry = next(e for e in plan if "utils" in e["file"])
        assert auth_entry["priority"] <= utils_entry["priority"]

    # -- change plan generation ---------------------------------------------

    def test_change_plan_generated(self, tmp_path: Path):
        (tmp_path / "models.py").write_text("class User: pass\n")
        (tmp_path / "views.py").write_text("from models import User\n")
        res = self._run({"goal": "Rename User model", "project_root": str(tmp_path)})
        assert res["affected_count"] >= 1
        assert len(res["change_plan"]) >= 1
        for entry in res["change_plan"]:
            assert "file" in entry
            assert "priority" in entry
            assert "suggested_action" in entry

    # -- empty goal ---------------------------------------------------------

    def test_empty_goal_returns_empty_plan(self):
        res = self._run({"goal": ""})
        assert res["change_plan"] == []
        assert res["affected_count"] == 0
        assert any("goal not provided" in w for w in res["warnings"])

    # -- max_files limit ----------------------------------------------------

    def test_max_files_limit(self, tmp_path: Path):
        for i in range(10):
            (tmp_path / f"mod_{i}.py").write_text(f"x = {i}\n")
        res = self._run({
            "goal": "refactor everything",
            "project_root": str(tmp_path),
            "max_files": 3,
        })
        assert res["affected_count"] <= 3
        assert len(res["change_plan"]) <= 3

    def test_symbol_map_boosts_files(self, tmp_path: Path):
        (tmp_path / "core.py").write_text("def process(): pass\n")
        (tmp_path / "helpers.py").write_text("def helper(): pass\n")
        res = self._run({
            "goal": "Fix process function",
            "project_root": str(tmp_path),
            "symbol_map": {
                "process": {"file": "core.py", "callers": [{"file": "helpers.py"}]},
            },
        })
        plan = res["change_plan"]
        core_entry = next((e for e in plan if "core" in e["file"]), None)
        assert core_entry is not None
        assert core_entry["priority"] == 1
