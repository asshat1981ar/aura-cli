"""Tests for agents/skills/test_coverage_analyzer.py."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.skills.test_coverage_analyzer import (
    _run_cmd,
    _heuristic_coverage,
    TestCoverageAnalyzerSkill,
)


# ---------------------------------------------------------------------------
# _run_cmd
# ---------------------------------------------------------------------------

class TestRunCmd:
    def test_returns_string_output(self, tmp_path):
        result = _run_cmd(["echo", "hello"], tmp_path)
        assert isinstance(result, str)
        assert "hello" in result

    def test_returns_none_on_os_error(self, tmp_path):
        result = _run_cmd(["nonexistent_binary_xyz"], tmp_path)
        assert result is None

    def test_captures_stderr_too(self, tmp_path):
        result = _run_cmd(["python3", "-c", "import sys; sys.stderr.write('err')"], tmp_path)
        assert result is not None
        assert "err" in result


# ---------------------------------------------------------------------------
# _heuristic_coverage
# ---------------------------------------------------------------------------

class TestHeuristicCoverage:
    def test_empty_dir_returns_dict(self, tmp_path):
        result = _heuristic_coverage(tmp_path)
        assert isinstance(result, dict)

    def test_required_keys_present(self, tmp_path):
        result = _heuristic_coverage(tmp_path)
        for key in ("coverage_pct", "missing_files", "untested_functions", "meets_target", "method"):
            assert key in result

    def test_method_is_heuristic(self, tmp_path):
        assert _heuristic_coverage(tmp_path)["method"] == "heuristic"

    def test_no_functions_zero_pct(self, tmp_path):
        (tmp_path / "empty.py").write_text("x = 1\n")
        result = _heuristic_coverage(tmp_path)
        assert result["coverage_pct"] == 0.0

    def test_tested_function_counted(self, tmp_path):
        (tmp_path / "module.py").write_text("def helper():\n    pass\n")
        (tmp_path / "test_module.py").write_text("def test_helper():\n    pass\n")
        result = _heuristic_coverage(tmp_path)
        assert result["coverage_pct"] > 0.0

    def test_private_function_excluded_from_missing(self, tmp_path):
        (tmp_path / "mod.py").write_text("def _private():\n    pass\n")
        result = _heuristic_coverage(tmp_path)
        assert not any(u["function"] == "_private" for u in result["untested_functions"])

    def test_meets_target_false_when_low_coverage(self, tmp_path):
        (tmp_path / "big.py").write_text(
            "\n".join(f"def func_{i}():\n    pass" for i in range(20))
        )
        result = _heuristic_coverage(tmp_path)
        assert result["meets_target"] is False

    def test_coverage_pct_numeric(self, tmp_path):
        (tmp_path / "m.py").write_text("def foo():\n    pass\n")
        result = _heuristic_coverage(tmp_path)
        assert isinstance(result["coverage_pct"], float)

    def test_excludes_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("def cached_fn():\n    pass\n")
        result = _heuristic_coverage(tmp_path)
        assert all("__pycache__" not in u["file"] for u in result["untested_functions"])


# ---------------------------------------------------------------------------
# TestCoverageAnalyzerSkill._run — coverage.py path
# ---------------------------------------------------------------------------

class TestCoverageAnalyzerSkillWithCoveragePy:
    @pytest.fixture
    def skill(self):
        return TestCoverageAnalyzerSkill()

    def _make_coverage_json(self, tmp_path, pct=85.0):
        data = {
            "totals": {"percent_covered": pct},
            "files": {
                "module.py": {"summary": {"percent_covered": pct}}
            }
        }
        (tmp_path / "coverage.json").write_text(json.dumps(data))

    def test_reads_coverage_json_when_present(self, skill, tmp_path):
        self._make_coverage_json(tmp_path, 85.0)
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=""):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["coverage_pct"] == 85.0

    def test_method_is_coverage_py(self, skill, tmp_path):
        self._make_coverage_json(tmp_path, 90.0)
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=""):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["method"] == "coverage.py"

    def test_meets_target_true_above_80(self, skill, tmp_path):
        self._make_coverage_json(tmp_path, 90.0)
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=""):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["meets_target"] is True

    def test_meets_target_false_below_80(self, skill, tmp_path):
        self._make_coverage_json(tmp_path, 50.0)
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=""):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["meets_target"] is False

    def test_cleans_up_coverage_json(self, skill, tmp_path):
        self._make_coverage_json(tmp_path, 80.0)
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=""):
            skill.run({"project_root": str(tmp_path)})
        assert not (tmp_path / "coverage.json").exists()


# ---------------------------------------------------------------------------
# TestCoverageAnalyzerSkill._run — heuristic fallback
# ---------------------------------------------------------------------------

class TestCoverageAnalyzerSkillHeuristicFallback:
    @pytest.fixture
    def skill(self):
        return TestCoverageAnalyzerSkill()

    def test_falls_back_to_heuristic(self, skill, tmp_path):
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert result["method"] == "heuristic"

    def test_heuristic_result_has_coverage_pct(self, skill, tmp_path):
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=None):
            result = skill.run({"project_root": str(tmp_path)})
        assert "coverage_pct" in result


# ---------------------------------------------------------------------------
# TestCoverageAnalyzerSkill.run_incremental
# ---------------------------------------------------------------------------

class TestRunIncremental:
    @pytest.fixture
    def skill(self):
        return TestCoverageAnalyzerSkill()

    def test_no_changed_files_falls_back_to_full(self, skill, tmp_path):
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=""):
            result = skill.run_incremental(str(tmp_path), changed_files=[])
        assert "coverage_pct" in result

    def test_git_failure_falls_back_to_full(self, skill, tmp_path):
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=None):
            result = skill.run_incremental(str(tmp_path), changed_files=None)
        assert "coverage_pct" in result

    def test_changed_files_no_test_match_falls_back(self, skill, tmp_path):
        with patch("agents.skills.test_coverage_analyzer._run_cmd", return_value=None):
            result = skill.run_incremental(str(tmp_path), changed_files=["core/planner.py"])
        assert "coverage_pct" in result

    def test_incremental_flag_routes_to_run_incremental(self, skill, tmp_path):
        with patch.object(skill, "run_incremental", return_value={"coverage_pct": 75.0, "method": "incremental"}) as mock_inc:
            skill.run({"project_root": str(tmp_path), "incremental": True})
        mock_inc.assert_called_once()
