"""Tests for agents/multi_agent_workflow.py."""

import re
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from agents.multi_agent_workflow import (
    _perform_fallback_validation,
    classify_failure_modes,
    compile_summary,
    engage_stakeholders,
    predict_failure_modes,
    suggest_architectural_improvements,
    validate_with_mcp_server,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_results(
    *,
    lint_errors: List[str] | None = None,
    type_errors: str = "",
    complexity_score: float = 0,
    coverage_pct: float = 100,
) -> Dict[str, Any]:
    """Build an architecture_results dict with the requested values."""
    return {
        "lint_results": {"errors": lint_errors or []},
        "type_check_results": {"errors": type_errors},
        "complexity": {"score": complexity_score},
        "coverage": {"percentage": coverage_pct},
    }


def _skill_unavailable(skill_name: str) -> Dict[str, Any]:
    return {"status": "skill_not_available", "skill": skill_name}


# ---------------------------------------------------------------------------
# TestPredictFailureModes
# ---------------------------------------------------------------------------


class TestPredictFailureModes:
    """Tests for predict_failure_modes()."""

    @patch("agents.multi_agent_workflow.log_json")
    def test_lint_errors_produce_failure_mode(self, mock_log):
        results = _clean_results(lint_errors=["E001 bad import"])
        modes = predict_failure_modes(results)
        failure_types = [m["failure_type"] for m in modes]
        assert "Runtime Error" in failure_types

    @patch("agents.multi_agent_workflow.log_json")
    def test_high_lint_errors_are_high_severity(self, mock_log):
        # 11 errors → severity "high"
        results = _clean_results(lint_errors=[f"E{i}" for i in range(11)])
        modes = predict_failure_modes(results)
        runtime_mode = next(m for m in modes if m["failure_type"] == "Runtime Error")
        assert runtime_mode["severity"] == "high"

    @patch("agents.multi_agent_workflow.log_json")
    def test_low_lint_errors_are_medium_severity(self, mock_log):
        # 3 errors (<=10) → severity "medium"
        results = _clean_results(lint_errors=["E1", "E2", "E3"])
        modes = predict_failure_modes(results)
        runtime_mode = next(m for m in modes if m["failure_type"] == "Runtime Error")
        assert runtime_mode["severity"] == "medium"

    @patch("agents.multi_agent_workflow.log_json")
    def test_exactly_ten_lint_errors_are_medium_severity(self, mock_log):
        results = _clean_results(lint_errors=[f"E{i}" for i in range(10)])
        modes = predict_failure_modes(results)
        runtime_mode = next(m for m in modes if m["failure_type"] == "Runtime Error")
        assert runtime_mode["severity"] == "medium"

    @patch("agents.multi_agent_workflow.log_json")
    def test_type_errors_produce_failure_mode(self, mock_log):
        results = _clean_results(type_errors="error: incompatible types")
        modes = predict_failure_modes(results)
        failure_types = [m["failure_type"] for m in modes]
        assert "Type Mismatch" in failure_types

    @patch("agents.multi_agent_workflow.log_json")
    def test_type_error_severity_is_high(self, mock_log):
        results = _clean_results(type_errors="error: something")
        modes = predict_failure_modes(results)
        type_mode = next(m for m in modes if m["failure_type"] == "Type Mismatch")
        assert type_mode["severity"] == "high"

    @patch("agents.multi_agent_workflow.log_json")
    def test_high_complexity_produces_failure_mode(self, mock_log):
        # complexity score 8 > 7 → Technical Debt Accumulation
        results = _clean_results(complexity_score=8)
        modes = predict_failure_modes(results)
        failure_types = [m["failure_type"] for m in modes]
        assert "Technical Debt Accumulation" in failure_types

    @patch("agents.multi_agent_workflow.log_json")
    def test_low_complexity_does_not_produce_failure_mode(self, mock_log):
        results = _clean_results(complexity_score=5)
        modes = predict_failure_modes(results)
        failure_types = [m["failure_type"] for m in modes]
        assert "Technical Debt Accumulation" not in failure_types

    @patch("agents.multi_agent_workflow.log_json")
    def test_low_coverage_under_50_produces_high_regression(self, mock_log):
        # coverage 40% (<50) → Regression, high severity
        results = _clean_results(coverage_pct=40)
        modes = predict_failure_modes(results)
        regression = next(m for m in modes if m["failure_type"] == "Regression")
        assert regression["severity"] == "high"

    @patch("agents.multi_agent_workflow.log_json")
    def test_medium_coverage_50_to_70_produces_medium_regression(self, mock_log):
        # coverage 60% (>=50 but <70) → Regression, medium severity
        results = _clean_results(coverage_pct=60)
        modes = predict_failure_modes(results)
        regression = next(m for m in modes if m["failure_type"] == "Regression")
        assert regression["severity"] == "medium"

    @patch("agents.multi_agent_workflow.log_json")
    def test_coverage_at_70_does_not_produce_regression(self, mock_log):
        results = _clean_results(coverage_pct=70)
        modes = predict_failure_modes(results)
        failure_types = [m["failure_type"] for m in modes]
        assert "Regression" not in failure_types

    @patch("agents.multi_agent_workflow.log_json")
    def test_missing_skills_produce_capability_gap(self, mock_log):
        results = {
            "lint_results": _skill_unavailable("pylint"),
            "type_check_results": {},
            "complexity": {},
            "coverage": {},
        }
        modes = predict_failure_modes(results)
        failure_types = [m["failure_type"] for m in modes]
        assert "Capability Gap" in failure_types

    @patch("agents.multi_agent_workflow.log_json")
    def test_missing_skill_severity_is_low(self, mock_log):
        results = {
            "lint_results": _skill_unavailable("pylint"),
            "type_check_results": {},
            "complexity": {},
            "coverage": {},
        }
        modes = predict_failure_modes(results)
        cap_gap = next(m for m in modes if m["failure_type"] == "Capability Gap")
        assert cap_gap["severity"] == "low"

    @patch("agents.multi_agent_workflow.log_json")
    def test_clean_results_produce_no_failures(self, mock_log):
        results = _clean_results()
        modes = predict_failure_modes(results)
        assert modes == []

    @patch("agents.multi_agent_workflow.log_json")
    def test_empty_input(self, mock_log):
        modes = predict_failure_modes({})
        assert modes == []

    @patch("agents.multi_agent_workflow.log_json")
    def test_each_failure_mode_has_required_keys(self, mock_log):
        results = _clean_results(lint_errors=["E001"], type_errors="err", complexity_score=9, coverage_pct=30)
        modes = predict_failure_modes(results)
        required_keys = {"component", "failure_type", "description", "triggering_conditions", "severity"}
        for mode in modes:
            assert required_keys.issubset(mode.keys()), f"Missing keys in {mode}"

    @patch("agents.multi_agent_workflow.log_json")
    def test_log_json_called_once(self, mock_log):
        predict_failure_modes({})
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# TestClassifyFailureModes
# ---------------------------------------------------------------------------


class TestClassifyFailureModes:
    """Tests for classify_failure_modes()."""

    @patch("agents.multi_agent_workflow.log_json")
    def test_empty_input_returns_empty_list(self, mock_log):
        assert classify_failure_modes([]) == []

    @patch("agents.multi_agent_workflow.log_json")
    def test_adds_likelihood_and_risk_level_keys(self, mock_log):
        modes = [{"component": "Code Quality", "failure_type": "Runtime Error", "severity": "high"}]
        classified = classify_failure_modes(modes)
        assert len(classified) == 1
        assert "likelihood" in classified[0]
        assert "risk_level" in classified[0]

    @patch("agents.multi_agent_workflow.log_json")
    def test_adds_priority_key(self, mock_log):
        modes = [{"component": "Code Quality", "failure_type": "Runtime Error", "severity": "high"}]
        classified = classify_failure_modes(modes)
        assert "priority" in classified[0]

    @patch("agents.multi_agent_workflow.log_json")
    def test_high_severity_code_quality_is_critical(self, mock_log):
        # Code Quality → likelihood "high"; severity "high" + likelihood "high" → Critical
        modes = [{"component": "Code Quality", "failure_type": "Runtime Error", "severity": "high"}]
        classified = classify_failure_modes(modes)
        assert classified[0]["risk_level"] == "Critical"

    @patch("agents.multi_agent_workflow.log_json")
    def test_medium_severity_type_system_classified_as_medium(self, mock_log):
        # Type System → likelihood "medium"; severity "medium" + likelihood "medium" → Medium
        modes = [{"component": "Type System", "failure_type": "Type Mismatch", "severity": "medium"}]
        classified = classify_failure_modes(modes)
        assert classified[0]["risk_level"] == "Medium"

    @patch("agents.multi_agent_workflow.log_json")
    def test_low_severity_operational_classified_as_low(self, mock_log):
        # Operational → likelihood "low"; severity "low" + likelihood "low" → Low
        modes = [{"component": "Operational", "failure_type": "Capability Gap", "severity": "low"}]
        classified = classify_failure_modes(modes)
        assert classified[0]["risk_level"] == "Low"

    @patch("agents.multi_agent_workflow.log_json")
    def test_high_severity_test_coverage_is_high_or_critical(self, mock_log):
        # Test Coverage → likelihood "high"; high severity → Critical
        modes = [{"component": "Test Coverage", "failure_type": "Regression", "severity": "high"}]
        classified = classify_failure_modes(modes)
        assert classified[0]["risk_level"] in ("High", "Critical")

    @patch("agents.multi_agent_workflow.log_json")
    def test_sorted_by_priority_descending(self, mock_log):
        modes = [
            {"component": "Operational", "failure_type": "Capability Gap", "severity": "low"},
            {"component": "Code Quality", "failure_type": "Runtime Error", "severity": "high"},
            {"component": "Type System", "failure_type": "Type Mismatch", "severity": "medium"},
        ]
        classified = classify_failure_modes(modes)
        priorities = [m["priority"] for m in classified]
        assert priorities == sorted(priorities, reverse=True)

    @patch("agents.multi_agent_workflow.log_json")
    def test_unknown_component_gets_medium_likelihood(self, mock_log):
        modes = [{"component": "Unknown", "failure_type": "SomeError", "severity": "medium"}]
        classified = classify_failure_modes(modes)
        assert classified[0]["likelihood"] == "medium"

    @patch("agents.multi_agent_workflow.log_json")
    def test_original_fields_preserved(self, mock_log):
        modes = [{"component": "Code Quality", "failure_type": "Runtime Error", "severity": "high", "description": "d"}]
        classified = classify_failure_modes(modes)
        assert classified[0]["description"] == "d"
        assert classified[0]["component"] == "Code Quality"

    @patch("agents.multi_agent_workflow.log_json")
    def test_log_json_called(self, mock_log):
        classify_failure_modes([])
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# TestSuggestImprovements
# ---------------------------------------------------------------------------


class TestSuggestImprovements:
    """Tests for suggest_architectural_improvements()."""

    _REQUIRED_FIELDS = {"target_component", "improvement_type", "description", "expected_benefit", "implementation_effort"}

    @patch("agents.multi_agent_workflow.log_json")
    def test_always_includes_documentation_suggestion(self, mock_log):
        # Even with perfectly clean results, the Documentation suggestion is always appended
        suggestions = suggest_architectural_improvements(_clean_results())
        target_components = [s["target_component"] for s in suggestions]
        assert "Documentation" in target_components

    @patch("agents.multi_agent_workflow.log_json")
    def test_clean_results_produce_only_docs_suggestion(self, mock_log):
        suggestions = suggest_architectural_improvements(_clean_results())
        # Only the always-added Documentation suggestion
        assert len(suggestions) == 1
        assert suggestions[0]["target_component"] == "Documentation"

    @patch("agents.multi_agent_workflow.log_json")
    def test_lint_errors_over_5_add_code_quality(self, mock_log):
        results = _clean_results(lint_errors=[f"E{i}" for i in range(6)])
        suggestions = suggest_architectural_improvements(results)
        components = [s["target_component"] for s in suggestions]
        assert "Code Quality" in components

    @patch("agents.multi_agent_workflow.log_json")
    def test_lint_errors_5_or_fewer_do_not_add_code_quality(self, mock_log):
        results = _clean_results(lint_errors=["E1", "E2", "E3", "E4", "E5"])
        suggestions = suggest_architectural_improvements(results)
        components = [s["target_component"] for s in suggestions]
        assert "Code Quality" not in components

    @patch("agents.multi_agent_workflow.log_json")
    def test_type_errors_add_type_system(self, mock_log):
        results = _clean_results(type_errors="error: bad type")
        suggestions = suggest_architectural_improvements(results)
        components = [s["target_component"] for s in suggestions]
        assert "Type System" in components

    @patch("agents.multi_agent_workflow.log_json")
    def test_high_complexity_adds_maintainability(self, mock_log):
        results = _clean_results(complexity_score=9)
        suggestions = suggest_architectural_improvements(results)
        components = [s["target_component"] for s in suggestions]
        assert "Maintainability" in components

    @patch("agents.multi_agent_workflow.log_json")
    def test_low_coverage_adds_test_coverage_suggestion(self, mock_log):
        results = _clean_results(coverage_pct=60)
        suggestions = suggest_architectural_improvements(results)
        components = [s["target_component"] for s in suggestions]
        assert "Test Coverage" in components

    @patch("agents.multi_agent_workflow.log_json")
    def test_coverage_at_80_does_not_add_test_coverage(self, mock_log):
        results = _clean_results(coverage_pct=80)
        suggestions = suggest_architectural_improvements(results)
        components = [s["target_component"] for s in suggestions]
        assert "Test Coverage" not in components

    @patch("agents.multi_agent_workflow.log_json")
    def test_missing_skills_add_toolchain_suggestion(self, mock_log):
        results = {
            "lint_results": _skill_unavailable("pylint"),
            "type_check_results": {},
            "complexity": {},
            "coverage": {},
        }
        suggestions = suggest_architectural_improvements(results)
        components = [s["target_component"] for s in suggestions]
        assert "Toolchain" in components

    @patch("agents.multi_agent_workflow.log_json")
    def test_each_suggestion_has_required_fields(self, mock_log):
        results = _clean_results(lint_errors=[f"E{i}" for i in range(7)], type_errors="err", complexity_score=9, coverage_pct=40)
        suggestions = suggest_architectural_improvements(results)
        for suggestion in suggestions:
            assert self._REQUIRED_FIELDS.issubset(suggestion.keys()), f"Missing fields in {suggestion}"

    @patch("agents.multi_agent_workflow.log_json")
    def test_empty_input(self, mock_log):
        suggestions = suggest_architectural_improvements({})
        # Only the Documentation suggestion
        assert len(suggestions) == 1

    @patch("agents.multi_agent_workflow.log_json")
    def test_log_json_called(self, mock_log):
        suggest_architectural_improvements({})
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# TestCompileSummary
# ---------------------------------------------------------------------------


class TestCompileSummary:
    """Tests for compile_summary()."""

    def _make_summary(
        self,
        python_results=None,
        typescript_results=None,
        execution_plan=None,
        classified_failure_modes=None,
        architectural_suggestions=None,
    ) -> str:
        return compile_summary(
            python_results or {},
            typescript_results or {},
            execution_plan or {},
            classified_failure_modes or [],
            architectural_suggestions or [],
        )

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_string(self, mock_log):
        result = self._make_summary()
        assert isinstance(result, str)

    @patch("agents.multi_agent_workflow.log_json")
    def test_starts_with_markdown_heading(self, mock_log):
        result = self._make_summary()
        assert result.startswith("#")

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_executive_summary_section(self, mock_log):
        # The title itself and table of contents appear; check the top-level heading
        result = self._make_summary()
        assert "Architecture Analysis Summary" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_python_analysis_section(self, mock_log):
        result = self._make_summary()
        assert "Python Analysis Results" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_typescript_analysis_section(self, mock_log):
        result = self._make_summary()
        assert "TypeScript Analysis Results" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_execution_plan_section(self, mock_log):
        result = self._make_summary()
        assert "Execution Plan" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_failure_modes_section(self, mock_log):
        result = self._make_summary()
        assert "Predicted Failure Modes" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_improvements_section(self, mock_log):
        result = self._make_summary()
        assert "Architectural Improvement Suggestions" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_recommendations_section(self, mock_log):
        result = self._make_summary()
        assert "Recommendations" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_timestamp(self, mock_log):
        result = self._make_summary()
        # Timestamp format: YYYY-MM-DD HH:MM:SS UTC
        assert re.search(r"\d{4}-\d{2}-\d{2}", result), "No date string found in output"

    @patch("agents.multi_agent_workflow.log_json")
    def test_contains_utc_label(self, mock_log):
        result = self._make_summary()
        assert "UTC" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_handles_empty_inputs_produces_valid_markdown(self, mock_log):
        result = self._make_summary()
        # Must still be a non-empty string with headings
        assert len(result) > 100
        assert "##" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_failure_modes_table_rendered_when_present(self, mock_log):
        failure_modes = [
            {
                "component": "Code Quality",
                "failure_type": "Runtime Error",
                "risk_level": "Critical",
                "likelihood": "high",
                "description": "Some long description text here",
            }
        ]
        result = self._make_summary(classified_failure_modes=failure_modes)
        # Markdown table pipes
        assert "| Component |" in result
        assert "Code Quality" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_no_failure_modes_shows_placeholder(self, mock_log):
        result = self._make_summary()
        assert "No failure modes predicted." in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_architectural_suggestions_rendered(self, mock_log):
        suggestions = [
            {
                "target_component": "Test Coverage",
                "improvement_type": "Test Suite Expansion",
                "description": "Expand the test suite",
                "expected_benefit": "Less regression risk",
                "implementation_effort": "medium",
            }
        ]
        result = self._make_summary(architectural_suggestions=suggestions)
        assert "Test Suite Expansion" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_no_suggestions_shows_placeholder(self, mock_log):
        result = self._make_summary()
        assert "No improvement suggestions generated." in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_high_severity_failure_mode_triggers_recommendation(self, mock_log):
        failure_modes = [
            {
                "component": "Code Quality",
                "failure_type": "Runtime Error",
                "risk_level": "Critical",
                "likelihood": "high",
                "severity": "high",
                "description": "A very long description that goes beyond 50 chars to test truncation",
            }
        ]
        result = self._make_summary(classified_failure_modes=failure_modes)
        assert "high-severity failure modes" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_footer_present(self, mock_log):
        result = self._make_summary()
        assert "AURA Multi-Agent Workflow" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_python_lint_errors_shown(self, mock_log):
        python_results = {"lint_results": {"errors": ["E001 bad import", "W002 unused var"]}}
        result = self._make_summary(python_results=python_results)
        assert "Issues found: 2" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_skill_not_available_shown_in_output(self, mock_log):
        python_results = {"lint_results": _skill_unavailable("pylint")}
        result = self._make_summary(python_results=python_results)
        assert "not available" in result


# ---------------------------------------------------------------------------
# TestFallbackValidation
# ---------------------------------------------------------------------------


class TestFallbackValidation:
    """Tests for _perform_fallback_validation()."""

    def _base_partial(self) -> Dict[str, Any]:
        return {
            "status": "fallback",
            "score": 0.0,
            "feedback": [],
            "errors": [],
            "timestamp": "2026-01-01T00:00:00",
        }

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_dict(self, mock_log):
        result = _perform_fallback_validation("# Hello\n## Section\n", self._base_partial())
        assert isinstance(result, dict)

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_dict_with_expected_keys(self, mock_log):
        result = _perform_fallback_validation("# Hello\n", self._base_partial())
        assert "score" in result
        assert "feedback" in result
        assert "status" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_score_is_float_between_0_and_1(self, mock_log):
        result = _perform_fallback_validation("x" * 2000 + "\n##\n|\nRecommendations", self._base_partial())
        assert 0.0 <= result["score"] <= 1.0

    @patch("agents.multi_agent_workflow.log_json")
    def test_score_increases_with_length_over_1000(self, mock_log):
        short = _perform_fallback_validation("short", self._base_partial())
        long_content = "x" * 1001
        long = _perform_fallback_validation(long_content, self._base_partial())
        assert long["score"] > short["score"]

    @patch("agents.multi_agent_workflow.log_json")
    def test_score_increases_with_headers(self, mock_log):
        no_headers = _perform_fallback_validation("no headers here", self._base_partial())
        with_headers = _perform_fallback_validation("## Section One", self._base_partial())
        assert with_headers["score"] > no_headers["score"]

    @patch("agents.multi_agent_workflow.log_json")
    def test_score_increases_with_tables(self, mock_log):
        no_table = _perform_fallback_validation("no table", self._base_partial())
        with_table = _perform_fallback_validation("| col1 | col2 |", self._base_partial())
        assert with_table["score"] > no_table["score"]

    @patch("agents.multi_agent_workflow.log_json")
    def test_score_increases_with_recommendations_section(self, mock_log):
        no_rec = _perform_fallback_validation("nothing", self._base_partial())
        with_rec = _perform_fallback_validation("Recommendations here", self._base_partial())
        assert with_rec["score"] > no_rec["score"]

    @patch("agents.multi_agent_workflow.log_json")
    def test_feedback_list_is_populated(self, mock_log):
        result = _perform_fallback_validation("## Header\n", self._base_partial())
        assert isinstance(result["feedback"], list)
        assert len(result["feedback"]) > 0

    @patch("agents.multi_agent_workflow.log_json")
    def test_feedback_contains_content_length(self, mock_log):
        content = "hello world"
        result = _perform_fallback_validation(content, self._base_partial())
        feedback_str = " ".join(result["feedback"])
        assert str(len(content)) in feedback_str

    @patch("agents.multi_agent_workflow.log_json")
    def test_feedback_mentions_fallback(self, mock_log):
        result = _perform_fallback_validation("anything", self._base_partial())
        feedback_str = " ".join(result["feedback"])
        assert "fallback" in feedback_str.lower()

    @patch("agents.multi_agent_workflow.log_json")
    def test_score_capped_at_1(self, mock_log):
        # Provide all positive signals: long content, headers, tables, recommendations
        content = "x" * 2000 + "\n## Section\n| col |\nRecommendations"
        result = _perform_fallback_validation(content, self._base_partial())
        assert result["score"] <= 1.0

    @patch("agents.multi_agent_workflow.log_json")
    def test_log_json_called(self, mock_log):
        _perform_fallback_validation("some content", self._base_partial())
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# TestValidateWithMcpServer
# ---------------------------------------------------------------------------


class TestValidateWithMcpServer:
    """Tests for validate_with_mcp_server() — focuses on the fallback path."""

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_dict_on_import_failure(self, mock_log):
        # core.mcp_client does not exist; ImportError forces fallback path
        result = validate_with_mcp_server("# Summary\n## Failure Modes\n")
        assert isinstance(result, dict)

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_score_key(self, mock_log):
        result = validate_with_mcp_server("## Header\n")
        assert "score" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_feedback_key(self, mock_log):
        result = validate_with_mcp_server("## Header\n")
        assert "feedback" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_status_key(self, mock_log):
        result = validate_with_mcp_server("## Header\n")
        assert "status" in result

    @patch("agents.multi_agent_workflow.log_json")
    def test_status_is_fallback_when_mcp_unavailable(self, mock_log):
        result = validate_with_mcp_server("## Header\n")
        # Since MCP server is not running in tests, status must be fallback
        assert result["status"] == "fallback"

    @patch("agents.multi_agent_workflow.log_json")
    def test_score_is_numeric(self, mock_log):
        result = validate_with_mcp_server("content")
        assert isinstance(result["score"], float)


# ---------------------------------------------------------------------------
# TestEngageStakeholders
# ---------------------------------------------------------------------------


class TestEngageStakeholders:
    """Tests for engage_stakeholders()."""

    def _make_validation_result(self, score: float = 0.9, status: str = "validated") -> Dict[str, Any]:
        return {
            "status": status,
            "score": score,
            "feedback": ["All good", "Well structured"],
            "errors": [],
        }

    @patch("agents.multi_agent_workflow.log_json")
    def test_returns_none(self, mock_log):
        result = engage_stakeholders(self._make_validation_result())
        assert result is None

    @patch("agents.multi_agent_workflow.log_json")
    def test_log_json_called(self, mock_log):
        engage_stakeholders(self._make_validation_result())
        assert mock_log.called

    @patch("agents.multi_agent_workflow.log_json")
    def test_log_json_called_at_least_twice(self, mock_log):
        # engage_stakeholders calls log_json at least twice (started + action)
        engage_stakeholders(self._make_validation_result())
        assert mock_log.call_count >= 2

    @patch("agents.multi_agent_workflow.log_json")
    def test_logs_engagement_started_event(self, mock_log):
        engage_stakeholders(self._make_validation_result())
        events = [call.args[1] for call in mock_log.call_args_list]
        assert "stakeholder_engagement_started" in events

    @patch("agents.multi_agent_workflow.log_json")
    def test_logs_engagement_action_event(self, mock_log):
        engage_stakeholders(self._make_validation_result())
        events = [call.args[1] for call in mock_log.call_args_list]
        assert "stakeholder_engagement_action" in events

    @patch("agents.multi_agent_workflow.log_json")
    def test_high_score_logs_inform_level(self, mock_log):
        engage_stakeholders(self._make_validation_result(score=0.85))
        # Find the stakeholder_engagement_action call
        action_call = next(call for call in mock_log.call_args_list if call.args[1] == "stakeholder_engagement_action")
        payload = action_call.args[2]
        assert payload["level"] == "inform"

    @patch("agents.multi_agent_workflow.log_json")
    def test_medium_score_logs_review_level(self, mock_log):
        engage_stakeholders(self._make_validation_result(score=0.6))
        action_call = next(call for call in mock_log.call_args_list if call.args[1] == "stakeholder_engagement_action")
        payload = action_call.args[2]
        assert payload["level"] == "review"

    @patch("agents.multi_agent_workflow.log_json")
    def test_low_score_logs_block_level(self, mock_log):
        engage_stakeholders(self._make_validation_result(score=0.3))
        action_call = next(call for call in mock_log.call_args_list if call.args[1] == "stakeholder_engagement_action")
        payload = action_call.args[2]
        assert payload["level"] == "block"

    @patch("agents.multi_agent_workflow.log_json")
    def test_notification_prepared_logged(self, mock_log):
        engage_stakeholders(self._make_validation_result())
        events = [call.args[1] for call in mock_log.call_args_list]
        assert "stakeholder_notification_prepared" in events

    @patch("agents.multi_agent_workflow.log_json")
    def test_does_not_raise_with_empty_validation_result(self, mock_log):
        engage_stakeholders({})  # Should not raise
