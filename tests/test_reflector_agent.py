"""Unit tests for agents/reflector.py — ReflectorAgent.

Coverage targets:
- __init__ / instantiation
- run() happy path and failure outcome
- _extract_skill_learnings() — all skill-context branches
- _build_skill_summary() — all extractor branches
- _analyze_context_quality() — all context-gap signal branches
- Edge: empty input_data
"""

from __future__ import annotations

import pytest

from agents.reflector import ReflectorAgent


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def agent() -> ReflectorAgent:
    return ReflectorAgent()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInit:
    def test_instantiation(self, agent: ReflectorAgent) -> None:
        assert isinstance(agent, ReflectorAgent)

    def test_agent_name(self, agent: ReflectorAgent) -> None:
        assert agent.name == "reflect"


# ---------------------------------------------------------------------------
# run() — happy path and structural guarantees
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_returns_expected_keys(self, agent: ReflectorAgent) -> None:
        result = agent.run({})
        assert "summary" in result
        assert "learnings" in result
        assert "next_actions" in result
        assert "skill_summary" in result
        assert "pipeline_run_id" in result

    def test_run_status_appears_in_summary(self, agent: ReflectorAgent) -> None:
        result = agent.run({"verification": {"status": "success"}})
        assert "success" in result["summary"]

    def test_run_default_status_is_skip(self, agent: ReflectorAgent) -> None:
        result = agent.run({})
        assert "skip" in result["summary"]

    def test_run_no_failures_yields_empty_learnings(self, agent: ReflectorAgent) -> None:
        result = agent.run({"verification": {"status": "success", "failures": []}})
        assert result["learnings"] == []

    def test_run_failures_appear_in_learnings(self, agent: ReflectorAgent) -> None:
        failures = ["NameError: x not defined", "ImportError: no module"]
        result = agent.run({"verification": {"failures": failures}})
        combined = " ".join(result["learnings"])
        assert "NameError" in combined or "Failures" in combined

    def test_run_next_actions_passed_through(self, agent: ReflectorAgent) -> None:
        result = agent.run({"next_actions": ["step1", "step2"]})
        assert result["next_actions"] == ["step1", "step2"]

    def test_run_pipeline_run_id_passed_through(self, agent: ReflectorAgent) -> None:
        result = agent.run({"pipeline_run_id": "run-42"})
        assert result["pipeline_run_id"] == "run-42"

    def test_run_pipeline_run_id_none_when_absent(self, agent: ReflectorAgent) -> None:
        result = agent.run({})
        assert result["pipeline_run_id"] is None

    def test_run_failure_outcome_status(self, agent: ReflectorAgent) -> None:
        """Failure status surfaces in summary and learnings."""
        result = agent.run(
            {
                "verification": {
                    "status": "failure",
                    "failures": ["test assertion failed"],
                }
            }
        )
        assert "failure" in result["summary"]
        assert any("test assertion failed" in l for l in result["learnings"])

    def test_run_empty_input_does_not_raise(self, agent: ReflectorAgent) -> None:
        """run() must tolerate a completely empty input dict."""
        result = agent.run({})
        assert isinstance(result, dict)

    def test_run_skill_context_populates_skill_summary(self, agent: ReflectorAgent) -> None:
        ctx = {"security_scanner": {"critical_count": 1, "findings": ["f1"]}}
        result = agent.run({"skill_context": ctx})
        assert "security_scanner" in result["skill_summary"]

    def test_run_skill_context_findings_in_learnings(self, agent: ReflectorAgent) -> None:
        ctx = {"security_scanner": {"critical_count": 2}}
        result = agent.run({"skill_context": ctx})
        assert any("security_scanner" in l for l in result["learnings"])


# ---------------------------------------------------------------------------
# _extract_skill_learnings() — branch coverage for every skill signal
# ---------------------------------------------------------------------------


class TestExtractSkillLearnings:
    # security_scanner ---------------------------------------------------

    def test_security_critical_count_adds_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"security_scanner": {"critical_count": 3}})
        assert any("security_scanner" in l and "3 critical" in l for l in learnings)

    def test_security_zero_critical_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"security_scanner": {"critical_count": 0}})
        assert not any("security_scanner" in l for l in learnings)

    # architecture_validator ---------------------------------------------

    def test_architecture_coupling_above_threshold_adds_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"architecture_validator": {"coupling_score": 2.5}})
        assert any("architecture coupling" in l and "2.50" in l for l in learnings)

    def test_architecture_coupling_exactly_at_threshold_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"architecture_validator": {"coupling_score": 1.0}})
        assert not any("architecture coupling" in l for l in learnings)

    def test_architecture_coupling_none_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"architecture_validator": {}})
        assert not any("architecture coupling" in l for l in learnings)

    # complexity_scorer --------------------------------------------------

    def test_complexity_high_risk_count_adds_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"complexity_scorer": {"high_risk_count": 5}})
        assert any("5 high-risk" in l for l in learnings)

    def test_complexity_zero_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"complexity_scorer": {"high_risk_count": 0}})
        assert not any("complexity_scorer" in l for l in learnings)

    # test_coverage_analyzer ---------------------------------------------

    def test_coverage_below_target_adds_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"test_coverage_analyzer": {"meets_target": False, "coverage_pct": 72.3}})
        assert any("72.3%" in l for l in learnings)

    def test_coverage_meets_target_true_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"test_coverage_analyzer": {"meets_target": True, "coverage_pct": 95.0}})
        assert not any("test coverage" in l for l in learnings)

    def test_coverage_meets_target_none_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"test_coverage_analyzer": {"meets_target": None}})
        assert not any("test coverage" in l for l in learnings)

    # tech_debt_quantifier -----------------------------------------------

    def test_debt_score_above_50_adds_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"tech_debt_quantifier": {"debt_score": 75}})
        assert any("tech_debt_score 75" in l for l in learnings)

    def test_debt_score_exactly_50_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"tech_debt_quantifier": {"debt_score": 50}})
        assert not any("tech_debt_score" in l for l in learnings)

    def test_debt_score_zero_no_alert(self, agent: ReflectorAgent) -> None:
        learnings = agent._extract_skill_learnings({"tech_debt_quantifier": {"debt_score": 0}})
        assert not any("tech_debt_score" in l for l in learnings)

    # general -----------------------------------------------------------

    def test_empty_context_returns_empty_list(self, agent: ReflectorAgent) -> None:
        assert agent._extract_skill_learnings({}) == []

    def test_multiple_skills_accumulate_learnings(self, agent: ReflectorAgent) -> None:
        ctx = {
            "security_scanner": {"critical_count": 1},
            "complexity_scorer": {"high_risk_count": 2},
        }
        learnings = agent._extract_skill_learnings(ctx)
        assert len(learnings) == 2


# ---------------------------------------------------------------------------
# _build_skill_summary() — extractor mapping
# ---------------------------------------------------------------------------


class TestBuildSkillSummary:
    def test_security_scanner_extracted(self, agent: ReflectorAgent) -> None:
        ctx = {"security_scanner": {"critical_count": 2, "findings": ["f1", "f2"]}}
        summary = agent._build_skill_summary(ctx)
        assert summary["security_scanner"] == {"critical": 2, "total": 2}

    def test_architecture_validator_extracted(self, agent: ReflectorAgent) -> None:
        ctx = {
            "architecture_validator": {
                "coupling_score": 1.5,
                "circular_deps": ["A", "B"],
            }
        }
        summary = agent._build_skill_summary(ctx)
        assert summary["architecture_validator"] == {
            "coupling": 1.5,
            "circular_deps": 2,
        }

    def test_complexity_scorer_extracted(self, agent: ReflectorAgent) -> None:
        ctx = {"complexity_scorer": {"high_risk_count": 3}}
        summary = agent._build_skill_summary(ctx)
        assert summary["complexity_scorer"] == {"high_risk": 3}

    def test_test_coverage_analyzer_extracted(self, agent: ReflectorAgent) -> None:
        ctx = {"test_coverage_analyzer": {"coverage_pct": 88.0, "meets_target": True}}
        summary = agent._build_skill_summary(ctx)
        assert summary["test_coverage_analyzer"] == {
            "coverage_pct": 88.0,
            "meets_target": True,
        }

    def test_tech_debt_quantifier_extracted(self, agent: ReflectorAgent) -> None:
        ctx = {"tech_debt_quantifier": {"debt_score": 30}}
        summary = agent._build_skill_summary(ctx)
        assert summary["tech_debt_quantifier"] == {"debt_score": 30}

    def test_linter_enforcer_extracted(self, agent: ReflectorAgent) -> None:
        ctx = {"linter_enforcer": {"violations": ["v1", "v2", "v3"]}}
        summary = agent._build_skill_summary(ctx)
        assert summary["linter_enforcer"] == {"violations": 3}

    def test_empty_context_returns_empty_dict(self, agent: ReflectorAgent) -> None:
        assert agent._build_skill_summary({}) == {}

    def test_unknown_skill_omitted_from_summary(self, agent: ReflectorAgent) -> None:
        ctx = {"unknown_tool": {"data": "x"}}
        summary = agent._build_skill_summary(ctx)
        assert "unknown_tool" not in summary

    def test_security_scanner_empty_findings(self, agent: ReflectorAgent) -> None:
        ctx = {"security_scanner": {"critical_count": 0}}
        summary = agent._build_skill_summary(ctx)
        assert summary["security_scanner"] == {"critical": 0, "total": 0}

    def test_architecture_validator_no_circular_deps(self, agent: ReflectorAgent) -> None:
        ctx = {"architecture_validator": {"coupling_score": 0.5}}
        summary = agent._build_skill_summary(ctx)
        assert summary["architecture_validator"]["circular_deps"] == 0


# ---------------------------------------------------------------------------
# _analyze_context_quality() — signal pattern matching
# ---------------------------------------------------------------------------


class TestAnalyzeContextQuality:
    def test_name_error_detected(self, agent: ReflectorAgent) -> None:
        gaps = agent._analyze_context_quality(["NameError: foo is not defined"])
        assert any("NameError" in g for g in gaps)

    def test_import_error_detected(self, agent: ReflectorAgent) -> None:
        gaps = agent._analyze_context_quality(["ImportError: cannot import name x"])
        assert any("ImportError" in g for g in gaps)

    def test_module_not_found_error_detected(self, agent: ReflectorAgent) -> None:
        gaps = agent._analyze_context_quality(["ModuleNotFoundError: no module named y"])
        assert any("ModuleNotFoundError" in g for g in gaps)

    def test_attribute_error_detected(self, agent: ReflectorAgent) -> None:
        gaps = agent._analyze_context_quality(["AttributeError: obj has no attribute z"])
        assert any("AttributeError" in g for g in gaps)

    def test_not_defined_signal_detected(self, agent: ReflectorAgent) -> None:
        gaps = agent._analyze_context_quality(["'foo' is not defined"])
        assert any("not defined" in g for g in gaps)

    def test_no_matching_signal_returns_empty(self, agent: ReflectorAgent) -> None:
        gaps = agent._analyze_context_quality(["some random failure text"])
        assert gaps == []

    def test_multiple_failures_each_produce_one_gap(self, agent: ReflectorAgent) -> None:
        failures = ["NameError: x", "ImportError: y"]
        gaps = agent._analyze_context_quality(failures)
        assert len(gaps) == 2

    def test_empty_failures_returns_empty(self, agent: ReflectorAgent) -> None:
        assert agent._analyze_context_quality([]) == []

    def test_failure_matching_multiple_signals_yields_single_gap(self, agent: ReflectorAgent) -> None:
        """A single failure line that contains multiple signal keywords must
        only add one gap entry — the ``break`` inside the loop prevents doubles."""
        failures = ["NameError and ImportError both appear in this message"]
        gaps = agent._analyze_context_quality(failures)
        assert len(gaps) == 1

    def test_gaps_include_trigger_token(self, agent: ReflectorAgent) -> None:
        """Each gap string must include the matched signal token for traceability."""
        gaps = agent._analyze_context_quality(["NameError: x"])
        assert all("NameError" in g for g in gaps)
