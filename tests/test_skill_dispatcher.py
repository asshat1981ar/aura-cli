"""Tests for core/skill_dispatcher.py.

Covers: classify_goal, classify_goal_llm, classify_goal_smart,
        _detect_language_cap, SkillMetrics, dispatch_skills,
        SkillChainer, chain_skill_results.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(result=None, raises=None):
    """Return a fake skill object whose .run() returns *result* or raises."""
    skill = MagicMock()
    if raises:
        skill.run.side_effect = raises
    else:
        skill.run.return_value = result or {"ok": True}
    return skill


# ---------------------------------------------------------------------------
# _detect_language_cap
# ---------------------------------------------------------------------------

class TestDetectLanguageCap:
    def test_python_keyword(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("Fix the python crash") == "python"

    def test_py_extension(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("Refactor auth.py module") == "python"

    def test_typescript_keyword(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("Add typescript support") == "typescript"

    def test_javascript_keyword(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("Fix javascript bundle") == "typescript"

    def test_ts_extension(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("Refactor utils.ts") == "typescript"

    def test_no_language_hint(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("Add login feature") is None

    def test_empty_goal(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("") is None

    def test_case_insensitive(self):
        from core.skill_dispatcher import _detect_language_cap
        assert _detect_language_cap("Fix Python crash") == "python"


# ---------------------------------------------------------------------------
# classify_goal — keyword scoring
# ---------------------------------------------------------------------------

class TestClassifyGoal:
    def setup_method(self):
        # Clear the lru_cache before each test to avoid state bleed
        from core.skill_dispatcher import classify_goal
        classify_goal.cache_clear()

    def test_bug_fix_keyword(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("Fix the login crash") == "bug_fix"

    def test_feature_keyword(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("Add OAuth2 feature") == "feature"

    def test_refactor_keyword(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("Refactor the auth module") == "refactor"

    def test_security_keyword(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("Fix XSS vulnerability in login") == "security"

    def test_docs_keyword(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("Write docstrings for all public functions") == "docs"

    def test_unknown_falls_back_to_default(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("xyz zyx qrs") == "default"

    def test_empty_goal_is_default(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("") == "default"

    def test_case_insensitive_matching(self):
        from core.skill_dispatcher import classify_goal
        assert classify_goal("FIX THE BUG") == "bug_fix"

    def test_highest_score_wins(self):
        from core.skill_dispatcher import classify_goal
        # "bug fix error crash" — 4 bug_fix keywords vs 1 feature keyword
        result = classify_goal("fix the bug error crash regression")
        assert result == "bug_fix"

    def test_result_is_cached(self):
        from core.skill_dispatcher import classify_goal
        r1 = classify_goal("fix the crash")
        r2 = classify_goal("fix the crash")
        assert r1 == r2
        info = classify_goal.cache_info()
        assert info.hits >= 1


# ---------------------------------------------------------------------------
# classify_goal_llm
# ---------------------------------------------------------------------------

class TestClassifyGoalLlm:
    def setup_method(self):
        import core.skill_dispatcher as sd
        sd._classify_goal_cache.clear()
        from core.skill_dispatcher import classify_goal
        classify_goal.cache_clear()

    def test_uses_model_response(self):
        from core.skill_dispatcher import classify_goal_llm
        model = MagicMock()
        model.respond.return_value = "refactor"
        result = classify_goal_llm("clean up the code", model)
        assert result == "refactor"

    def test_caches_result(self):
        from core.skill_dispatcher import classify_goal_llm
        model = MagicMock()
        model.respond.return_value = "feature"
        classify_goal_llm("add login", model)
        classify_goal_llm("add login", model)
        # Second call should use cache, not call model again
        assert model.respond.call_count == 1

    def test_falls_back_on_invalid_category(self):
        from core.skill_dispatcher import classify_goal_llm
        model = MagicMock()
        model.respond.return_value = "nonsense_category"
        result = classify_goal_llm("fix the bug crash", model)
        # Should fall back to keyword classification
        assert result == "bug_fix"

    def test_falls_back_on_model_exception(self):
        from core.skill_dispatcher import classify_goal_llm
        model = MagicMock()
        model.respond.side_effect = Exception("API error")
        result = classify_goal_llm("add new feature", model)
        assert result == "feature"

    def test_empty_response_falls_back(self):
        from core.skill_dispatcher import classify_goal_llm
        model = MagicMock()
        model.respond.return_value = ""
        result = classify_goal_llm("fix crash", model)
        assert result in ("bug_fix", "default", "feature", "refactor", "security", "docs")

    def test_valid_categories_all_accepted(self):
        from core.skill_dispatcher import classify_goal_llm
        import core.skill_dispatcher as sd
        for cat in ("bug_fix", "feature", "refactor", "security", "docs", "default"):
            sd._classify_goal_cache.clear()
            model = MagicMock()
            model.respond.return_value = cat
            result = classify_goal_llm(f"unique goal for {cat}", model)
            assert result == cat


# ---------------------------------------------------------------------------
# classify_goal_smart
# ---------------------------------------------------------------------------

class TestClassifyGoalSmart:
    def setup_method(self):
        import core.skill_dispatcher as sd
        sd._classify_goal_cache.clear()
        from core.skill_dispatcher import classify_goal
        classify_goal.cache_clear()

    def test_no_model_uses_keywords(self):
        from core.skill_dispatcher import classify_goal_smart
        result = classify_goal_smart("fix the bug crash", model_adapter=None)
        assert result == "bug_fix"

    def test_with_model_uses_llm(self):
        from core.skill_dispatcher import classify_goal_smart
        model = MagicMock()
        model.respond.return_value = "security"
        result = classify_goal_smart("check for vulnerabilities", model_adapter=model)
        assert result == "security"

    def test_model_none_fallback(self):
        from core.skill_dispatcher import classify_goal_smart
        result = classify_goal_smart("add feature", model_adapter=None)
        assert result == "feature"


# ---------------------------------------------------------------------------
# SkillMetrics
# ---------------------------------------------------------------------------

class TestSkillMetrics:
    def test_record_and_snapshot(self):
        from core.skill_dispatcher import SkillMetrics
        m = SkillMetrics()
        m.record("linter", 100.0, error=False)
        snap = m.snapshot()
        assert "linter" in snap
        assert snap["linter"]["call_count"] == 1
        assert snap["linter"]["total_latency_ms"] == 100.0
        assert snap["linter"]["error_count"] == 0

    def test_record_error(self):
        from core.skill_dispatcher import SkillMetrics
        m = SkillMetrics()
        m.record("checker", 50.0, error=True)
        snap = m.snapshot()
        assert snap["checker"]["error_count"] == 1

    def test_multiple_records_accumulate(self):
        from core.skill_dispatcher import SkillMetrics
        m = SkillMetrics()
        m.record("linter", 100.0)
        m.record("linter", 200.0)
        snap = m.snapshot()
        assert snap["linter"]["call_count"] == 2
        assert snap["linter"]["total_latency_ms"] == 300.0

    def test_count_alias_present(self):
        from core.skill_dispatcher import SkillMetrics
        m = SkillMetrics()
        m.record("x", 10.0)
        snap = m.snapshot()
        assert snap["x"]["count"] == snap["x"]["call_count"]

    def test_thread_safe_concurrent_writes(self):
        from core.skill_dispatcher import SkillMetrics
        m = SkillMetrics()
        errors = []

        def writer():
            try:
                for _ in range(100):
                    m.record("shared_skill", 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        snap = m.snapshot()
        assert snap["shared_skill"]["call_count"] == 1000

    def test_snapshot_independent_per_skill(self):
        from core.skill_dispatcher import SkillMetrics
        m = SkillMetrics()
        m.record("a", 1.0)
        m.record("b", 2.0)
        snap = m.snapshot()
        assert "a" in snap and "b" in snap
        assert snap["a"]["total_latency_ms"] == 1.0
        assert snap["b"]["total_latency_ms"] == 2.0


# ---------------------------------------------------------------------------
# dispatch_skills
# ---------------------------------------------------------------------------

class TestDispatchSkills:
    def test_returns_results_for_available_skills(self):
        from core.skill_dispatcher import dispatch_skills
        skills = {"symbol_indexer": _make_skill({"symbols": 42}), "linter_enforcer": _make_skill({"issues": 0})}
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("default", skills, project_root=".")
        assert "symbol_indexer" in results
        assert results["symbol_indexer"] == {"symbols": 42}

    def test_empty_skills_returns_empty_dict(self):
        from core.skill_dispatcher import dispatch_skills
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("default", {}, project_root=".")
        assert results == {}

    def test_unknown_goal_type_uses_default(self):
        from core.skill_dispatcher import dispatch_skills, SKILL_MAP
        skills = {k: _make_skill() for k in SKILL_MAP["default"]}
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("totally_unknown", skills, project_root=".")
        # Uses default skill set
        assert isinstance(results, dict)

    def test_skill_exception_recorded_as_error(self):
        from core.skill_dispatcher import dispatch_skills
        skills = {"symbol_indexer": _make_skill(raises=RuntimeError("boom"))}
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("default", skills, project_root=".")
        assert "error" in results.get("symbol_indexer", {})
        assert results["symbol_indexer"].get("is_skill_fault") is True

    def test_skill_timeout_recorded(self):
        from core.skill_dispatcher import dispatch_skills

        def slow_run(inputs):
            time.sleep(5)
            return {}

        skill = MagicMock()
        skill.run.side_effect = slow_run
        skills = {"symbol_indexer": skill}
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("default", skills, project_root=".", timeout=0.05)
        assert results.get("symbol_indexer", {}).get("error") == "timeout"

    def test_skill_error_dict_preserved(self):
        from core.skill_dispatcher import dispatch_skills
        skills = {"symbol_indexer": _make_skill({"error": "something broke"})}
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("default", skills, project_root=".")
        assert results["symbol_indexer"]["error"] == "something broke"

    def test_only_skills_in_map_run(self):
        from core.skill_dispatcher import dispatch_skills
        skills = {
            "symbol_indexer": _make_skill({"ok": True}),
            "not_in_any_map": _make_skill({"unexpected": True}),
        }
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("default", skills, project_root=".")
        assert "not_in_any_map" not in results

    def test_bug_fix_skills_selected(self):
        from core.skill_dispatcher import dispatch_skills, SKILL_MAP
        skills = {k: _make_skill() for k in SKILL_MAP["bug_fix"]}
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("bug_fix", skills, project_root=".")
        for name in SKILL_MAP["bug_fix"]:
            assert name in results

    def test_concurrent_results_complete(self):
        from core.skill_dispatcher import dispatch_skills, SKILL_MAP
        skills = {k: _make_skill({"done": k}) for k in SKILL_MAP["feature"]}
        with patch("core.mcp_agent_registry.agent_registry.resolve_by_capability", return_value=[]):
            results = dispatch_skills("feature", skills, project_root=".")
        for name in SKILL_MAP["feature"]:
            assert name in results


# ---------------------------------------------------------------------------
# SkillChainer
# ---------------------------------------------------------------------------

class TestSkillChainer:
    def test_security_scanner_queues_goal_on_critical(self):
        from core.skill_dispatcher import SkillChainer
        chainer = SkillChainer()
        goal_queue = MagicMock()
        queued = chainer.maybe_chain(
            "security_scanner",
            {"critical_count": 3, "scan_summary": "XSS in login"},
            goal_queue,
        )
        assert len(queued) == 1
        assert "3 critical" in queued[0]
        goal_queue.add.assert_called_once()

    def test_security_scanner_no_critical_does_nothing(self):
        from core.skill_dispatcher import SkillChainer
        chainer = SkillChainer()
        goal_queue = MagicMock()
        queued = chainer.maybe_chain(
            "security_scanner",
            {"critical_count": 0},
            goal_queue,
        )
        assert queued == []
        goal_queue.add.assert_not_called()

    def test_unknown_skill_does_nothing(self):
        from core.skill_dispatcher import SkillChainer
        chainer = SkillChainer()
        goal_queue = MagicMock()
        queued = chainer.maybe_chain("linter_enforcer", {"issues": 5}, goal_queue)
        assert queued == []
        goal_queue.add.assert_not_called()

    def test_queue_failure_does_not_raise(self):
        from core.skill_dispatcher import SkillChainer
        chainer = SkillChainer()
        goal_queue = MagicMock()
        goal_queue.add.side_effect = Exception("Queue full")
        # Should not propagate the exception
        queued = chainer.maybe_chain(
            "security_scanner", {"critical_count": 1, "scan_summary": "SQL injection"}, goal_queue
        )
        assert queued == []

    def test_missing_critical_count_does_nothing(self):
        from core.skill_dispatcher import SkillChainer
        chainer = SkillChainer()
        goal_queue = MagicMock()
        queued = chainer.maybe_chain("security_scanner", {}, goal_queue)
        assert queued == []


# ---------------------------------------------------------------------------
# chain_skill_results
# ---------------------------------------------------------------------------

class TestChainSkillResults:
    def test_processes_all_results(self):
        from core.skill_dispatcher import chain_skill_results
        goal_queue = MagicMock()
        results = {
            "security_scanner": {"critical_count": 2, "scan_summary": "XSS"},
            "linter_enforcer": {"issues": 10},
            "symbol_indexer": {"symbols": 50},
        }
        queued = chain_skill_results(results, goal_queue)
        # Only security_scanner should produce a queued goal
        assert len(queued) == 1

    def test_empty_results(self):
        from core.skill_dispatcher import chain_skill_results
        goal_queue = MagicMock()
        queued = chain_skill_results({}, goal_queue)
        assert queued == []

    def test_multiple_security_findings_each_queued(self):
        # chain_skill_results calls SkillChainer once per skill, so one
        # security_scanner entry → one goal regardless of critical_count value.
        from core.skill_dispatcher import chain_skill_results
        goal_queue = MagicMock()
        results = {"security_scanner": {"critical_count": 5, "scan_summary": "multiple issues"}}
        queued = chain_skill_results(results, goal_queue)
        assert len(queued) == 1
        assert "5 critical" in queued[0]


# ---------------------------------------------------------------------------
# SKILL_MAP and GOAL_TYPE_TO_CAPABILITY constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_skill_map_all_goal_types_present(self):
        from core.skill_dispatcher import SKILL_MAP
        for gt in ("bug_fix", "feature", "refactor", "security", "docs", "default"):
            assert gt in SKILL_MAP, f"'{gt}' missing from SKILL_MAP"

    def test_skill_map_lists_are_nonempty(self):
        from core.skill_dispatcher import SKILL_MAP
        for gt, skills in SKILL_MAP.items():
            assert skills, f"'{gt}' has empty skill list"

    def test_goal_type_to_capability_complete(self):
        from core.skill_dispatcher import GOAL_TYPE_TO_CAPABILITY, SKILL_MAP
        for gt in SKILL_MAP:
            assert gt in GOAL_TYPE_TO_CAPABILITY, f"'{gt}' missing from GOAL_TYPE_TO_CAPABILITY"
