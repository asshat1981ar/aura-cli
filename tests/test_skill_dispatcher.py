"""Tests for SKILL_MAP and classify_goal in core/skill_dispatcher.py."""
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import core.skill_dispatcher as skill_dispatcher
from core.skill_dispatcher import (
    SKILL_MAP,
    _GOAL_TYPE_HINTS,
    classify_goal,
    dispatch_skills,
)


# ---------------------------------------------------------------------------
# SKILL_MAP — structural assertions
# ---------------------------------------------------------------------------

class TestSkillMapContents:
    """Verify new skills appear in the right goal-type buckets."""

    def test_lint_in_bug_fix(self):
        assert "lint" in SKILL_MAP["bug_fix"]

    def test_test_and_observe_in_bug_fix(self):
        assert "test_and_observe" in SKILL_MAP["bug_fix"]

    def test_lint_in_feature(self):
        assert "lint" in SKILL_MAP["feature"]

    def test_lint_in_refactor(self):
        assert "lint" in SKILL_MAP["refactor"]

    def test_test_and_observe_in_refactor(self):
        assert "test_and_observe" in SKILL_MAP["refactor"]

    def test_lint_in_security(self):
        assert "lint" in SKILL_MAP["security"]

    def test_lint_in_default(self):
        assert "lint" in SKILL_MAP["default"]

    def test_docs_unchanged(self):
        """docs is low-noise; neither new skill should appear there."""
        assert "lint" not in SKILL_MAP["docs"]
        assert "test_and_observe" not in SKILL_MAP["docs"]

    def test_no_duplicates_in_any_category(self):
        for goal_type, skills in SKILL_MAP.items():
            assert len(skills) == len(set(skills)), (
                f"Duplicate skill found in SKILL_MAP[{goal_type!r}]"
            )

    def test_test_and_observe_before_lint_in_bug_fix(self):
        """test_and_observe should run before lint (higher diagnostic value)."""
        skills = SKILL_MAP["bug_fix"]
        assert skills.index("test_and_observe") < skills.index("lint")

    def test_all_goal_types_present(self):
        expected = {"bug_fix", "feature", "refactor", "security", "docs", "default"}
        assert set(SKILL_MAP.keys()) == expected


# ---------------------------------------------------------------------------
# _GOAL_TYPE_HINTS — keyword additions
# ---------------------------------------------------------------------------

class TestGoalTypeHints:
    def test_observe_in_bug_fix_hints(self):
        assert "observe" in _GOAL_TYPE_HINTS["bug_fix"]

    def test_diagnose_in_bug_fix_hints(self):
        assert "diagnose" in _GOAL_TYPE_HINTS["bug_fix"]

    def test_existing_bug_fix_keywords_preserved(self):
        existing = {"fix", "bug", "error", "crash", "traceback", "exception"}
        for kw in existing:
            assert kw in _GOAL_TYPE_HINTS["bug_fix"], f"Keyword {kw!r} was removed"

    def test_no_cross_contamination_in_other_types(self):
        """New keywords are only in bug_fix; other types unchanged."""
        for gt in ("feature", "refactor", "security", "docs"):
            assert "observe" not in _GOAL_TYPE_HINTS[gt]
            assert "diagnose" not in _GOAL_TYPE_HINTS[gt]


# ---------------------------------------------------------------------------
# classify_goal — routing with new keywords
# ---------------------------------------------------------------------------

class TestClassifyGoalRouting:
    def setup_method(self):
        # Clear LRU cache so each test gets a fresh classification
        classify_goal.cache_clear()

    def test_observe_routes_to_bug_fix(self):
        assert classify_goal("observe and trace the memory leak") == "bug_fix"

    def test_diagnose_routes_to_bug_fix(self):
        assert classify_goal("diagnose the startup crash") == "bug_fix"

    def test_fix_still_routes_to_bug_fix(self):
        assert classify_goal("fix the broken login handler") == "bug_fix"

    def test_refactor_goal_unchanged(self):
        assert classify_goal("refactor the auth module") == "refactor"

    def test_feature_goal_unchanged(self):
        assert classify_goal("add a new export endpoint") == "feature"

    def test_security_goal_unchanged(self):
        assert classify_goal("fix the sql injection vulnerability") == "security"

    def test_unknown_goal_defaults(self):
        assert classify_goal("xyzzy frobnicate blargle") == "default"


# ---------------------------------------------------------------------------
# dispatch_skills — new skills are selected when present
# ---------------------------------------------------------------------------

class TestDispatchSkillsNewSkills:
    def setup_method(self):
        skill_dispatcher._SKILL_RESULT_CACHE.clear()

    def _make_skill(self, name: str, result: dict = None) -> MagicMock:
        skill = MagicMock()
        skill.name = name
        skill.run.return_value = result or {"status": "ok", "skill": name}
        return skill

    def test_lint_runs_for_bug_fix(self):
        skills = {
            "lint": self._make_skill("lint"),
            "symbol_indexer": self._make_skill("symbol_indexer"),
        }
        results = dispatch_skills("bug_fix", skills, project_root=".")
        assert "lint" in results
        assert results["lint"]["status"] == "ok"

    def test_test_and_observe_runs_for_bug_fix(self):
        skills = {
            "test_and_observe": self._make_skill("test_and_observe"),
        }
        results = dispatch_skills("bug_fix", skills, project_root=".")
        assert "test_and_observe" in results

    def test_lint_runs_for_refactor(self):
        skills = {"lint": self._make_skill("lint")}
        results = dispatch_skills("refactor", skills, project_root=".")
        assert "lint" in results

    def test_test_and_observe_runs_for_refactor(self):
        skills = {"test_and_observe": self._make_skill("test_and_observe")}
        results = dispatch_skills("refactor", skills, project_root=".")
        assert "test_and_observe" in results

    def test_lint_runs_for_default(self):
        skills = {"lint": self._make_skill("lint")}
        results = dispatch_skills("default", skills, project_root=".")
        assert "lint" in results

    def test_unknown_goal_type_runs_provided_skill_subset(self):
        skills = {
            "complexity_scorer": self._make_skill("complexity_scorer"),
            "tech_debt_quantifier": self._make_skill("tech_debt_quantifier"),
        }
        results = dispatch_skills("health_monitor", skills, project_root=".")
        assert set(results) == {"complexity_scorer", "tech_debt_quantifier"}

    def test_skills_absent_from_registry_are_silently_skipped(self):
        """If lint/test_and_observe not registered, dispatch doesn't crash."""
        skills = {"symbol_indexer": self._make_skill("symbol_indexer")}
        results = dispatch_skills("bug_fix", skills, project_root=".")
        assert "lint" not in results
        assert "test_and_observe" not in results
        assert "symbol_indexer" in results

    def test_lint_not_dispatched_for_docs(self):
        skills = {"lint": self._make_skill("lint"),
                  "doc_generator": self._make_skill("doc_generator")}
        results = dispatch_skills("docs", skills, project_root=".")
        assert "lint" not in results

    def test_skill_error_does_not_prevent_others(self):
        """A lint failure should not prevent test_and_observe from running."""
        lint_skill = self._make_skill("lint")
        lint_skill.run.side_effect = RuntimeError("flake8 crashed")
        tao_skill = self._make_skill("test_and_observe")

        skills = {"lint": lint_skill, "test_and_observe": tao_skill}
        results = dispatch_skills("bug_fix", skills, project_root=".")

        assert "lint" in results
        assert "error" in results["lint"]
        assert "test_and_observe" in results
        assert results["test_and_observe"]["status"] == "ok"

    def test_cacheable_skill_reuses_result_when_project_fingerprint_is_unchanged(self, tmp_path: Path):
        (tmp_path / "demo.py").write_text("def demo():\n    return 1\n", encoding="utf-8")
        cached_skill = self._make_skill(
            "complexity_scorer",
            {"high_risk_count": 1, "file_avg_complexity": 1.0},
        )

        results_first = dispatch_skills(
            "health_monitor",
            {"complexity_scorer": cached_skill},
            project_root=str(tmp_path),
        )
        results_second = dispatch_skills(
            "health_monitor",
            {"complexity_scorer": cached_skill},
            project_root=str(tmp_path),
        )

        assert results_first["complexity_scorer"]["high_risk_count"] == 1
        assert results_second["complexity_scorer"]["high_risk_count"] == 1
        assert cached_skill.run.call_count == 1

    def test_timeout_returns_without_waiting_for_late_completion(self, tmp_path: Path):
        (tmp_path / "demo.py").write_text("def demo():\n    return 1\n", encoding="utf-8")
        finished = {"done": False}

        class _SlowSkill:
            name = "complexity_scorer"

            def run(self, _input_data):
                time.sleep(0.2)
                finished["done"] = True
                return {"high_risk_count": 0}

        started = time.monotonic()
        results = dispatch_skills(
            "health_monitor",
            {"complexity_scorer": _SlowSkill()},
            project_root=str(tmp_path),
            timeout=0.01,
        )
        elapsed = time.monotonic() - started

        assert elapsed < 0.15
        assert results["complexity_scorer"]["error"] == "timeout"

        deadline = time.monotonic() + 1.0
        while not finished["done"] and time.monotonic() < deadline:
            time.sleep(0.02)

        assert finished["done"] is True
