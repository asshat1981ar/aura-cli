"""Tests for core/learning_coordinator.py — LearningCoordinator."""

import dataclasses
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.learning_coordinator import LearningCoordinator
from core.learning_types import LearningArtifact
from memory.store import MemoryStore


def make_store() -> MemoryStore:
    """Return a fresh MemoryStore backed by a temp directory."""
    d = tempfile.mkdtemp()
    return MemoryStore(Path(d))


def make_cycle_entry(cycle_id="c1", goal="Add feature", goal_type="feature") -> dict:
    return {"cycle_id": cycle_id, "goal": goal, "goal_type": goal_type}


def make_alert(
    alert_type="threshold_breach",
    metric="health_score",
    current=0.3,
    previous=0.8,
    threshold=0.4,
    severity="high",
    suggested_goal="Fix health score",
):
    """Create a mock TrendAlert-like object."""
    alert = MagicMock()
    alert.alert_type = alert_type
    alert.metric = metric
    alert.current_value = current
    alert.previous_value = previous
    alert.threshold = threshold
    alert.severity = severity
    alert.suggested_goal = suggested_goal
    return alert


class TestOnCycleCompleteEmpty:
    def test_empty_reflection_and_no_alerts_returns_empty_goals(self):
        coord = LearningCoordinator(make_store())
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, [])
        assert goals == []

    def test_empty_returns_no_persisted_artifacts(self):
        store = make_store()
        coord = LearningCoordinator(store)
        coord.on_cycle_complete(make_cycle_entry(), {}, [])
        arts = store.query("learning_artifacts")
        assert arts == []

    def test_reflection_with_no_learnings_key(self):
        coord = LearningCoordinator(make_store())
        goals = coord.on_cycle_complete(make_cycle_entry(), {"summary": "ok"}, [])
        assert goals == []


class TestCycleLearningArtifacts:
    def test_learnings_list_creates_artifacts(self):
        store = make_store()
        coord = LearningCoordinator(store)
        coord.on_cycle_complete(
            make_cycle_entry(),
            {"learnings": ["Import error suggests missing dep", "NameError in plan"]},
            [],
        )
        arts = store.query("learning_artifacts")
        assert len(arts) == 2

    def test_cycle_learning_artifact_fields(self):
        store = make_store()
        coord = LearningCoordinator(store)
        coord.on_cycle_complete(
            make_cycle_entry(cycle_id="c99", goal="Fix bug", goal_type="bug_fix"),
            {"learnings": ["Context gap detected"]},
            [],
        )
        art = store.query("learning_artifacts")[0]
        assert art["artifact_type"] == "cycle_learning"
        assert art["severity"] == "low"
        assert art["cycle_id"] == "c99"
        assert art["goal"] == "Fix bug"
        assert art["goal_type"] == "bug_fix"
        assert "Context gap" in art["insight"]

    def test_low_severity_learnings_not_returned_as_goals(self):
        coord = LearningCoordinator(make_store())
        goals = coord.on_cycle_complete(
            make_cycle_entry(),
            {"learnings": ["some learning"]},
            [],
        )
        assert goals == []

    def test_empty_strings_in_learnings_ignored(self):
        store = make_store()
        coord = LearningCoordinator(store)
        coord.on_cycle_complete(
            make_cycle_entry(),
            {"learnings": ["", "   ", "real learning"]},
            [],
        )
        arts = store.query("learning_artifacts")
        assert len(arts) == 1  # only "real learning"


class TestQualityAlertArtifacts:
    def test_high_severity_alert_returns_goal(self):
        coord = LearningCoordinator(make_store())
        alert = make_alert(severity="high", suggested_goal="Fix health score")
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, [alert])
        assert "Fix health score" in goals

    def test_critical_severity_alert_returns_goal(self):
        coord = LearningCoordinator(make_store())
        alert = make_alert(severity="critical", suggested_goal="Critical fix")
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, [alert])
        assert "Critical fix" in goals

    def test_low_severity_alert_not_returned_as_goal(self):
        coord = LearningCoordinator(make_store())
        alert = make_alert(severity="low", suggested_goal="Minor fix")
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, [alert])
        assert goals == []

    def test_medium_severity_alert_not_returned_as_goal(self):
        coord = LearningCoordinator(make_store())
        alert = make_alert(severity="medium", suggested_goal="Medium fix")
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, [alert])
        assert goals == []

    def test_alert_creates_quality_regression_artifact(self):
        store = make_store()
        coord = LearningCoordinator(store)
        alert = make_alert(metric="health_score", current=0.3)
        coord.on_cycle_complete(make_cycle_entry(), {}, [alert])
        arts = store.query("learning_artifacts")
        assert any(a["artifact_type"] == "quality_regression" for a in arts)

    def test_alert_without_suggested_goal_not_returned(self):
        coord = LearningCoordinator(make_store())
        alert = make_alert(severity="high", suggested_goal="")
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, [alert])
        assert goals == []

    def test_alert_evidence_persisted(self):
        store = make_store()
        coord = LearningCoordinator(store)
        alert = make_alert(metric="syntax_errors", current=5.0, threshold=3.0)
        coord.on_cycle_complete(make_cycle_entry(), {}, [alert])
        arts = store.query("learning_artifacts")
        assert any("syntax_errors" in str(a.get("evidence", {})) for a in arts)


class TestGoalCap:
    def test_goals_capped_at_max_per_cycle(self):
        coord = LearningCoordinator(make_store())
        alerts = [
            make_alert(severity="high", suggested_goal=f"Fix thing {i}")
            for i in range(10)
        ]
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, alerts)
        assert len(goals) <= LearningCoordinator.MAX_GOALS_PER_CYCLE

    def test_cap_is_three_by_default(self):
        assert LearningCoordinator.MAX_GOALS_PER_CYCLE == 3

    def test_overflow_goals_are_deferred_to_backlog(self):
        coord = LearningCoordinator(make_store())
        alerts = [
            make_alert(severity="high", suggested_goal=f"Fix thing {i}")
            for i in range(5)
        ]
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, alerts)
        backlog = coord.generate_backlog(limit=10)
        assert goals == ["Fix thing 0", "Fix thing 1", "Fix thing 2"]
        assert backlog == ["Fix thing 3", "Fix thing 4"]


class TestGenerateBacklog:
    def test_generate_backlog_empty_initially(self):
        coord = LearningCoordinator(make_store())
        assert coord.generate_backlog() == []

    def test_generate_backlog_returns_pending_goals(self):
        coord = LearningCoordinator(make_store())
        coord._pending_goals = ["goal A", "goal B", "goal C"]
        result = coord.generate_backlog(limit=2)
        assert result == ["goal A", "goal B"]

    def test_generate_backlog_drains_queue(self):
        coord = LearningCoordinator(make_store())
        coord._pending_goals = ["goal A", "goal B"]
        coord.generate_backlog(limit=2)
        assert coord._pending_goals == []

    def test_generate_backlog_advances_cursor(self):
        coord = LearningCoordinator(make_store())
        coord._pending_goals = ["A", "B", "C", "D"]
        first = coord.generate_backlog(limit=2)
        second = coord.generate_backlog(limit=2)
        assert first == ["A", "B"]
        assert second == ["C", "D"]

    def test_generate_backlog_second_call_empty(self):
        coord = LearningCoordinator(make_store())
        coord._pending_goals = ["X"]
        coord.generate_backlog(limit=3)
        assert coord.generate_backlog(limit=3) == []


class TestGetRecentArtifacts:
    def test_returns_empty_when_no_artifacts(self):
        coord = LearningCoordinator(make_store())
        assert coord.get_recent_artifacts() == []

    def test_returns_persisted_artifacts(self):
        store = make_store()
        coord = LearningCoordinator(store)
        coord.on_cycle_complete(
            make_cycle_entry(),
            {"learnings": ["lesson one"]},
            [],
        )
        arts = coord.get_recent_artifacts()
        assert len(arts) == 1


class TestSyncReflectionReports:
    def _write_report(self, store: MemoryStore, ts: float, insights: list) -> None:
        store.put(
            "reflection_reports",
            {
                "timestamp": ts,
                "cycles_analyzed": 5,
                "insights": insights,
            },
        )

    def test_skips_old_report(self):
        store = make_store()
        coord = LearningCoordinator(store)
        coord._last_reflection_ts = time.time()  # already "seen" this timestamp

        self._write_report(
            store,
            ts=coord._last_reflection_ts - 1,
            insights=[
                {"type": "phase_failure", "phase": "plan", "failure_rate": 0.8, "severity": "HIGH", "message": "plan failing"},
            ],
        )
        artifacts, goals = coord._sync_reflection_reports(make_cycle_entry())
        assert artifacts == []
        assert goals == []

    def test_processes_new_report(self):
        store = make_store()
        coord = LearningCoordinator(store)
        new_ts = time.time() + 10

        self._write_report(
            store,
            ts=new_ts,
            insights=[
                {"type": "phase_failure", "phase": "plan", "failure_rate": 0.8, "severity": "HIGH", "message": "plan failing 80% of the time"},
            ],
        )
        artifacts, goals = coord._sync_reflection_reports(make_cycle_entry())
        assert len(artifacts) == 1
        assert len(goals) == 1
        assert "plan" in goals[0]

    def test_updates_last_reflection_ts(self):
        store = make_store()
        coord = LearningCoordinator(store)
        new_ts = time.time() + 10
        self._write_report(store, ts=new_ts, insights=[])
        coord._sync_reflection_reports(make_cycle_entry())
        assert coord._last_reflection_ts == new_ts

    def test_medium_severity_insight_no_goal(self):
        store = make_store()
        coord = LearningCoordinator(store)
        self._write_report(
            store,
            ts=time.time() + 1,
            insights=[
                {"type": "low_value_skill", "skill": "linter", "actionable_rate": 0.1, "severity": "LOW", "message": "linter low signal"},
            ],
        )
        artifacts, goals = coord._sync_reflection_reports(make_cycle_entry())
        assert len(artifacts) == 1
        assert goals == []  # LOW severity → no goal

    def test_critical_severity_is_normalized_and_actionable(self):
        store = make_store()
        coord = LearningCoordinator(store)
        self._write_report(
            store,
            ts=time.time() + 1,
            insights=[
                {
                    "type": "phase_failure",
                    "phase": "verify",
                    "failure_rate": 0.95,
                    "severity": "CRITICAL",
                    "message": "verify failing badly",
                }
            ],
        )
        artifacts, goals = coord._sync_reflection_reports(make_cycle_entry())
        assert artifacts[0].severity == "critical"
        assert len(goals) == 1

    def test_empty_report(self):
        store = make_store()
        coord = LearningCoordinator(store)
        self._write_report(store, ts=time.time() + 1, insights=[])
        artifacts, goals = coord._sync_reflection_reports(make_cycle_entry())
        assert artifacts == []
        assert goals == []

    def test_no_reports_returns_empty(self):
        coord = LearningCoordinator(make_store())
        artifacts, goals = coord._sync_reflection_reports(make_cycle_entry())
        assert artifacts == []
        assert goals == []


class TestInsightToGoal:
    def setup_method(self):
        self.coord = LearningCoordinator(make_store())

    def test_phase_failure_insight(self):
        goal = self.coord._insight_to_goal(
            {
                "type": "phase_failure",
                "phase": "act",
                "failure_rate": 0.72,
            }
        )
        assert goal is not None
        assert "act" in goal
        assert "72%" in goal

    def test_goal_type_struggling_insight(self):
        goal = self.coord._insight_to_goal(
            {
                "type": "goal_type_struggling",
                "goal_type": "refactor",
                "success_rate": 0.25,
            }
        )
        assert goal is not None
        assert "refactor" in goal
        assert "25%" in goal

    def test_unknown_type_returns_message(self):
        goal = self.coord._insight_to_goal(
            {
                "type": "unknown",
                "message": "Some other insight",
            }
        )
        assert goal == "Some other insight"

    def test_missing_message_returns_none(self):
        goal = self.coord._insight_to_goal({"type": "unknown"})
        assert goal is None


class TestErrorHandling:
    def test_on_cycle_complete_swallows_exceptions(self):
        coord = LearningCoordinator(make_store())
        # Malformed alert that will raise during conversion
        bad_alert = MagicMock(spec=[])  # no attributes at all
        goals = coord.on_cycle_complete(make_cycle_entry(), {}, [bad_alert])
        # Should not raise; returns gracefully
        assert isinstance(goals, list)
