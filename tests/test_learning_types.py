"""Tests for core/learning_types.py — LearningArtifact dataclass."""

import dataclasses
import time

import pytest

from core.learning_types import LearningArtifact, ARTIFACT_TYPES, SEVERITIES


class TestLearningArtifactDefaults:
    def test_artifact_id_auto_generated(self):
        art = LearningArtifact()
        assert isinstance(art.artifact_id, str)
        assert len(art.artifact_id) == 32  # uuid4().hex

    def test_artifact_ids_are_unique(self):
        a1 = LearningArtifact()
        a2 = LearningArtifact()
        assert a1.artifact_id != a2.artifact_id

    def test_created_at_is_recent_float(self):
        before = time.time()
        art = LearningArtifact()
        after = time.time()
        assert before <= art.created_at <= after

    def test_acted_on_defaults_false(self):
        art = LearningArtifact()
        assert art.acted_on is False

    def test_default_severity_is_low(self):
        art = LearningArtifact()
        assert art.severity == "low"

    def test_default_artifact_type_is_cycle_learning(self):
        art = LearningArtifact()
        assert art.artifact_type == "cycle_learning"

    def test_evidence_defaults_empty_dict(self):
        art = LearningArtifact()
        assert art.evidence == {}

    def test_suggested_goal_defaults_none(self):
        art = LearningArtifact()
        assert art.suggested_goal is None


class TestLearningArtifactFields:
    def test_all_fields_settable(self):
        art = LearningArtifact(
            cycle_id="cycle-123",
            goal="Add auth",
            goal_type="feature",
            artifact_type="phase_failure",
            insight="Plan phase failing 70% of the time",
            evidence={"failure_rate": 0.7, "phase": "plan"},
            suggested_goal="Fix plan phase failure",
            severity="high",
        )
        assert art.cycle_id == "cycle-123"
        assert art.goal == "Add auth"
        assert art.goal_type == "feature"
        assert art.artifact_type == "phase_failure"
        assert "70%" in art.insight
        assert art.evidence["failure_rate"] == 0.7
        assert art.suggested_goal == "Fix plan phase failure"
        assert art.severity == "high"


class TestLearningArtifactMethods:
    def test_is_actionable_with_goal_not_acted_on(self):
        art = LearningArtifact(suggested_goal="Fix something", acted_on=False)
        assert art.is_actionable() is True

    def test_is_actionable_with_goal_acted_on(self):
        art = LearningArtifact(suggested_goal="Fix something", acted_on=True)
        assert art.is_actionable() is False

    def test_is_actionable_without_goal(self):
        art = LearningArtifact(suggested_goal=None, acted_on=False)
        assert art.is_actionable() is False

    def test_is_actionable_empty_goal_string(self):
        art = LearningArtifact(suggested_goal="", acted_on=False)
        assert art.is_actionable() is False

    def test_mark_acted_on_sets_flag(self):
        art = LearningArtifact(suggested_goal="Fix X")
        assert art.acted_on is False
        art.mark_acted_on()
        assert art.acted_on is True

    def test_mark_acted_on_idempotent(self):
        art = LearningArtifact(suggested_goal="Fix X")
        art.mark_acted_on()
        art.mark_acted_on()
        assert art.acted_on is True


class TestLearningArtifactSerialization:
    def test_asdict_round_trip(self):
        art = LearningArtifact(
            cycle_id="c1",
            goal="test",
            goal_type="bug_fix",
            artifact_type="quality_regression",
            insight="Health score dropped",
            evidence={"metric": "health_score", "value": 0.3},
            suggested_goal="Fix quality regression",
            severity="high",
        )
        d = dataclasses.asdict(art)
        assert d["cycle_id"] == "c1"
        assert d["goal"] == "test"
        assert d["artifact_type"] == "quality_regression"
        assert d["severity"] == "high"
        assert d["acted_on"] is False
        assert d["suggested_goal"] == "Fix quality regression"
        assert isinstance(d["artifact_id"], str)
        assert isinstance(d["created_at"], float)

    def test_asdict_contains_all_fields(self):
        art = LearningArtifact()
        d = dataclasses.asdict(art)
        expected_keys = {
            "artifact_id",
            "cycle_id",
            "goal",
            "goal_type",
            "artifact_type",
            "insight",
            "evidence",
            "suggested_goal",
            "severity",
            "created_at",
            "acted_on",
        }
        assert expected_keys == set(d.keys())


class TestConstants:
    def test_artifact_types_contains_expected(self):
        assert "phase_failure" in ARTIFACT_TYPES
        assert "skill_weakness" in ARTIFACT_TYPES
        assert "quality_regression" in ARTIFACT_TYPES
        assert "cycle_learning" in ARTIFACT_TYPES
        assert "success_pattern" in ARTIFACT_TYPES

    def test_severities_ordered(self):
        assert SEVERITIES == ("low", "medium", "high", "critical")
