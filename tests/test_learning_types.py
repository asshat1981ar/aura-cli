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


class TestLearningArtifactSeverityValues:
    def test_low_severity_accepted(self):
        art = LearningArtifact(severity="low")
        assert art.severity == "low"

    def test_high_severity_accepted(self):
        art = LearningArtifact(severity="high")
        assert art.severity == "high"

    def test_critical_severity_accepted(self):
        art = LearningArtifact(severity="critical")
        assert art.severity == "critical"

    def test_severity_affects_is_actionable(self):
        # severity alone doesn't affect is_actionable — needs a goal
        art = LearningArtifact(severity="critical", suggested_goal=None)
        assert art.is_actionable() is False

    def test_multiple_artifacts_independent(self):
        a1 = LearningArtifact(cycle_id="c1", severity="high")
        a2 = LearningArtifact(cycle_id="c2", severity="low")
        assert a1.severity != a2.severity
        assert a1.cycle_id != a2.cycle_id


class TestLearningArtifactBulk:
    def test_ten_artifacts_all_unique_ids(self):
        arts = [LearningArtifact() for _ in range(10)]
        ids = [a.artifact_id for a in arts]
        assert len(set(ids)) == 10

    def test_mark_acted_on_does_not_affect_others(self):
        a1 = LearningArtifact(suggested_goal="g1")
        a2 = LearningArtifact(suggested_goal="g2")
        a1.mark_acted_on()
        assert a2.acted_on is False

    def test_filter_actionable_from_list(self):
        arts = [
            LearningArtifact(suggested_goal="fix x"),
            LearningArtifact(suggested_goal=None),
            LearningArtifact(suggested_goal="fix y", acted_on=True),
            LearningArtifact(suggested_goal="fix z"),
        ]
        actionable = [a for a in arts if a.is_actionable()]
        assert len(actionable) == 2

    def test_artifact_type_frozenset_immutable(self):
        import pytest

        with pytest.raises((AttributeError, TypeError)):
            ARTIFACT_TYPES.add("new_type")  # type: ignore[union-attr]


class TestConstants:
    def test_artifact_types_contains_expected(self):
        assert "phase_failure" in ARTIFACT_TYPES
        assert "skill_weakness" in ARTIFACT_TYPES
        assert "quality_regression" in ARTIFACT_TYPES
        assert "cycle_learning" in ARTIFACT_TYPES
        assert "success_pattern" in ARTIFACT_TYPES

    def test_artifact_types_exact_count(self):
        assert len(ARTIFACT_TYPES) == 5

    def test_severities_ordered(self):
        assert SEVERITIES == ("low", "medium", "high", "critical")

    def test_severities_exact_count(self):
        assert len(SEVERITIES) == 4

    def test_low_is_first_severity(self):
        assert SEVERITIES[0] == "low"

    def test_critical_is_last_severity(self):
        assert SEVERITIES[-1] == "critical"

    def test_medium_severity_in_severities(self):
        assert "medium" in SEVERITIES
