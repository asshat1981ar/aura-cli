"""20 focused tests for CycleOutcome dataclass."""
import json
import time
import uuid
import pytest
from core.cycle_outcome import CycleOutcome


class TestCycleOutcomeDefaults:
    def test_cycle_id_is_string(self):
        co = CycleOutcome()
        assert isinstance(co.cycle_id, str)

    def test_cycle_id_is_uuid(self):
        co = CycleOutcome()
        # Should be parseable as UUID
        uuid.UUID(co.cycle_id)

    def test_cycle_id_unique(self):
        a = CycleOutcome()
        b = CycleOutcome()
        assert a.cycle_id != b.cycle_id

    def test_goal_default_empty(self):
        co = CycleOutcome()
        assert co.goal == ""

    def test_goal_type_default_empty(self):
        co = CycleOutcome()
        assert co.goal_type == ""

    def test_started_at_is_float(self):
        co = CycleOutcome()
        assert isinstance(co.started_at, float)

    def test_started_at_recent(self):
        before = time.time()
        co = CycleOutcome()
        after = time.time()
        assert before <= co.started_at <= after

    def test_completed_at_default_zero(self):
        co = CycleOutcome()
        assert co.completed_at == 0.0

    def test_phases_completed_default_empty(self):
        co = CycleOutcome()
        assert co.phases_completed == []

    def test_changes_applied_default_zero(self):
        co = CycleOutcome()
        assert co.changes_applied == 0

    def test_success_default_false(self):
        co = CycleOutcome()
        assert co.success is False

    def test_failure_phase_default_none(self):
        co = CycleOutcome()
        assert co.failure_phase is None

    def test_failure_reason_default_none(self):
        co = CycleOutcome()
        assert co.failure_reason is None


class TestCycleOutcomeMarkComplete:
    def test_mark_complete_sets_completed_at(self):
        co = CycleOutcome()
        before = time.time()
        co.mark_complete(True)
        assert co.completed_at >= before

    def test_mark_complete_success_true(self):
        co = CycleOutcome()
        co.mark_complete(True)
        assert co.success is True

    def test_mark_complete_success_false(self):
        co = CycleOutcome()
        co.mark_complete(False)
        assert co.success is False

    def test_mark_complete_sets_failure_phase(self):
        co = CycleOutcome()
        co.mark_complete(False, failure_phase="act")
        assert co.failure_phase == "act"

    def test_mark_complete_sets_failure_reason(self):
        co = CycleOutcome()
        co.mark_complete(False, failure_reason="timeout")
        assert co.failure_reason == "timeout"

    def test_mark_complete_computes_tests_delta(self):
        co = CycleOutcome(tests_before=10, tests_after=13)
        co.mark_complete(True)
        assert co.tests_delta == 3

    def test_mark_complete_negative_delta(self):
        co = CycleOutcome(tests_before=10, tests_after=8)
        co.mark_complete(False)
        assert co.tests_delta == -2


class TestCycleOutcomeDuration:
    def test_duration_s(self):
        co = CycleOutcome()
        co.started_at = 100.0
        co.completed_at = 105.5
        assert co.duration_s() == pytest.approx(5.5)


class TestCycleOutcomeJSON:
    def test_to_json_returns_string(self):
        co = CycleOutcome(goal="fix bug")
        assert isinstance(co.to_json(), str)

    def test_to_json_valid_json(self):
        co = CycleOutcome(goal="fix bug")
        d = json.loads(co.to_json())
        assert d["goal"] == "fix bug"

    def test_from_json_round_trip(self):
        co = CycleOutcome(goal="refactor", goal_type="refactor", strategy_used="deep")
        co.mark_complete(True)
        co2 = CycleOutcome.from_json(co.to_json())
        assert co2.cycle_id == co.cycle_id
        assert co2.goal == "refactor"
        assert co2.goal_type == "refactor"
        assert co2.success is True

    def test_from_json_preserves_all_fields(self):
        co = CycleOutcome(
            goal="test",
            goal_type="bug_fix",
            changes_applied=3,
            tests_before=10,
            tests_after=15,
            brain_entries_added=2,
        )
        co.mark_complete(True)
        co2 = CycleOutcome.from_json(co.to_json())
        assert co2.changes_applied == 3
        assert co2.tests_before == 10
        assert co2.tests_after == 15
        assert co2.tests_delta == 5
        assert co2.brain_entries_added == 2
