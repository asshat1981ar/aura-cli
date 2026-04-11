"""Tests for core/goal_queue.py — GoalQueue."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from core.goal_queue import GoalQueue


@pytest.fixture
def queue_path(tmp_path):
    return tmp_path / "goal_queue.json"


@pytest.fixture
def q(queue_path):
    return GoalQueue(queue_path=str(queue_path))


# ---------------------------------------------------------------------------
# Init and persistence
# ---------------------------------------------------------------------------

class TestGoalQueueInit:
    def test_empty_on_fresh_file(self, q):
        assert not q.has_goals()

    def test_queue_path_stored_as_path(self, queue_path):
        q = GoalQueue(queue_path=str(queue_path))
        assert isinstance(q.queue_path, Path)

    def test_loads_existing_queue(self, queue_path):
        queue_path.write_text(json.dumps({"queue": ["goal_a"], "in_flight": {}}))
        q = GoalQueue(queue_path=str(queue_path))
        assert q.has_goals()
        assert q.next() == "goal_a"

    def test_loads_legacy_list_format(self, queue_path):
        queue_path.write_text(json.dumps(["legacy_goal"]))
        q = GoalQueue(queue_path=str(queue_path))
        assert q.next() == "legacy_goal"

    def test_loads_in_flight_from_file(self, queue_path):
        queue_path.write_text(json.dumps({"queue": [], "in_flight": {"my_goal": 1234567890.0}}))
        q = GoalQueue(queue_path=str(queue_path))
        assert q.is_inflight("my_goal")

    def test_corrupted_json_starts_empty(self, queue_path):
        queue_path.write_text("{BROKEN JSON{{")
        q = GoalQueue(queue_path=str(queue_path))
        assert not q.has_goals()


# ---------------------------------------------------------------------------
# add / batch_add / prepend_batch
# ---------------------------------------------------------------------------

class TestGoalQueueAdd:
    def test_add_single_goal(self, q):
        q.add("goal_1")
        assert q.has_goals()

    def test_add_persists_to_disk(self, q, queue_path):
        q.add("goal_1")
        data = json.loads(queue_path.read_text())
        assert "goal_1" in data["queue"]

    def test_batch_add_multiple(self, q):
        q.batch_add(["a", "b", "c"])
        assert q.next() == "a"

    def test_batch_add_persists_once(self, q, queue_path):
        q.batch_add(["x", "y"])
        data = json.loads(queue_path.read_text())
        assert data["queue"] == ["x", "y"]

    def test_prepend_batch_puts_at_front(self, q):
        q.add("old")
        q.prepend_batch(["new1", "new2"])
        assert q.next() == "new1"
        assert q.next() == "new2"
        assert q.next() == "old"

    def test_prepend_batch_preserves_order(self, q):
        q.prepend_batch(["first", "second"])
        assert q.next() == "first"


# ---------------------------------------------------------------------------
# next / has_goals / clear
# ---------------------------------------------------------------------------

class TestGoalQueueNext:
    def test_next_returns_none_when_empty(self, q):
        assert q.next() is None

    def test_next_removes_from_queue(self, q):
        q.add("g")
        q.next()
        assert not q.has_goals()

    def test_next_adds_to_in_flight(self, q):
        q.add("g")
        q.next()
        assert q.is_inflight("g")

    def test_next_fifo_order(self, q):
        q.batch_add(["first", "second"])
        assert q.next() == "first"
        assert q.next() == "second"

    def test_clear_empties_queue(self, q):
        q.batch_add(["a", "b", "c"])
        q.clear()
        assert not q.has_goals()

    def test_has_goals_true_when_items(self, q):
        q.add("x")
        assert q.has_goals()


# ---------------------------------------------------------------------------
# complete / fail / recover
# ---------------------------------------------------------------------------

class TestGoalQueueLifecycle:
    def test_complete_removes_from_in_flight(self, q):
        q.add("g")
        q.next()
        q.complete("g")
        assert not q.is_inflight("g")

    def test_complete_not_in_flight_no_crash(self, q):
        q.complete("nonexistent")  # Should not raise

    def test_fail_requeues_at_front(self, q):
        q.add("g")
        q.next()
        q.fail("g")
        assert q.next() == "g"

    def test_fail_removes_from_in_flight(self, q):
        q.add("g")
        q.next()
        q.fail("g")
        assert not q.is_inflight("g")

    def test_recover_restores_in_flight(self, q):
        q.add("g")
        q.next()
        # Simulate crash: create new instance that loads same file
        q2 = GoalQueue(queue_path=str(q.queue_path))
        count = q2.recover()
        assert count == 1
        assert q2.has_goals()

    def test_recover_returns_zero_when_nothing_in_flight(self, q):
        assert q.recover() == 0

    def test_recover_orders_by_timestamp_oldest_first(self, tmp_path):
        qp = tmp_path / "q.json"
        qp.write_text(json.dumps({
            "queue": [],
            "in_flight": {"newer_goal": 2000.0, "older_goal": 1000.0}
        }))
        q = GoalQueue(queue_path=str(qp))
        q.recover()
        # oldest should come first
        first = q.next()
        assert first == "older_goal"


# ---------------------------------------------------------------------------
# cancel / promote
# ---------------------------------------------------------------------------

class TestGoalQueueManagement:
    def test_cancel_removes_by_index(self, q):
        q.batch_add(["a", "b", "c"])
        removed = q.cancel(1)
        assert removed == "b"
        assert q.next() == "a"
        assert q.next() == "c"

    def test_cancel_out_of_range_raises(self, q):
        q.add("only")
        with pytest.raises(IndexError):
            q.cancel(5)

    def test_promote_moves_to_front(self, q):
        q.batch_add(["a", "b", "c"])
        promoted = q.promote(2)
        assert promoted == "c"
        assert q.next() == "c"

    def test_promote_index_zero_is_noop(self, q):
        q.batch_add(["a", "b"])
        result = q.promote(0)
        assert result == "a"
        assert q.next() == "a"


# ---------------------------------------------------------------------------
# _goal_key
# ---------------------------------------------------------------------------

class TestGoalKey:
    def test_string_goal_key(self, q):
        assert GoalQueue._goal_key("my goal") == "my goal"

    def test_dict_goal_key_is_json(self, q):
        key = GoalQueue._goal_key({"id": 1, "text": "build X"})
        parsed = json.loads(key)
        assert parsed["id"] == 1

    def test_non_serialisable_goal_uses_str(self, q):
        class Weird:
            def __str__(self):
                return "weird_repr"
        key = GoalQueue._goal_key(Weird())
        assert "weird_repr" in key


# ---------------------------------------------------------------------------
# in_flight_keys
# ---------------------------------------------------------------------------

class TestInFlightKeys:
    def test_in_flight_keys_empty_initially(self, q):
        assert q.in_flight_keys() == []

    def test_in_flight_keys_after_next(self, q):
        q.add("task")
        q.next()
        assert "task" in q.in_flight_keys()

    def test_in_flight_keys_cleared_after_complete(self, q):
        q.add("task")
        q.next()
        q.complete("task")
        assert q.in_flight_keys() == []
