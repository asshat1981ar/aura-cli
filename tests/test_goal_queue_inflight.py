"""
Tests for GoalQueue InFlightTracker (Issue #301).

Verifies that next() moves goals to in_flight instead of deleting them,
complete() removes from in_flight, fail() re-queues at front, recover()
restores all in-flight goals, and that state is persisted across save/load.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from core.goal_queue import GoalQueue


@pytest.fixture
def queue_file(tmp_path) -> Path:
    return tmp_path / "test_goal_queue.json"


@pytest.fixture
def gq(queue_file) -> GoalQueue:
    return GoalQueue(queue_path=str(queue_file))


class TestNextMovesToInFlight:
    def test_next_removes_from_queue(self, gq):
        gq.add("goal-a")
        gq.next()
        assert not gq.has_goals()

    def test_next_places_in_inflight(self, gq):
        gq.add("goal-a")
        goal = gq.next()
        assert goal == "goal-a"
        assert "goal-a" in gq._in_flight

    def test_next_on_empty_returns_none(self, gq):
        assert gq.next() is None
        assert gq._in_flight == {}

    def test_next_records_timestamp(self, gq):
        import time
        before = time.time()
        gq.add("goal-b")
        gq.next()
        after = time.time()
        ts = gq._in_flight["goal-b"]
        assert before <= ts <= after


class TestComplete:
    def test_complete_removes_from_inflight(self, gq):
        gq.add("goal-a")
        gq.next()
        assert "goal-a" in gq._in_flight
        gq.complete("goal-a")
        assert "goal-a" not in gq._in_flight

    def test_complete_does_not_requeue(self, gq):
        gq.add("goal-a")
        gq.next()
        gq.complete("goal-a")
        assert not gq.has_goals()

    def test_complete_unknown_goal_does_not_raise(self, gq):
        # Should warn but not raise
        gq.complete("nonexistent-goal")


class TestFail:
    def test_fail_removes_from_inflight(self, gq):
        gq.add("goal-a")
        gq.next()
        gq.fail("goal-a")
        assert "goal-a" not in gq._in_flight

    def test_fail_puts_goal_at_front_of_queue(self, gq):
        gq.add("goal-a")
        gq.add("goal-b")
        first = gq.next()
        assert first == "goal-a"
        gq.fail("goal-a")
        # goal-a should now be at the front, before goal-b
        assert gq.next() == "goal-a"
        assert gq.next() == "goal-b"

    def test_fail_goal_not_inflight_still_requeues(self, gq):
        """fail() should tolerate goals not currently in _in_flight."""
        gq.fail("orphan-goal")
        assert gq.next() == "orphan-goal"


class TestRecover:
    def test_recover_moves_inflight_to_front(self, gq):
        gq.add("goal-a")
        gq.add("goal-b")
        gq.next()  # goal-a goes in_flight
        # Add another goal after popping
        gq.add("goal-c")
        count = gq.recover()
        assert count == 1
        # goal-a should be back at front
        assert gq.next() == "goal-a"

    def test_recover_clears_inflight(self, gq):
        gq.add("goal-a")
        gq.next()
        gq.recover()
        assert gq._in_flight == {}

    def test_recover_multiple_inflight_oldest_first(self, gq):
        import time
        gq.add("goal-x")
        gq.add("goal-y")
        # Manually set timestamps so order is deterministic
        goal_x = gq.next()
        time.sleep(0.01)
        goal_y = gq.next()
        count = gq.recover()
        assert count == 2
        # Oldest (goal-x) should be at front
        assert gq.next() == goal_x
        assert gq.next() == goal_y

    def test_recover_empty_inflight_returns_zero(self, gq):
        gq.add("goal-a")
        assert gq.recover() == 0
        # Queue untouched
        assert gq.next() == "goal-a"


class TestPersistence:
    def test_inflight_persisted_to_json(self, queue_file):
        gq = GoalQueue(queue_path=str(queue_file))
        gq.add("goal-a")
        gq.next()

        raw = json.loads(queue_file.read_text())
        assert isinstance(raw, dict)
        assert "in_flight" in raw
        assert "goal-a" in raw["in_flight"]

    def test_queue_persisted_alongside_inflight(self, queue_file):
        gq = GoalQueue(queue_path=str(queue_file))
        gq.add("goal-a")
        gq.add("goal-b")
        gq.next()

        raw = json.loads(queue_file.read_text())
        assert raw["queue"] == ["goal-b"]
        assert "goal-a" in raw["in_flight"]

    def test_reload_restores_inflight(self, queue_file):
        gq = GoalQueue(queue_path=str(queue_file))
        gq.add("goal-a")
        gq.next()

        # Reload from disk
        gq2 = GoalQueue(queue_path=str(queue_file))
        assert "goal-a" in gq2._in_flight

    def test_reload_then_recover(self, queue_file):
        gq = GoalQueue(queue_path=str(queue_file))
        gq.add("goal-a")
        gq.add("goal-b")
        gq.next()  # goal-a in flight

        # Simulate restart
        gq2 = GoalQueue(queue_path=str(queue_file))
        count = gq2.recover()
        assert count == 1
        # goal-a back at front, goal-b behind it
        assert gq2.next() == "goal-a"
        assert gq2.next() == "goal-b"

    def test_legacy_format_loads_as_empty_inflight(self, queue_file):
        """Plain JSON array (old format) should load without in_flight entries."""
        queue_file.write_text(json.dumps(["goal-old-a", "goal-old-b"]), encoding="utf-8")
        gq = GoalQueue(queue_path=str(queue_file))
        assert list(gq.queue) == ["goal-old-a", "goal-old-b"]
        assert gq._in_flight == {}

    def test_complete_removes_from_persisted_inflight(self, queue_file):
        gq = GoalQueue(queue_path=str(queue_file))
        gq.add("goal-a")
        gq.next()
        gq.complete("goal-a")

        raw = json.loads(queue_file.read_text())
        assert "goal-a" not in raw.get("in_flight", {})
