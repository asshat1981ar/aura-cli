"""Unit tests for core.in_flight_tracker.InFlightTracker."""

import json
from pathlib import Path

import pytest

from core.in_flight_tracker import InFlightTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker(tmp_path: Path) -> InFlightTracker:
    return InFlightTracker(path=tmp_path / "in_flight_goal.json")


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------


class TestWrite:
    def test_creates_file(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Fix auth bug", cycle_limit=3)
        assert (tmp_path / "in_flight_goal.json").exists()

    def test_file_shape(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Refactor queue", cycle_limit=5)
        data = json.loads((tmp_path / "in_flight_goal.json").read_text())
        assert data["goal"] == "Refactor queue"
        assert data["cycle_limit"] == 5
        assert data["phase"] == "ingest"
        assert "started_at" in data
        assert "T" in data["started_at"]  # ISO-8601 format

    def test_custom_phase(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Goal", cycle_limit=1, phase="plan")
        data = json.loads((tmp_path / "in_flight_goal.json").read_text())
        assert data["phase"] == "plan"

    def test_atomic_no_tmp_left(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Goal", cycle_limit=2)
        assert not (tmp_path / "in_flight_goal.tmp").exists()

    def test_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        t = InFlightTracker(path=nested / "in_flight_goal.json")
        t.write("Goal", cycle_limit=1)
        assert (nested / "in_flight_goal.json").exists()

    def test_overwrites_existing(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("First goal", cycle_limit=1)
        t.write("Second goal", cycle_limit=2)
        data = json.loads((tmp_path / "in_flight_goal.json").read_text())
        assert data["goal"] == "Second goal"


# ---------------------------------------------------------------------------
# read()
# ---------------------------------------------------------------------------


class TestRead:
    def test_returns_none_when_absent(self, tmp_path):
        t = _tracker(tmp_path)
        assert t.read() is None

    def test_returns_record_after_write(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Fix tests", cycle_limit=4)
        result = t.read()
        assert result is not None
        assert result["goal"] == "Fix tests"
        assert result["cycle_limit"] == 4

    def test_returns_none_on_corrupt_json(self, tmp_path):
        path = tmp_path / "in_flight_goal.json"
        path.write_text("not valid json{{{{")
        t = InFlightTracker(path=path)
        assert t.read() is None

    def test_returns_none_on_empty_file(self, tmp_path):
        path = tmp_path / "in_flight_goal.json"
        path.write_text("")
        t = InFlightTracker(path=path)
        assert t.read() is None


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    def test_removes_file(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Goal", cycle_limit=1)
        assert t.exists()
        t.clear()
        assert not t.exists()

    def test_safe_when_file_absent(self, tmp_path):
        t = _tracker(tmp_path)
        # Must not raise even though file was never created
        t.clear()

    def test_read_returns_none_after_clear(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Goal", cycle_limit=1)
        t.clear()
        assert t.read() is None


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


class TestExists:
    def test_false_when_absent(self, tmp_path):
        assert not _tracker(tmp_path).exists()

    def test_true_after_write(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Goal", cycle_limit=1)
        assert t.exists()

    def test_false_after_clear(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Goal", cycle_limit=1)
        t.clear()
        assert not t.exists()


# ---------------------------------------------------------------------------
# try/finally pattern (the critical usage contract)
# ---------------------------------------------------------------------------


class TestFinallyPattern:
    def test_clear_called_even_on_exception(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Fragile goal", cycle_limit=2)
        try:
            raise RuntimeError("simulated crash")
        except RuntimeError:
            pass
        finally:
            t.clear()
        assert not t.exists()

    def test_write_then_clear_idempotent(self, tmp_path):
        t = _tracker(tmp_path)
        t.write("Goal", cycle_limit=1)
        t.clear()
        t.clear()  # second clear must not raise
        assert not t.exists()
