"""Tests for memory.store.MemoryStore — JSONL decision log + tier storage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from memory.store import MemoryStore, _LOG_MAX_BYTES, _LOG_KEEP_ROTATIONS


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(root=tmp_path)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestMemoryStoreInit:
    def test_creates_root_directory(self, tmp_path):
        nested = tmp_path / "a" / "b"
        s = MemoryStore(root=nested)
        assert nested.exists()

    def test_log_path_set_correctly(self, store, tmp_path):
        assert store.log_path == tmp_path / "decision_log.jsonl"

    def test_root_stored_as_path(self, store, tmp_path):
        assert store.root == tmp_path


# ---------------------------------------------------------------------------
# put / query
# ---------------------------------------------------------------------------


class TestPutQuery:
    def test_put_creates_tier_file(self, store, tmp_path):
        store.put("working", {"content": "test"})
        assert (tmp_path / "working.json").exists()

    def test_put_and_query_single_record(self, store):
        store.put("session", {"data": 42})
        results = store.query("session")
        assert len(results) == 1
        assert results[0]["data"] == 42

    def test_put_multiple_records_appends(self, store):
        store.put("session", {"n": 1})
        store.put("session", {"n": 2})
        store.put("session", {"n": 3})
        results = store.query("session")
        assert len(results) == 3

    def test_query_respects_limit(self, store):
        for i in range(10):
            store.put("project", {"i": i})
        results = store.query("project", limit=3)
        assert len(results) == 3

    def test_query_returns_tail(self, store):
        for i in range(5):
            store.put("project", {"i": i})
        results = store.query("project", limit=2)
        assert results[0]["i"] == 3
        assert results[1]["i"] == 4

    def test_query_missing_tier_returns_empty(self, store):
        assert store.query("nonexistent") == []

    def test_put_handles_corrupted_file(self, store, tmp_path):
        tier_file = tmp_path / "broken.json"
        tier_file.write_text("NOT VALID JSON", encoding="utf-8")
        store.put("broken", {"recovery": True})
        results = store.query("broken")
        assert results == [{"recovery": True}]

    def test_query_handles_corrupted_file(self, store, tmp_path):
        tier_file = tmp_path / "corrupt.json"
        tier_file.write_text("<<<invalid>>>", encoding="utf-8")
        assert store.query("corrupt") == []


# ---------------------------------------------------------------------------
# append_log / read_log
# ---------------------------------------------------------------------------


class TestAppendReadLog:
    def test_append_log_creates_file(self, store):
        store.append_log({"event": "test"})
        assert store.log_path.exists()

    def test_append_log_readable(self, store):
        store.append_log({"action": "deploy"})
        entries = store.read_log()
        assert len(entries) == 1
        assert entries[0]["action"] == "deploy"

    def test_read_log_multiple_entries(self, store):
        for i in range(5):
            store.append_log({"step": i})
        entries = store.read_log()
        assert len(entries) == 5

    def test_read_log_limit(self, store):
        for i in range(10):
            store.append_log({"i": i})
        entries = store.read_log(limit=4)
        assert len(entries) == 4

    def test_read_log_limit_zero_returns_all(self, store):
        for i in range(5):
            store.append_log({"i": i})
        entries = store.read_log(limit=0)
        assert len(entries) == 5

    def test_read_log_missing_file_returns_empty(self, store):
        assert store.read_log() == []

    def test_read_log_skips_blank_lines(self, store):
        with store.log_path.open("w", encoding="utf-8") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')
        entries = store.read_log()
        assert len(entries) == 2

    def test_read_log_skips_invalid_json_lines(self, store):
        with store.log_path.open("w", encoding="utf-8") as f:
            f.write('{"valid": true}\nINVALID\n{"also": "valid"}\n')
        entries = store.read_log()
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------


class TestLogRotation:
    def test_rotate_not_triggered_when_small(self, store):
        store.append_log({"tiny": "entry"})
        assert not (store.log_path.parent / "decision_log.jsonl.1").exists()

    def test_rotate_triggered_when_exceeds_max(self, store, monkeypatch):
        import memory.store as ms_module

        monkeypatch.setattr(ms_module, "_LOG_MAX_BYTES", 10)

        big_entry = {"data": "x" * 50}
        store.append_log(big_entry)
        # Write enough to exceed threshold
        with store.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"more": "y" * 50}) + "\n")

        # Trigger rotation
        store.append_log({"trigger": "rotation"})
        rotated = store.log_path.parent / "decision_log.jsonl.1"
        assert rotated.exists()

    def test_rotate_shifts_existing_rotations(self, store, monkeypatch):
        import memory.store as ms_module

        monkeypatch.setattr(ms_module, "_LOG_MAX_BYTES", 1)

        # Create existing .1 rotation file
        rot1 = store.log_path.with_suffix(".jsonl.1")
        rot1.write_text('{"old": true}\n', encoding="utf-8")

        # Write something to log so it's non-empty and exceeds 1 byte
        with store.log_path.open("w", encoding="utf-8") as f:
            f.write('{"current": true}\n')

        store._rotate_log_if_needed()

        # .1 should now hold the old current, .2 should hold the old .1
        rot2 = store.log_path.with_suffix(".jsonl.2")
        assert rot2.exists()

    def test_rotate_no_op_when_file_missing(self, store):
        # Should not raise even when log file doesn't exist
        store._rotate_log_if_needed()


# ---------------------------------------------------------------------------
# tier_path helper
# ---------------------------------------------------------------------------


class TestTierPath:
    def test_tier_path_under_root(self, store, tmp_path):
        assert store._tier_path("working") == tmp_path / "working.json"

    def test_different_tiers_different_files(self, store):
        assert store._tier_path("a") != store._tier_path("b")
