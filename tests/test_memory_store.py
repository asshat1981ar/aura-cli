"""Tests for memory/store.py — atomic writes, locking, schema versioning."""
import json
import tempfile
import unittest
from pathlib import Path

from memory.store import MemoryStore, _unwrap_versioned, _wrap_versioned


class TestSchemaVersioning(unittest.TestCase):
    def test_wrap_versioned(self):
        data = [{"a": 1}]
        wrapped = _wrap_versioned(data)
        assert wrapped["schema_version"] == 1
        assert wrapped["data"] == data

    def test_unwrap_versioned(self):
        raw = {"schema_version": 1, "data": [{"a": 1}]}
        assert _unwrap_versioned(raw) == [{"a": 1}]

    def test_unwrap_legacy_list(self):
        raw = [{"a": 1}, {"b": 2}]
        assert _unwrap_versioned(raw) == raw

    def test_unwrap_invalid(self):
        assert _unwrap_versioned("not a list or dict") == []


class TestMemoryStorePutQuery(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(root=Path(self.tmpdir))

    def test_put_and_query(self):
        self.store.put("test_tier", {"key": "value1"})
        self.store.put("test_tier", {"key": "value2"})
        results = self.store.query("test_tier")
        assert len(results) == 2
        assert results[0]["key"] == "value1"
        assert results[1]["key"] == "value2"

    def test_query_limit(self):
        for i in range(10):
            self.store.put("tier", {"i": i})
        results = self.store.query("tier", limit=3)
        assert len(results) == 3
        assert results[0]["i"] == 7  # last 3 of 0-9

    def test_query_empty_tier(self):
        results = self.store.query("nonexistent")
        assert results == []

    def test_data_persisted_with_version_envelope(self):
        self.store.put("versioned", {"x": 1})
        path = self.store._tier_path("versioned")
        raw = json.loads(path.read_text())
        assert "schema_version" in raw
        assert raw["schema_version"] == 1
        assert isinstance(raw["data"], list)

    def test_reads_legacy_bare_list(self):
        """Old data without version envelope should still be readable."""
        path = self.store._tier_path("legacy")
        path.write_text(json.dumps([{"old": True}]))
        results = self.store.query("legacy")
        assert len(results) == 1
        assert results[0]["old"] is True

    def test_put_on_legacy_migrates_to_versioned(self):
        """Appending to a legacy file should wrap it in the version envelope."""
        path = self.store._tier_path("migrate")
        path.write_text(json.dumps([{"old": True}]))
        self.store.put("migrate", {"new": True})
        raw = json.loads(path.read_text())
        assert raw["schema_version"] == 1
        assert len(raw["data"]) == 2

    def test_query_repairs_from_backup_when_json_corrupted(self):
        path = self.store._tier_path("repair")
        backup = path.with_suffix(".json.bak")
        backup.write_text(json.dumps(_wrap_versioned([{"ok": True}])))
        path.write_text("{not-json")

        results = self.store.query("repair")

        assert results == [{"ok": True}]
        assert json.loads(path.read_text())["data"] == [{"ok": True}]


class TestMemoryStoreLog(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(root=Path(self.tmpdir))

    def test_append_and_read_log(self):
        self.store.append_log({"event": "a"})
        self.store.append_log({"event": "b"})
        entries = self.store.read_log()
        assert len(entries) == 2
        assert entries[0]["event"] == "a"

    def test_read_log_limit(self):
        for i in range(10):
            self.store.append_log({"i": i})
        entries = self.store.read_log(limit=3)
        assert len(entries) == 3
        assert entries[0]["i"] == 7


if __name__ == "__main__":
    unittest.main()
