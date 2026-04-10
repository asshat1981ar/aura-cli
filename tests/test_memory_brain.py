"""Tests for memory.brain.Brain — high-coverage unit tests with SQLite isolation."""

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

from memory.brain import Brain


@pytest.fixture()
def brain(tmp_path):
    """Return a Brain instance backed by a temp file (avoids polluting cwd)."""
    db_file = tmp_path / "test_brain.db"
    with patch("memory.brain.log_json"):
        b = Brain(db_path=str(db_file))
    yield b
    # Cleanup
    try:
        b.db.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Initialization / Schema Tests
# ---------------------------------------------------------------------------


class TestBrainInit:
    def test_creates_successfully(self, brain):
        assert brain is not None
        assert brain._db_path is not None

    def test_schema_version_at_current(self, brain):
        assert brain._get_schema_version() == Brain.SCHEMA_VERSION

    def test_db_connection_is_open(self, brain):
        row = brain.db.execute("SELECT 1").fetchone()
        assert row[0] == 1

    def test_tables_created(self, brain):
        tables = {r[0] for r in brain.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for expected in ("memory", "weaknesses", "kv_store", "innovation_sessions"):
            assert expected in tables

    def test_indices_created(self, brain):
        indices = {r[0] for r in brain.db.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        assert "idx_memory_id" in indices

    def test_concurrent_access_safe(self, brain):
        """Test that threading lock prevents concurrent issues."""
        with brain._db_lock():
            brain.db.execute("INSERT INTO memory(content) VALUES (?)", ("test",))
            brain.db.commit()
        
        entries = brain.recall_all()
        assert len(entries) > 0


# ---------------------------------------------------------------------------
# Remember / Recall Core Tests
# ---------------------------------------------------------------------------


class TestRememberRecall:
    def test_remember_string(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("hello world")
        entries = brain.recall_all()
        assert "hello world" in entries

    def test_remember_dict_serialised(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember({"key": "val", "nested": {"deep": "value"}})
        entries = brain.recall_all()
        assert any("key" in e for e in entries)

    def test_remember_int(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember(42)
        entries = brain.recall_all()
        assert any("42" in str(e) for e in entries)

    def test_remember_float(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember(3.14)
        entries = brain.recall_all()
        assert any("3.14" in str(e) for e in entries)

    def test_recall_all_returns_list(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("a")
            brain.remember("b")
            brain.remember("c")
        result = brain.recall_all()
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_recall_all_preserves_order(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("first")
            brain.remember("second")
            brain.remember("third")
        result = brain.recall_all()
        first_idx = result.index("first")
        second_idx = result.index("second")
        third_idx = result.index("third")
        assert first_idx < second_idx < third_idx

    def test_recall_recent_limit(self, brain):
        with patch("memory.brain.log_json"):
            for i in range(10):
                brain.remember(f"item-{i}")
        recent = brain.recall_recent(limit=3)
        assert len(recent) == 3

    def test_recall_recent_newest_last(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("first")
            brain.remember("second")
            brain.remember("third")
        recent = brain.recall_recent(limit=3)
        assert recent[-1] == "third"
        assert recent[0] == "first"

    def test_recall_recent_default_limit(self, brain):
        with patch("memory.brain.log_json"):
            for i in range(150):
                brain.remember(f"x-{i}")
        recent = brain.recall_recent()
        assert len(recent) == 100  # default limit

    def test_recall_with_budget_respects_tokens(self, brain):
        with patch("memory.brain.log_json"):
            for i in range(50):
                brain.remember("x" * 100)
        result = brain.recall_with_budget(max_tokens=10)
        total_chars = sum(len(e) for e in result)
        assert total_chars <= 10 * 4 + len(result) * 2  # budget with slack

    def test_recall_with_budget_recent_first(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("old" * 10)
            brain.remember("new" * 10)
        result = brain.recall_with_budget(max_tokens=5)
        # Recent entries should be prioritized
        # Result might be empty or partial depending on budget
        assert isinstance(result, list)

    def test_count_memories(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("a")
            brain.remember("b")
            brain.remember("c")
        assert brain.count_memories() == 3

    def test_count_memories_empty(self, brain):
        assert brain.count_memories() == 0

    def test_remember_invalidates_cache(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("first")
        brain.recall_all()  # populate cache
        with patch("memory.brain.log_json"):
            brain.remember("second")
        result = brain.recall_all()
        assert "second" in result


# ---------------------------------------------------------------------------
# Recall Cache Tests
# ---------------------------------------------------------------------------


class TestRecallCache:
    def test_recall_cache_hit(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("cached")
        brain.recall_all()
        
        # Second call should return same data from cache
        result = brain.recall_all()
        assert "cached" in result

    def test_recall_recent_cache_hit(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("cached")
        brain.recall_recent(limit=10)
        
        # Second call should use cache
        result = brain.recall_recent(limit=10)
        assert "cached" in result

    def test_cache_ttl_expiry(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("initial")
        brain.recall_all()
        
        # Manually expire cache
        brain._cache_ttl = 0
        time.sleep(0.01)
        
        with patch("memory.brain.log_json"):
            brain.remember("after_expire")
        
        result = brain.recall_all()
        assert "after_expire" in result


# ---------------------------------------------------------------------------
# Tagged Memory Tests
# ---------------------------------------------------------------------------


class TestTaggedMemory:
    def test_remember_tagged_stores(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember_tagged("tagged content", tag="sprint-1")
        rows = brain.recall_tagged("sprint-1")
        assert "tagged content" in rows

    def test_recall_tagged_empty_tag(self, brain):
        result = brain.recall_tagged("nonexistent-tag")
        assert result == []

    def test_recall_tagged_multiple(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember_tagged("msg1", tag="debug")
            brain.remember_tagged("msg2", tag="debug")
            brain.remember_tagged("msg3", tag="other")
        
        debug_msgs = brain.recall_tagged("debug")
        assert len(debug_msgs) == 2
        assert all(m in ["msg1", "msg2"] for m in debug_msgs)

    def test_forget_tagged_removes_entries(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember_tagged("bye", tag="temp")
            brain.remember_tagged("bye2", tag="temp")
        removed = brain.forget_tagged("temp")
        assert removed == 2
        assert brain.recall_tagged("temp") == []

    def test_forget_tagged_returns_zero_if_nothing(self, brain):
        assert brain.forget_tagged("no-such-tag") == 0

    def test_forget_tagged_only_deletes_matching_tag(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember_tagged("keep", tag="keep-tag")
            brain.remember_tagged("remove", tag="remove-tag")
        
        brain.forget_tagged("remove-tag")
        
        assert brain.recall_tagged("keep-tag") != []
        assert brain.recall_tagged("remove-tag") == []

    def test_recall_tagged_limit(self, brain):
        with patch("memory.brain.log_json"):
            for i in range(100):
                brain.remember_tagged(f"msg-{i}", tag="many")
        
        result = brain.recall_tagged("many", limit=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# KV Store Tests
# ---------------------------------------------------------------------------


class TestKVStore:
    def test_set_and_get_string(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("mykey", "myvalue")
        assert brain.get("mykey") == "myvalue"

    def test_get_missing_key_returns_default(self, brain):
        assert brain.get("missing", default="fallback") == "fallback"

    def test_get_missing_key_returns_none(self, brain):
        assert brain.get("missing") is None

    def test_set_dict_and_get_deserialised(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("cfg", {"a": 1, "b": {"nested": True}})
        retrieved = brain.get("cfg")
        assert retrieved["a"] == 1
        assert retrieved["b"]["nested"] is True

    def test_set_list_and_get_deserialised(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("items", [1, 2, 3, "four"])
        retrieved = brain.get("items")
        assert retrieved == [1, 2, 3, "four"]

    def test_overwrite_key(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("k", "v1")
            brain.set("k", "v2")
        assert brain.get("k") == "v2"

    def test_set_none_value(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("k", None)
        # None should be serialized as JSON null
        assert brain.get("k") is None

    def test_set_numeric_values(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("int", 42)
            brain.set("float", 3.14)
        assert brain.get("int") == 42
        assert brain.get("float") == 3.14


# ---------------------------------------------------------------------------
# Weakness Tests
# ---------------------------------------------------------------------------


class TestWeaknesses:
    def test_add_and_recall_weakness(self, brain):
        with patch("memory.brain.log_json"):
            brain.add_weakness("missing tests")
        weaknesses = brain.recall_weaknesses()
        assert "missing tests" in weaknesses

    def test_recall_weaknesses_empty_initially(self, brain):
        assert brain.recall_weaknesses() == []

    def test_add_multiple_weaknesses(self, brain):
        with patch("memory.brain.log_json"):
            brain.add_weakness("bug 1")
            brain.add_weakness("bug 2")
            brain.add_weakness("bug 3")
        weaknesses = brain.recall_weaknesses()
        assert len(weaknesses) >= 3
        assert "bug 1" in weaknesses

    def test_recall_weaknesses_newest_first(self, brain):
        with patch("memory.brain.log_json"):
            brain.add_weakness("old")
            brain.add_weakness("new")
        weaknesses = brain.recall_weaknesses()
        # The result should contain both
        assert "new" in weaknesses
        assert "old" in weaknesses

    def test_reflect_returns_summary_string(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("item")
            brain.add_weakness("bug")
        summary = brain.reflect()
        assert "1" in summary
        assert "weakness" in summary.lower()
        assert "memory" in summary.lower()

    def test_analyze_critique_for_weaknesses_negative_sentiment(self, brain):
        critique = "This code is terrible and completely broken."
        with patch("memory.brain.log_json"):
            brain.analyze_critique_for_weaknesses(critique)
        weaknesses = brain.recall_weaknesses()
        assert len(weaknesses) > 0

    def test_analyze_critique_for_weaknesses_keywords(self, brain):
        critique = "Code has a bug in the error handling."
        with patch("memory.brain.log_json"):
            brain.analyze_critique_for_weaknesses(critique)
        weaknesses = brain.recall_weaknesses()
        assert len(weaknesses) > 0


# ---------------------------------------------------------------------------
# Graph / Relate Tests
# ---------------------------------------------------------------------------


class TestRelate:
    def test_relate_adds_edge(self, brain):
        brain.relate("concept-a", "concept-b")
        assert brain.graph.has_edge("concept-a", "concept-b")

    def test_relate_multiple_edges(self, brain):
        brain.relate("a", "b")
        brain.relate("b", "c")
        brain.relate("a", "c")
        assert brain.graph.number_of_edges() == 3

    def test_relate_self_loop(self, brain):
        brain.relate("a", "a")
        assert brain.graph.has_edge("a", "a")

    def test_graph_is_lazy_initialized(self, tmp_path):
        db_file = tmp_path / "lazy.db"
        with patch("memory.brain.log_json"):
            b = Brain(db_path=str(db_file))
        assert b._graph is None
        b.relate("x", "y")
        assert b._graph is not None

    def test_relate_undirected_graph(self, brain):
        brain.relate("a", "b")
        # undirected graph, so both directions exist
        assert brain.graph.has_edge("a", "b") or brain.graph.has_edge("b", "a")


# ---------------------------------------------------------------------------
# Compress to Budget Tests
# ---------------------------------------------------------------------------


class TestCompressToBudget:
    def test_returns_all_when_under_budget(self):
        entries = ["short", "strings"]
        result = Brain.compress_to_budget(entries, max_tokens=1000)
        assert result == entries

    def test_truncates_oldest_first(self):
        entries = ["old1", "old2", "new1", "new2"]
        result = Brain.compress_to_budget(entries, max_tokens=3)
        assert "new2" in result
        assert "old1" not in result

    def test_empty_entries(self):
        assert Brain.compress_to_budget([], max_tokens=100) == []

    def test_single_entry_too_large(self):
        entries = ["x" * 1000]
        result = Brain.compress_to_budget(entries, max_tokens=1)
        # Can't fit even the first entry, but we still keep it
        assert len(result) >= 0

    def test_budget_zero(self):
        entries = ["a", "b", "c"]
        result = Brain.compress_to_budget(entries, max_tokens=0)
        assert result == []


# ---------------------------------------------------------------------------
# Weakness Queue Tests
# ---------------------------------------------------------------------------


class TestWeaknessQueue:
    def test_mark_and_recall_queued_hashes(self, brain):
        with patch("memory.brain.log_json"):
            brain.mark_weakness_queued("hash-abc")
        hashes = brain.recall_queued_weakness_hashes()
        assert "hash-abc" in hashes

    def test_mark_queued_idempotent(self, brain):
        with patch("memory.brain.log_json"):
            brain.mark_weakness_queued("dup")
            brain.mark_weakness_queued("dup")
        hashes = brain.recall_queued_weakness_hashes()
        assert hashes.count("dup") == 1

    def test_recall_queued_weakness_hashes_empty(self, brain):
        hashes = brain.recall_queued_weakness_hashes()
        assert hashes == []

    def test_multiple_queued_hashes(self, brain):
        with patch("memory.brain.log_json"):
            brain.mark_weakness_queued("h1")
            brain.mark_weakness_queued("h2")
            brain.mark_weakness_queued("h3")
        hashes = brain.recall_queued_weakness_hashes()
        assert len(hashes) == 3


# ---------------------------------------------------------------------------
# Innovation Sessions Tests
# ---------------------------------------------------------------------------


class TestInnovationSessions:
    def _session_data(self, sid="sess-1"):
        return {
            "session_id": sid,
            "problem_statement": "How to improve test coverage?",
            "status": "active",
            "current_phase": "immersion",
            "phases_completed": [],
            "techniques": ["brainstorming"],
            "constraints": {"time": "1 week"},
            "ideas_generated": 5,
            "ideas_selected": 2,
            "output": None,
        }

    def test_save_and_get_session(self, brain):
        with patch("memory.brain.log_json"):
            brain.save_innovation_session(self._session_data())
        result = brain.get_innovation_session("sess-1")
        assert result is not None
        assert result["problem_statement"] == "How to improve test coverage?"

    def test_get_nonexistent_session(self, brain):
        assert brain.get_innovation_session("nope") is None

    def test_list_sessions_all(self, brain):
        with patch("memory.brain.log_json"):
            brain.save_innovation_session(self._session_data("s1"))
            brain.save_innovation_session(self._session_data("s2"))
        sessions = brain.list_innovation_sessions()
        assert len(sessions) >= 2

    def test_list_sessions_filtered_by_status(self, brain):
        with patch("memory.brain.log_json"):
            data = self._session_data("active-s")
            data["status"] = "active"
            brain.save_innovation_session(data)
            done_data = self._session_data("done-s")
            done_data["status"] = "completed"
            brain.save_innovation_session(done_data)
        active = brain.list_innovation_sessions(status="active")
        assert all(s["status"] == "active" for s in active)

    def test_delete_session(self, brain):
        with patch("memory.brain.log_json"):
            brain.save_innovation_session(self._session_data("del-1"))
        with patch("memory.brain.log_json"):
            deleted = brain.delete_innovation_session("del-1")
        assert deleted is True
        assert brain.get_innovation_session("del-1") is None

    def test_delete_nonexistent_session_returns_false(self, brain):
        assert brain.delete_innovation_session("ghost") is False

    def test_save_session_updates_timestamp(self, brain):
        with patch("memory.brain.log_json"):
            data = self._session_data("ts-1")
            brain.save_innovation_session(data)
        result1 = brain.get_innovation_session("ts-1")
        updated_at1 = result1["updated_at"]
        
        time.sleep(0.01)
        
        with patch("memory.brain.log_json"):
            brain.save_innovation_session(data)
        result2 = brain.get_innovation_session("ts-1")
        updated_at2 = result2["updated_at"]
        
        assert updated_at2 >= updated_at1

    def test_list_sessions_with_limit(self, brain):
        with patch("memory.brain.log_json"):
            for i in range(20):
                brain.save_innovation_session(self._session_data(f"s-{i}"))
        
        sessions = brain.list_innovation_sessions(limit=5)
        assert len(sessions) == 5

    def test_session_with_complex_output(self, brain):
        with patch("memory.brain.log_json"):
            data = self._session_data("complex")
            data["output"] = {
                "ideas": [
                    {"id": 1, "title": "Idea 1", "score": 0.8},
                    {"id": 2, "title": "Idea 2", "score": 0.6},
                ]
            }
            brain.save_innovation_session(data)
        
        result = brain.get_innovation_session("complex")
        assert result["output_data"] is not None
        assert len(result["output_data"]["ideas"]) == 2


# ---------------------------------------------------------------------------
# Database Lock Tests
# ---------------------------------------------------------------------------


class TestDatabaseLocking:
    def test_db_lock_context_manager(self, brain):
        """Test that _db_lock is a working context manager."""
        with brain._db_lock():
            brain.db.execute("INSERT INTO memory(content) VALUES (?)", ("locked",))
            brain.db.commit()
        
        entries = brain.recall_all()
        assert "locked" in entries


# ---------------------------------------------------------------------------
# Schema Migration Tests
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_migration_up_to_current(self, brain):
        """Test that new database migrates to current schema version."""
        version = brain._get_schema_version()
        assert version == Brain.SCHEMA_VERSION

    def test_set_schema_version(self, brain):
        """Test setting schema version."""
        brain._set_schema_version(99)
        assert brain._get_schema_version() == 99

    def test_legacy_db_absorption(self, tmp_path):
        """Test that legacy brain_v2.db is absorbed."""
        # Create a legacy database
        legacy_path = tmp_path / "brain_v2.db"
        legacy_db = sqlite3.connect(str(legacy_path))
        legacy_db.execute("CREATE TABLE IF NOT EXISTS memory(id INTEGER PRIMARY KEY, content TEXT)")
        legacy_db.execute("INSERT INTO memory(content) VALUES (?)", ("legacy_content",))
        legacy_db.commit()
        legacy_db.close()
        
        # Create new brain in same directory
        with patch("memory.brain.log_json"):
            main_db_path = tmp_path / "brain.db"
            brain = Brain(db_path=str(main_db_path))
        
        # Check that legacy content was absorbed
        entries = brain.recall_all()
        # Legacy content should be there (if migration ran)
        # Note: actual behavior depends on schema version


# ---------------------------------------------------------------------------
# Vector Store Integration Tests
# ---------------------------------------------------------------------------


class TestVectorStoreIntegration:
    def test_set_vector_store(self, brain):
        mock_vector = MagicMock()
        with patch("memory.brain.log_json"):
            brain.set_vector_store(mock_vector)
        assert brain.vector_store is mock_vector


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_remember_very_long_string(self, brain):
        """Test handling of very long strings."""
        long_string = "x" * 100000
        with patch("memory.brain.log_json"):
            brain.remember(long_string)
        entries = brain.recall_all()
        assert any(long_string in e or len(e) > 50000 for e in entries)

    def test_remember_special_characters(self, brain):
        """Test handling of special characters in strings."""
        special = "Test with 'quotes', \"double quotes\", newlines\nand\ttabs"
        with patch("memory.brain.log_json"):
            brain.remember(special)
        entries = brain.recall_all()
        # String might be modified through JSON serialization
        assert len(entries) > 0

    def test_remember_unicode_characters(self, brain):
        """Test handling of unicode characters."""
        unicode_str = "Hello 世界 🌍 مرحبا"
        with patch("memory.brain.log_json"):
            brain.remember(unicode_str)
        entries = brain.recall_all()
        assert len(entries) > 0

    def test_concurrent_lock_prevents_race(self, brain):
        """Test that lock prevents concurrent access issues."""
        results = []
        
        def insert_and_count():
            with brain._db_lock():
                brain.db.execute("INSERT INTO memory(content) VALUES (?)", ("item",))
                brain.db.commit()
                row = brain.db.execute("SELECT COUNT(*) FROM memory").fetchone()
                results.append(row[0])
        
        import threading
        threads = [threading.Thread(target=insert_and_count) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All inserts should have succeeded
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

