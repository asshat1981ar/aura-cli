"""Tests for memory.brain.Brain — uses SQLite :memory: for isolation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from memory.brain import Brain


@pytest.fixture()
def brain(tmp_path):
    """Return a Brain instance backed by a temp file (avoids polluting cwd)."""
    db_file = tmp_path / "test_brain.db"
    with patch("memory.brain.log_json"):
        b = Brain(db_path=str(db_file))
    return b


# ---------------------------------------------------------------------------
# Initialisation / schema
# ---------------------------------------------------------------------------


class TestBrainInit:
    def test_creates_successfully(self, brain):
        assert brain is not None

    def test_schema_version_at_current(self, brain):
        assert brain._get_schema_version() == Brain.SCHEMA_VERSION

    def test_db_connection_is_open(self, brain):
        row = brain.db.execute("SELECT 1").fetchone()
        assert row[0] == 1

    def test_tables_created(self, brain):
        tables = {r[0] for r in brain.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for expected in ("memory", "weaknesses", "kv_store", "innovation_sessions"):
            assert expected in tables


# ---------------------------------------------------------------------------
# remember / recall
# ---------------------------------------------------------------------------


class TestRememberRecall:
    def test_remember_string(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("hello world")
        entries = brain.recall_all()
        assert "hello world" in entries

    def test_remember_dict_serialised(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember({"key": "val"})
        entries = brain.recall_all()
        assert any("key" in e for e in entries)

    def test_recall_all_returns_list(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("a")
            brain.remember("b")
        result = brain.recall_all()
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_recall_recent_limit(self, brain):
        with patch("memory.brain.log_json"):
            for i in range(10):
                brain.remember(f"item-{i}")
        recent = brain.recall_recent(limit=3)
        assert len(recent) == 3

    def test_recall_recent_newest_last(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("first")
            brain.remember("last")
        recent = brain.recall_recent(limit=2)
        assert recent[-1] == "last"

    def test_recall_with_budget_respects_tokens(self, brain):
        with patch("memory.brain.log_json"):
            for i in range(50):
                brain.remember("x" * 100)
        result = brain.recall_with_budget(max_tokens=10)
        total_chars = sum(len(e) for e in result)
        assert total_chars <= 10 * 4 + len(result)  # slight slack for separators

    def test_recall_cache_hit(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("cached")
        brain.recall_all()
        # Second call should return same data from cache
        result = brain.recall_all()
        assert "cached" in result

    def test_count_memories(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("a")
            brain.remember("b")
            brain.remember("c")
        assert brain.count_memories() == 3


# ---------------------------------------------------------------------------
# Tagged memory
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

    def test_forget_tagged_removes_entries(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember_tagged("bye", tag="temp")
            brain.remember_tagged("bye2", tag="temp")
        removed = brain.forget_tagged("temp")
        assert removed == 2
        assert brain.recall_tagged("temp") == []

    def test_forget_tagged_returns_zero_if_nothing(self, brain):
        assert brain.forget_tagged("no-such-tag") == 0


# ---------------------------------------------------------------------------
# KV store
# ---------------------------------------------------------------------------


class TestKVStore:
    def test_set_and_get_string(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("mykey", "myvalue")
        assert brain.get("mykey") == "myvalue"

    def test_get_missing_key_returns_default(self, brain):
        assert brain.get("missing", default="fallback") == "fallback"

    def test_set_dict_and_get_deserialised(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("cfg", {"a": 1})
        assert brain.get("cfg") == {"a": 1}

    def test_overwrite_key(self, brain):
        with patch("memory.brain.log_json"):
            brain.set("k", "v1")
            brain.set("k", "v2")
        assert brain.get("k") == "v2"


# ---------------------------------------------------------------------------
# Weaknesses
# ---------------------------------------------------------------------------


class TestWeaknesses:
    def test_add_and_recall_weakness(self, brain):
        with patch("memory.brain.log_json"):
            brain.add_weakness("missing tests")
        weaknesses = brain.recall_weaknesses()
        assert "missing tests" in weaknesses

    def test_recall_weaknesses_empty_initially(self, brain):
        assert brain.recall_weaknesses() == []

    def test_reflect_returns_summary_string(self, brain):
        with patch("memory.brain.log_json"):
            brain.remember("item")
            brain.add_weakness("bug")
        summary = brain.reflect()
        assert "1" in summary
        assert "weakness" in summary.lower()


# ---------------------------------------------------------------------------
# Graph / relate
# ---------------------------------------------------------------------------


class TestRelate:
    def test_relate_adds_edge(self, brain):
        brain.relate("concept-a", "concept-b")
        assert brain.graph.has_edge("concept-a", "concept-b")

    def test_relate_multiple_edges(self, brain):
        brain.relate("a", "b")
        brain.relate("b", "c")
        assert brain.graph.number_of_edges() == 2

    def test_graph_is_lazy_initialized(self, tmp_path):
        db_file = tmp_path / "lazy.db"
        with patch("memory.brain.log_json"):
            b = Brain(db_path=str(db_file))
        assert b._graph is None
        b.relate("x", "y")
        assert b._graph is not None


# ---------------------------------------------------------------------------
# compress_to_budget (static)
# ---------------------------------------------------------------------------


class TestCompressToBudget:
    def test_returns_all_when_under_budget(self):
        entries = ["short", "strings"]
        result = Brain.compress_to_budget(entries, max_tokens=1000)
        assert result == entries

    def test_truncates_oldest_first(self):
        # Budget: 3 tokens = 12 chars. Each entry is 4 chars + 1 sep = 5.
        # Room for ~2 entries → only newest two fit.
        entries = ["old1", "old2", "new1", "new2"]
        result = Brain.compress_to_budget(entries, max_tokens=3)
        assert "new2" in result
        assert "old1" not in result

    def test_empty_entries(self):
        assert Brain.compress_to_budget([], max_tokens=100) == []


# ---------------------------------------------------------------------------
# Weakness queue
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


# ---------------------------------------------------------------------------
# Innovation sessions
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
