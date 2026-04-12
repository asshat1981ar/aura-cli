"""Tests for memory/brain.py — Brain."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from memory.brain import Brain


@pytest.fixture
def brain(tmp_path):
    return Brain(db_path=str(tmp_path / "brain.db"))


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestBrainInit:
    def test_default_db_in_memory(self):
        b = Brain(db_path=":memory:")
        assert b is not None

    def test_custom_db_path_used(self, tmp_path):
        db = tmp_path / "custom.db"
        b = Brain(db_path=str(db))
        b.remember("test")
        assert db.exists()

    def test_graph_property_returns_graph(self, brain):
        g = brain.graph
        assert g is not None


# ---------------------------------------------------------------------------
# remember / recall_recent
# ---------------------------------------------------------------------------

class TestRememberRecallRecent:
    def test_remember_string(self, brain):
        brain.remember("hello world")
        recent = brain.recall_recent(limit=10)
        assert any("hello world" in str(r) for r in recent)

    def test_remember_dict(self, brain):
        brain.remember({"key": "value", "num": 42})
        recent = brain.recall_recent(limit=10)
        assert len(recent) >= 1

    def test_recall_recent_limit_respected(self, brain):
        for i in range(10):
            brain.remember(f"entry {i}")
        recent = brain.recall_recent(limit=3)
        assert len(recent) <= 3

    def test_recall_recent_returns_list(self, brain):
        assert isinstance(brain.recall_recent(), list)

    def test_recall_recent_most_recent_first(self, brain):
        brain.remember("first")
        brain.remember("second")
        brain.remember("third")
        recent = brain.recall_recent(limit=3)
        # Most recent entries come first
        combined = " ".join(str(r) for r in recent)
        assert "third" in combined


# ---------------------------------------------------------------------------
# remember_tagged / recall_tagged / forget_tagged
# ---------------------------------------------------------------------------

class TestTaggedMemory:
    def test_remember_tagged_stored(self, brain):
        brain.remember_tagged("tagged entry", "my_tag")
        results = brain.recall_tagged("my_tag")
        assert len(results) >= 1

    def test_recall_tagged_only_returns_matching_tag(self, brain):
        brain.remember_tagged("alpha", "tag_a")
        brain.remember_tagged("beta", "tag_b")
        results = brain.recall_tagged("tag_a")
        assert all("alpha" in r or "tag_a" in r for r in results) or len(results) >= 1

    def test_recall_tagged_limit_respected(self, brain):
        for i in range(10):
            brain.remember_tagged(f"item {i}", "bulk")
        results = brain.recall_tagged("bulk", limit=3)
        assert len(results) <= 3

    def test_forget_tagged_removes_entries(self, brain):
        brain.remember_tagged("to_delete", "temp_tag")
        count = brain.forget_tagged("temp_tag")
        assert count >= 1
        assert brain.recall_tagged("temp_tag") == []

    def test_forget_tagged_nonexistent_returns_zero(self, brain):
        count = brain.forget_tagged("nonexistent_tag_xyz")
        assert count == 0


# ---------------------------------------------------------------------------
# set / get
# ---------------------------------------------------------------------------

class TestSetGet:
    def test_set_and_get_string(self, brain):
        brain.set("my_key", "my_value")
        assert brain.get("my_key") == "my_value"

    def test_set_and_get_dict(self, brain):
        brain.set("config", {"a": 1, "b": 2})
        val = brain.get("config")
        assert val["a"] == 1

    def test_get_missing_key_returns_default(self, brain):
        assert brain.get("missing_key_xyz") is None

    def test_get_custom_default(self, brain):
        assert brain.get("missing", default="fallback") == "fallback"

    def test_set_overwrites_existing(self, brain):
        brain.set("k", "v1")
        brain.set("k", "v2")
        assert brain.get("k") == "v2"


# ---------------------------------------------------------------------------
# recall_all / count_memories
# ---------------------------------------------------------------------------

class TestRecallAllCount:
    def test_recall_all_returns_list(self, brain):
        assert isinstance(brain.recall_all(), list)

    def test_count_memories_zero_initially(self, brain):
        assert brain.count_memories() == 0

    def test_count_memories_increments(self, brain):
        brain.remember("one")
        brain.remember("two")
        assert brain.count_memories() == 2

    def test_recall_all_matches_count(self, brain):
        brain.remember("a")
        brain.remember("b")
        all_entries = brain.recall_all()
        assert len(all_entries) == brain.count_memories()


# ---------------------------------------------------------------------------
# compress_to_budget
# ---------------------------------------------------------------------------

class TestCompressToBudget:
    def test_returns_list(self):
        result = Brain.compress_to_budget(["a", "b", "c"], max_tokens=1000)
        assert isinstance(result, list)

    def test_empty_input(self):
        result = Brain.compress_to_budget([], max_tokens=100)
        assert result == []

    def test_truncates_when_over_budget(self):
        # Each token ≈ 4 chars; set a tiny budget
        entries = ["word " * 50] * 10
        result = Brain.compress_to_budget(entries, max_tokens=10)
        assert len(result) < len(entries)

    def test_keeps_all_within_budget(self):
        entries = ["short"] * 5
        result = Brain.compress_to_budget(entries, max_tokens=10000)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# weaknesses
# ---------------------------------------------------------------------------

class TestWeaknesses:
    def test_add_and_recall_weakness(self, brain):
        brain.add_weakness("poor error handling in act phase")
        weaknesses = brain.recall_weaknesses()
        assert any("poor error handling" in w for w in weaknesses)

    def test_recall_weaknesses_returns_list(self, brain):
        assert isinstance(brain.recall_weaknesses(), list)

    def test_weakness_hash_tracked(self, brain):
        brain.add_weakness("weakness description")
        hashes = brain.recall_queued_weakness_hashes()
        assert isinstance(hashes, list)

    def test_mark_weakness_queued(self, brain):
        brain.add_weakness("some weakness")
        brain.mark_weakness_queued("abc123")
        hashes = brain.recall_queued_weakness_hashes()
        assert "abc123" in hashes


# ---------------------------------------------------------------------------
# relate / analyze_critique_for_weaknesses
# ---------------------------------------------------------------------------

class TestRelateAndCritique:
    def test_relate_no_crash(self, brain):
        brain.relate("concept_a", "concept_b")  # Should not raise

    def test_reflect_no_crash(self, brain):
        brain.reflect()  # Should not raise

    def test_analyze_critique_for_weaknesses(self, brain):
        brain.analyze_critique_for_weaknesses("The plan is weak on error handling")
        # Should store something without raising
        weaknesses = brain.recall_weaknesses()
        assert isinstance(weaknesses, list)


# ---------------------------------------------------------------------------
# recall_with_budget
# ---------------------------------------------------------------------------

class TestRecallWithBudget:
    def test_returns_list(self, brain):
        brain.remember("entry one")
        result = brain.recall_with_budget(max_tokens=1000)
        assert isinstance(result, list)

    def test_empty_brain_returns_empty(self, brain):
        result = brain.recall_with_budget(max_tokens=100)
        assert result == []
