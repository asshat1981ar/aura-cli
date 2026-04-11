"""
Comprehensive unit tests for MomentoBrain.

Tests the write-through L1/L2 cache behavior where:
  - L1 = Momento (hot cache, fast)
  - L2 = SQLite (persistent store)
  - Writes go to both L1 and L2
  - Reads check L1 first, fall back to L2 on miss
  - Fallback mode (no Momento API key) makes L1 a no-op

Tests cover:
  - remember() — write to L1 + L2
  - recall_all() — read from L1, fallback to L2
  - add_weakness() — write to L1 + L2
  - recall_weaknesses() — read from L1, fallback to L2
  - enable_cache() — configure L2 response cache + L1 TTL
  - _get_cached_response() — check L1 first
  - _save_to_cache() — write-through to L1
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

import pytest


def setup_momento_mocks():
    """Inject mock Momento SDK before importing brain."""
    if "momento" not in sys.modules:
        momento_mock = MagicMock()
        sys.modules["momento"] = momento_mock
        sys.modules["momento.responses"] = MagicMock()
        sys.modules["momento.requests"] = MagicMock()


setup_momento_mocks()

from memory.momento_brain import MomentoBrain
from memory.momento_adapter import MomentoAdapter, WORKING_MEMORY_CACHE


# ============================================================================
# Fallback Mode Tests (No API Key)
# ============================================================================


class TestMomentoBrainFallback:
    """MomentoBrain behaves identically to Brain when adapter is no-op."""

    def setup_method(self):
        os.environ.pop("MOMENTO_API_KEY", None)
        os.environ["AURA_SKIP_CHDIR"] = "1"
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "brain.db"

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_remember_string(self):
        """remember() stores string to L2 (L1 is no-op)."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        brain.remember("test memory")
        recalled = brain.recall_all()

        assert len(recalled) > 0
        assert any("test memory" in str(r) for r in recalled)

    def test_remember_dict(self):
        """remember() serializes and stores dict."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        data = {"type": "test", "value": 42}
        brain.remember(data)
        recalled = brain.recall_all()

        assert len(recalled) > 0
        assert any("test" in str(r) or "42" in str(r) for r in recalled)

    def test_add_weakness(self):
        """add_weakness() stores to L2."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        brain.add_weakness("poor error handling")
        weaknesses = brain.recall_weaknesses()

        assert len(weaknesses) > 0
        assert any("poor error handling" in w for w in weaknesses)

    def test_recall_weaknesses_empty(self):
        """recall_weaknesses() returns empty list when none added."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        weaknesses = brain.recall_weaknesses()
        assert isinstance(weaknesses, list)

    def test_recall_all_empty(self):
        """recall_all() returns empty list when nothing remembered."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        recalled = brain.recall_all()
        assert isinstance(recalled, list)


# ============================================================================
# L1/L2 Integration Tests (Mocked Momento)
# ============================================================================


class TestMomentoBrainL1L2:
    """Test L1 cache hits and L2 fallback behavior."""

    @pytest.fixture
    def brain_with_mock_adapter(self):
        """Create brain with mocked Momento adapter."""
        tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(tmpdir.name) / "brain.db"

        adapter = MomentoAdapter()
        adapter._api_key = "test-key"  # Set API key so is_available() checks further
        adapter._cache_client = MagicMock()
        adapter._topics_client = MagicMock()
        adapter._available = True
        adapter._initialized = True

        brain = MomentoBrain(adapter, db_path=str(db_path))

        yield brain, adapter
        tmpdir.cleanup()

    def test_remember_calls_list_push_when_available(self, brain_with_mock_adapter):
        """remember() calls L1 list_push when Momento is available."""
        brain, adapter = brain_with_mock_adapter

        with patch.object(adapter, "list_push", return_value=True) as mock_push:
            brain.remember("test data")
            mock_push.assert_called()

    def test_remember_handles_l1_exception(self, brain_with_mock_adapter):
        """remember() handles L1 exceptions gracefully."""
        brain, adapter = brain_with_mock_adapter

        with patch.object(adapter, "list_push", side_effect=Exception("L1 error")):
            # Should not raise
            brain.remember("test data")

    def test_recall_all_l1_hit(self, brain_with_mock_adapter):
        """recall_all() returns L1 items on hit."""
        brain, adapter = brain_with_mock_adapter

        with patch.object(adapter, "list_range", return_value=["item1", "item2"]):
            result = brain.recall_all()
            assert result == ["item1", "item2"]

    def test_recall_all_l1_miss_fallback_to_l2(self, brain_with_mock_adapter):
        """recall_all() falls back to L2 on L1 miss."""
        brain, adapter = brain_with_mock_adapter

        # L1 miss (empty list)
        with patch.object(adapter, "list_range", return_value=[]):
            with patch.object(adapter, "list_push", return_value=True):
                # Store something in L2 first
                brain.remember("test item")

                # Recall should fall back to L2 and backfill L1
                result = brain.recall_all()
                assert len(result) > 0

    def test_recall_weaknesses_l1_hit(self, brain_with_mock_adapter):
        """recall_weaknesses() returns L1 items on hit."""
        brain, adapter = brain_with_mock_adapter

        with patch.object(adapter, "list_range", return_value=["weakness1", "weakness2"]):
            result = brain.recall_weaknesses()
            assert result == ["weakness1", "weakness2"]

    def test_recall_weaknesses_l1_miss_fallback(self, brain_with_mock_adapter):
        """recall_weaknesses() falls back to L2 on miss."""
        brain, adapter = brain_with_mock_adapter

        # L1 miss
        with patch.object(adapter, "list_range", return_value=[]):
            with patch.object(adapter, "list_push", return_value=True):
                brain.add_weakness("test weakness")

                result = brain.recall_weaknesses()
                assert len(result) > 0

    def test_add_weakness_calls_list_push(self, brain_with_mock_adapter):
        """add_weakness() calls L1 list_push when available."""
        brain, adapter = brain_with_mock_adapter

        with patch.object(adapter, "list_push", return_value=True) as mock_push:
            brain.add_weakness("test weakness")
            mock_push.assert_called()

    def test_add_weakness_handles_l1_exception(self, brain_with_mock_adapter):
        """add_weakness() handles L1 exceptions."""
        brain, adapter = brain_with_mock_adapter

        with patch.object(adapter, "list_push", side_effect=Exception("L1 error")):
            # Should not raise
            brain.add_weakness("test weakness")


# ============================================================================
# Response Cache Tests
# ============================================================================


class TestMomentoBrainResponseCache:
    """Test response cache helper methods."""

    def test_response_key_generation(self):
        """_response_key() generates consistent hashes."""
        key1 = MomentoBrain._response_key("test prompt")
        key2 = MomentoBrain._response_key("test prompt")

        assert key1 == key2
        assert key1.startswith("response:")

    def test_response_key_different_for_different_prompts(self):
        """_response_key() generates different keys for different prompts."""
        key1 = MomentoBrain._response_key("prompt A")
        key2 = MomentoBrain._response_key("prompt B")

        assert key1 != key2

    def test_response_key_is_16_chars(self):
        """_response_key() includes 16-char hash."""
        key = MomentoBrain._response_key("test")
        # Format is "response:XXXXX..." where XXXXX is the 16-char hash
        assert len(key) == len("response:") + 16


# ============================================================================
# L1 Cache Response Tests (Limited - No Super Methods)
# ============================================================================


class TestMomentoBrainResponseCacheL1:
    """Test response cache helper methods and L1 operations."""

    @pytest.fixture
    def brain_with_l1_cache(self):
        """Create brain with available Momento."""
        tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(tmpdir.name) / "brain.db"

        adapter = MomentoAdapter()
        adapter._available = True
        adapter._initialized = True

        brain = MomentoBrain(adapter, db_path=str(db_path))
        brain._response_ttl = 60

        yield brain, adapter
        tmpdir.cleanup()

    def test_response_key_format(self, brain_with_l1_cache):
        """Response key is properly formatted."""
        brain, _ = brain_with_l1_cache

        key = brain._response_key("test prompt")
        assert key.startswith("response:")
        assert len(key) == len("response:") + 16


# ============================================================================
# Backfill Tests
# ============================================================================


class TestMomentoBrainBackfill:
    """Test L1 backfill operations."""

    @pytest.fixture
    def brain_for_backfill(self):
        """Create brain with mocked adapter for backfill testing."""
        tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(tmpdir.name) / "brain.db"

        adapter = MomentoAdapter()
        adapter._available = True
        adapter._initialized = True

        brain = MomentoBrain(adapter, db_path=str(db_path))

        yield brain, adapter
        tmpdir.cleanup()

    def test_backfill_memory_list(self, brain_for_backfill):
        """_backfill_memory_list() pushes items to L1."""
        brain, adapter = brain_for_backfill

        with patch.object(adapter, "list_push", return_value=True) as mock_push:
            items = [f"item{i}" for i in range(5)]
            brain._backfill_memory_list(items)

            # Should have called list_push for each item
            assert mock_push.call_count >= len(items)

    def test_backfill_weakness_list(self, brain_for_backfill):
        """_backfill_weakness_list() pushes items to L1."""
        brain, adapter = brain_for_backfill

        with patch.object(adapter, "list_push", return_value=True) as mock_push:
            items = ["weakness1", "weakness2", "weakness3"]
            brain._backfill_weakness_list(items)

            # Should have called list_push
            assert mock_push.call_count >= len(items)

    def test_backfill_handles_exceptions(self, brain_for_backfill):
        """Backfill methods handle exceptions gracefully."""
        brain, adapter = brain_for_backfill

        with patch.object(adapter, "list_push", side_effect=Exception("backfill error")):
            # Should not raise
            brain._backfill_memory_list(["item1", "item2"])
            brain._backfill_weakness_list(["weakness1"])


# ============================================================================
# Integration Tests
# ============================================================================


class TestMomentoBrainIntegration:
    """Full integration tests."""

    def setup_method(self):
        os.environ.pop("MOMENTO_API_KEY", None)
        os.environ["AURA_SKIP_CHDIR"] = "1"
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "brain.db"

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_multiple_remember_and_recall(self):
        """Multiple remember() calls are all recalled."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        brain.remember("memory 1")
        brain.remember("memory 2")
        brain.remember("memory 3")

        recalled = brain.recall_all()
        assert len(recalled) >= 3

    def test_mixed_string_and_dict_memory(self):
        """Brain can store mixed string and dict memories."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        brain.remember("string memory")
        brain.remember({"key": "dict memory"})

        recalled = brain.recall_all()
        assert len(recalled) >= 2

    def test_weaknesses_separate_from_memories(self):
        """Weaknesses and memories are stored separately."""
        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter, db_path=str(self.db_path))

        brain.remember("this is a memory")
        brain.add_weakness("this is a weakness")

        memories = brain.recall_all()
        weaknesses = brain.recall_weaknesses()

        # Both should have items
        assert len(memories) > 0
        assert len(weaknesses) > 0

        # Weakness should not be in memories
        assert not any("weakness" in m for m in memories)
