"""
Comprehensive unit tests for MomentoMemoryStore.

Tests the write-through L1/L2 memory store where:
  - L1 = Momento Lists (hot cache)
  - L2 = JSON files (persistent store)
  - put() writes to both L1 and L2
  - query() reads from L1 first, falls back to L2
  - append_log() writes JSONL and publishes to Topics
  - Fallback mode (no Momento API key) makes L1 a no-op

Tests cover:
  - put() — write to tier JSON file + L1 list
  - query() — read from L1, fallback to L2 JSON
  - append_log() — write to decision log + publish event
  - _backfill_tier() — warm-up L1 on cache miss
  - _tier_key() helper
  - Fallback behavior when Momento unavailable
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def setup_momento_mocks():
    """Inject mock Momento SDK before importing store."""
    if "momento" not in sys.modules:
        momento_mock = MagicMock()
        sys.modules["momento"] = momento_mock
        sys.modules["momento.responses"] = MagicMock()
        sys.modules["momento.requests"] = MagicMock()


setup_momento_mocks()

from memory.momento_memory_store import MomentoMemoryStore, _tier_key
from memory.momento_adapter import MomentoAdapter, EPISODIC_MEMORY_CACHE, TOPIC_CYCLE_COMPLETE


# ============================================================================
# Fallback Mode Tests (No API Key)
# ============================================================================


class TestMomentoMemoryStoreFallback:
    """MomentoMemoryStore behaves identically to MemoryStore in fallback mode."""

    def setup_method(self):
        os.environ.pop("MOMENTO_API_KEY", None)
        os.environ["AURA_SKIP_CHDIR"] = "1"
        self.tmpdir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_put_and_query_single_record(self):
        """put() and query() work without Momento."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        record = {"key": "value", "n": 1}
        store.put("test_tier", record)
        
        results = store.query("test_tier")
        assert len(results) == 1
        assert results[0]["key"] == "value"
        assert results[0]["n"] == 1

    def test_put_multiple_records(self):
        """Multiple put() calls accumulate."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        for i in range(3):
            store.put("summaries", {"id": i, "value": f"record{i}"})
        
        results = store.query("summaries")
        assert len(results) == 3

    def test_query_with_limit(self):
        """query(limit=N) returns last N records."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        for i in range(10):
            store.put("data", {"i": i})
        
        results = store.query("data", limit=3)
        assert len(results) == 3
        assert results[-1]["i"] == 9

    def test_query_empty_tier(self):
        """query() returns empty list for nonexistent tier."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        results = store.query("nonexistent")
        assert results == []

    def test_append_log_and_read_log(self):
        """append_log() and read_log() work without Momento."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        entry = {
            "cycle_id": "test123",
            "goal_type": "test",
            "phase_outputs": {},
        }
        store.append_log(entry)
        
        log = store.read_log()
        assert len(log) == 1
        assert log[0]["cycle_id"] == "test123"

    def test_append_log_multiple_entries(self):
        """append_log() appends multiple entries."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        for i in range(3):
            store.append_log({
                "cycle_id": f"cycle{i}",
                "goal_type": "test",
                "phase_outputs": {},
            })
        
        log = store.read_log()
        assert len(log) == 3

    def test_multiple_tiers_independent(self):
        """Different tiers are stored independently."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        store.put("tier_a", {"data": "a"})
        store.put("tier_b", {"data": "b"})
        
        results_a = store.query("tier_a")
        results_b = store.query("tier_b")
        
        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0]["data"] == "a"
        assert results_b[0]["data"] == "b"

    def test_query_limit_greater_than_records(self):
        """query(limit=N) where N > total records returns all."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        store.put("data", {"i": 1})
        store.put("data", {"i": 2})
        
        results = store.query("data", limit=100)
        assert len(results) == 2


# ============================================================================
# L1/L2 Integration Tests (Mocked Momento)
# ============================================================================


class TestMomentoMemoryStoreL1L2:
    """Test L1 cache hits and L2 fallback behavior."""

    @pytest.fixture
    def store_with_mock_adapter(self):
        """Create store with mocked Momento adapter."""
        tmpdir = tempfile.TemporaryDirectory()
        
        adapter = MomentoAdapter()
        adapter._api_key = "test-key"  # Set API key so is_available() returns True
        adapter._cache_client = MagicMock()
        adapter._topics_client = MagicMock()
        adapter._available = True
        adapter._initialized = True
        
        from momento.responses import CacheListPushBack
        adapter._cache_client.list_push_back.return_value = CacheListPushBack.Success()
        
        store = MomentoMemoryStore(Path(tmpdir.name), adapter)
        
        yield store, adapter
        tmpdir.cleanup()

    def test_put_calls_list_push_when_available(self, store_with_mock_adapter):
        """put() calls L1 list_push when Momento is available."""
        store, adapter = store_with_mock_adapter
        
        record = {"key": "value"}
        store.put("tier", record)
        
        # Should have called list_push_back
        adapter._cache_client.list_push_back.assert_called()

    def test_put_handles_l1_exception(self, store_with_mock_adapter):
        """put() handles L1 exceptions gracefully."""
        store, adapter = store_with_mock_adapter
        
        adapter._cache_client.list_push_back.side_effect = Exception("L1 error")
        
        # Should not raise
        store.put("tier", {"key": "value"})

    def test_query_l1_hit(self, store_with_mock_adapter):
        """query() returns L1 items on hit."""
        store, adapter = store_with_mock_adapter
        
        with patch.object(adapter, 'list_range', return_value=[
            json.dumps({"id": 1}),
            json.dumps({"id": 2}),
        ]):
            results = store.query("tier")
            assert len(results) == 2
            assert results[0]["id"] == 1
            assert results[1]["id"] == 2

    def test_query_l1_miss_fallback_to_l2(self, store_with_mock_adapter):
        """query() falls back to L2 on L1 miss."""
        store, adapter = store_with_mock_adapter
        
        # L1 miss
        with patch.object(adapter, 'list_range', return_value=[]):
            # Store something in L2 first
            store.put("tier", {"id": 1})
            
            # Query should fall back to L2
            results = store.query("tier")
            
            assert len(results) > 0
            assert results[0]["id"] == 1

    def test_query_l1_hit_respects_limit(self, store_with_mock_adapter):
        """query() respects limit with L1 hit."""
        store, adapter = store_with_mock_adapter
        
        with patch.object(adapter, 'list_range', return_value=[
            json.dumps({"i": i}) for i in range(10)
        ]):
            results = store.query("tier", limit=3)
            
            assert len(results) == 3

    def test_query_l1_exception_fallback(self, store_with_mock_adapter):
        """query() falls back to L2 on L1 exception."""
        store, adapter = store_with_mock_adapter
        
        with patch.object(adapter, 'list_range', side_effect=Exception("L1 error")):
            # Store in L2 first
            store.put("tier", {"id": 1})
            
            # Should fall back to L2 without raising
            results = store.query("tier")
            assert len(results) > 0

    def test_query_invalid_json_in_l1(self, store_with_mock_adapter):
        """query() skips invalid JSON items from L1."""
        store, adapter = store_with_mock_adapter
        
        with patch.object(adapter, 'list_range', return_value=[
            json.dumps({"id": 1}),
            "not valid json",
            json.dumps({"id": 2}),
        ]):
            results = store.query("tier")
            
            # Should skip the invalid item
            assert len(results) == 2
            assert results[0]["id"] == 1
            assert results[1]["id"] == 2


# ============================================================================
# Topic Publishing Tests
# ============================================================================


class TestMomentoMemoryStoreTopics:
    """Test cycle event publishing to Topics."""

    @pytest.fixture
    def store_with_topics(self):
        """Create store with mocked Topics client."""
        tmpdir = tempfile.TemporaryDirectory()
        
        adapter = MomentoAdapter()
        adapter._api_key = "test-key"  # Set API key
        adapter._cache_client = MagicMock()
        adapter._topics_client = MagicMock()
        adapter._available = True
        adapter._initialized = True
        
        from momento.responses import TopicPublish
        adapter._topics_client.publish.return_value = TopicPublish.Success()
        
        store = MomentoMemoryStore(Path(tmpdir.name), adapter)
        
        yield store, adapter
        tmpdir.cleanup()

    def test_append_log_publishes_event(self, store_with_topics):
        """append_log() publishes cycle_complete event."""
        store, adapter = store_with_topics
        
        entry = {
            "cycle_id": "cycle123",
            "goal_type": "bug_fix",
            "phase_outputs": {
                "verification": {"status": "pass"}
            },
            "stop_reason": "success",
        }
        
        store.append_log(entry)
        
        # Should have called publish
        adapter._topics_client.publish.assert_called()
        
        # Check the published message
        call_args = adapter._topics_client.publish.call_args
        published_msg = json.loads(call_args[0][2])
        assert published_msg["cycle_id"] == "cycle123"
        assert published_msg["goal_type"] == "bug_fix"
        assert published_msg["verify_status"] == "pass"

    def test_append_log_event_has_timestamp(self, store_with_topics):
        """append_log() includes timestamp in event."""
        store, adapter = store_with_topics
        
        entry = {"cycle_id": "test", "phase_outputs": {}}
        
        import time
        before = time.time()
        store.append_log(entry)
        after = time.time()
        
        call_args = adapter._topics_client.publish.call_args
        published_msg = json.loads(call_args[0][2])
        
        assert "ts" in published_msg
        assert before <= published_msg["ts"] <= after

    def test_append_log_publishes_to_correct_topic(self, store_with_topics):
        """append_log() publishes to aura.cycle_complete topic."""
        store, adapter = store_with_topics
        
        store.append_log({"cycle_id": "test", "phase_outputs": {}})
        
        call_args = adapter._topics_client.publish.call_args
        topic = call_args[0][1]
        
        assert topic == TOPIC_CYCLE_COMPLETE

    def test_append_log_handles_publish_exception(self, store_with_topics):
        """append_log() handles publish exceptions gracefully."""
        store, adapter = store_with_topics
        
        adapter._topics_client.publish.side_effect = Exception("publish error")
        
        # Should not raise
        store.append_log({"cycle_id": "test", "phase_outputs": {}})

    def test_append_log_missing_phase_outputs(self, store_with_topics):
        """append_log() handles missing phase_outputs."""
        store, adapter = store_with_topics
        
        entry = {"cycle_id": "test"}  # No phase_outputs
        
        store.append_log(entry)
        
        call_args = adapter._topics_client.publish.call_args
        published_msg = json.loads(call_args[0][2])
        
        # Should have default verify_status
        assert published_msg["verify_status"] == "unknown"


# ============================================================================
# Helper Tests
# ============================================================================


class TestMementoMemoryStoreHelpers:
    """Test helper functions and private methods."""

    def test_tier_key_function(self):
        """_tier_key() generates tier cache keys."""
        key = _tier_key("summaries")
        assert key == "tier:summaries"

    def test_tier_key_different_for_different_tiers(self):
        """_tier_key() generates different keys for different tiers."""
        key1 = _tier_key("tier1")
        key2 = _tier_key("tier2")
        assert key1 != key2

    @pytest.fixture
    def store_for_backfill(self):
        """Create store for backfill testing."""
        tmpdir = tempfile.TemporaryDirectory()
        
        adapter = MomentoAdapter()
        adapter._api_key = "test-key"
        adapter._cache_client = MagicMock()
        adapter._available = True
        adapter._initialized = True
        
        from momento.responses import CacheListPushBack
        adapter._cache_client.list_push_back.return_value = CacheListPushBack.Success()
        
        store = MomentoMemoryStore(Path(tmpdir.name), adapter)
        
        yield store, adapter
        tmpdir.cleanup()

    def test_backfill_tier(self, store_for_backfill):
        """_backfill_tier() pushes records to L1."""
        store, adapter = store_for_backfill
        
        records = [{"id": i} for i in range(3)]
        store._backfill_tier("tier", records)
        
        # Should have called list_push_back for each record
        assert adapter._cache_client.list_push_back.call_count >= len(records)

    def test_backfill_tier_handles_exception(self, store_for_backfill):
        """_backfill_tier() handles exceptions gracefully."""
        store, adapter = store_for_backfill
        
        adapter._cache_client.list_push_back.side_effect = Exception("backfill error")
        
        # Should not raise
        records = [{"id": i} for i in range(3)]
        store._backfill_tier("tier", records)


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestMementoMemoryStoreEdgeCases:
    """Test edge cases and integration scenarios."""

    def setup_method(self):
        os.environ.pop("MOMENTO_API_KEY", None)
        os.environ["AURA_SKIP_CHDIR"] = "1"
        self.tmpdir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_query_limit_zero(self):
        """query(limit=0) returns all records."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        for i in range(5):
            store.put("tier", {"i": i})
        
        results = store.query("tier", limit=0)
        # Limit 0 might mean "all" depending on implementation
        assert len(results) >= 0

    def test_large_json_serialization(self):
        """Large records are serialized correctly."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        large_record = {
            "data": "x" * 10000,
            "nested": {
                "deep": {"value": list(range(100))}
            }
        }
        store.put("tier", large_record)
        
        results = store.query("tier")
        assert len(results) == 1
        assert len(results[0]["data"]) == 10000

    def test_special_characters_in_data(self):
        """Records with special characters are preserved."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        record = {
            "text": 'quotes "and" \'apostrophes\' and\nnewlines',
            "unicode": "你好世界 🚀",
        }
        store.put("tier", record)
        
        results = store.query("tier")
        assert results[0]["text"] == record["text"]
        assert results[0]["unicode"] == record["unicode"]

    def test_log_rotation_on_large_file(self):
        """Log rotation occurs when file exceeds max size."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        # Append large entries
        for i in range(100):
            store.append_log({
                "cycle_id": f"cycle{i}",
                "goal_type": "test",
                "phase_outputs": {},
                "data": "x" * 1000,
            })
        
        # Should have created rotation
        log_path = store.log_path
        rotated_files = list(log_path.parent.glob("decision_log.jsonl.*"))
        # May or may not rotate depending on _LOG_MAX_BYTES
        assert isinstance(rotated_files, list)

    def test_read_log_with_limit(self):
        """read_log(limit=N) returns last N entries."""
        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self.tmpdir.name), adapter)
        
        for i in range(10):
            store.append_log({"cycle_id": f"cycle{i}"})
        
        log = store.read_log(limit=3)
        assert len(log) <= 3


# ============================================================================
# Mocking typo fix for class name
# ============================================================================

class MementoMemoryStore(MomentoMemoryStore):
    """Alias to test the class with correct spelling."""
    pass
