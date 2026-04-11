"""Tests for error resolution cache."""

import os
import pytest
import tempfile

from aura.error_resolution.cache import LRUCache, SQLiteCache, FourLayerCache
from aura.error_resolution.types import CacheKey, ResolutionConfidence, ResolutionResult


class TestLRUCache:
    """Tests for L1 in-memory LRU cache."""

    def test_get_returns_none_for_missing_key(self):
        """Getting a missing key should return None."""
        cache = LRUCache(maxsize=10)
        result = cache.get("missing")
        assert result is None

    def test_set_and_get(self):
        """Should be able to set and get a value."""
        cache = LRUCache(maxsize=10)
        result = ResolutionResult(
            original_error="test",
            explanation="test explanation",
            suggested_fix="test fix",
            confidence=ResolutionConfidence.HIGH,
            auto_applied=False,
            cache_hit=False,
            provider="test",
            execution_time_ms=100,
        )

        cache.set("key", result)
        retrieved = cache.get("key")

        assert retrieved is not None
        assert retrieved.original_error == "test"
        assert retrieved.suggested_fix == "test fix"

    def test_cache_respects_maxsize(self):
        """Cache should evict oldest items when at capacity."""
        cache = LRUCache(maxsize=2)

        result1 = self._make_result("error1")
        result2 = self._make_result("error2")
        result3 = self._make_result("error3")

        cache.set("key1", result1)
        cache.set("key2", result2)
        cache.set("key3", result3)  # Should evict key1

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert len(cache) == 2

    def test_access_moves_to_front(self):
        """Accessing an item should make it most recently used."""
        cache = LRUCache(maxsize=2)

        result1 = self._make_result("error1")
        result2 = self._make_result("error2")
        result3 = self._make_result("error3")

        cache.set("key1", result1)
        cache.set("key2", result2)
        cache.get("key1")  # Access key1, makes it MRU
        cache.set("key3", result3)  # Should evict key2

        assert cache.get("key1") is not None  # Still there
        assert cache.get("key2") is None  # Evicted

    def test_clear_removes_all(self):
        """Clear should remove all items."""
        cache = LRUCache(maxsize=10)
        cache.set("key1", self._make_result("error1"))
        cache.set("key2", self._make_result("error2"))

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache) == 0

    def _make_result(self, error: str) -> ResolutionResult:
        return ResolutionResult(
            original_error=error,
            explanation="test",
            suggested_fix="test",
            confidence=ResolutionConfidence.HIGH,
            auto_applied=False,
            cache_hit=False,
            provider="test",
            execution_time_ms=100,
        )


class TestSQLiteCache:
    """Tests for L2 SQLite disk cache."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        os.unlink(path)

    def test_set_and_get(self, temp_db):
        """Should persist and retrieve values."""
        cache = SQLiteCache(path=temp_db, ttl_seconds=3600)
        result = ResolutionResult(
            original_error="test error",
            explanation="explanation",
            suggested_fix="fix command",
            confidence=ResolutionConfidence.MEDIUM,
            auto_applied=False,
            cache_hit=False,
            provider="openai",
            execution_time_ms=500,
        )

        cache.set("mykey", result)
        retrieved = cache.get("mykey")

        assert retrieved is not None
        assert retrieved.original_error == "test error"
        assert retrieved.confidence == ResolutionConfidence.MEDIUM
        assert retrieved.provider == "openai"

    def test_expired_entries_return_none(self, temp_db):
        """Expired entries should be removed and return None."""
        cache = SQLiteCache(path=temp_db, ttl_seconds=0)  # Immediate expiry
        result = self._make_result("test")

        cache.set("key", result)
        retrieved = cache.get("key")

        assert retrieved is None

    def test_clear_removes_all(self, temp_db):
        """Clear should delete all entries."""
        cache = SQLiteCache(path=temp_db, ttl_seconds=3600)
        cache.set("key1", self._make_result("error1"))
        cache.set("key2", self._make_result("error2"))

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cleanup_expired(self, temp_db):
        """Should clean up expired entries."""
        cache = SQLiteCache(path=temp_db, ttl_seconds=1)

        # Add entry
        cache.set("key", self._make_result("test"))

        # Manually expire the entry by updating its timestamp
        import time
        import sqlite3

        with sqlite3.connect(temp_db) as conn:
            # Set created_at to 1 hour ago
            conn.execute("UPDATE cache SET created_at = ? WHERE key = ?", (time.time() - 3600, "key"))
            conn.commit()

        # Cleanup should remove it
        cache.cleanup_expired()

        assert cache.get("key") is None

    def _make_result(self, error: str) -> ResolutionResult:
        return ResolutionResult(
            original_error=error,
            explanation="test",
            suggested_fix="test",
            confidence=ResolutionConfidence.HIGH,
            auto_applied=False,
            cache_hit=False,
            provider="test",
            execution_time_ms=100,
        )


class TestFourLayerCache:
    """Tests for the combined 4-layer cache."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        os.unlink(path)

    def test_l1_cache_hit(self, temp_db):
        """L1 cache should return result without hitting L2."""
        cache = FourLayerCache(l1_size=10, l2_path=temp_db)
        key = CacheKey("TypeError", "invalid type", "hash123")
        result = self._make_result("test")

        cache.set(key, result)

        # Should come from L1
        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.original_error == "test"

    def test_l2_promotes_to_l1(self, temp_db):
        """L2 hit should promote to L1."""
        cache = FourLayerCache(l1_size=10, l2_path=temp_db)
        key = CacheKey("ValueError", "bad value", "hash456")
        result = self._make_result("promotion test")

        # Store directly in L2
        cache.l2_disk.set(str(key), result)

        # First get should promote to L1
        retrieved = cache.get(key)
        assert retrieved is not None

        # Clear L2, should still be in L1
        cache.l2_disk.clear()
        retrieved2 = cache.get(key)
        assert retrieved2 is not None

    def test_make_key_consistency(self):
        """Same error and context should produce same key."""
        cache = FourLayerCache()

        error = ValueError("test error")
        context = {"command": "test_cmd", "cwd": "/home"}

        key1 = cache.make_key(error, context)
        key2 = cache.make_key(error, context)

        assert key1.error_type == key2.error_type
        assert key1.error_message == key2.error_message
        assert key1.command_hash == key2.command_hash

    def test_make_key_different_contexts(self):
        """Different contexts should produce different keys."""
        cache = FourLayerCache()

        error = ValueError("test error")
        context1 = {"command": "cmd1"}
        context2 = {"command": "cmd2"}

        key1 = cache.make_key(error, context1)
        key2 = cache.make_key(error, context2)

        assert key1.command_hash != key2.command_hash

    def test_clear_clears_all_layers(self, temp_db):
        """Clear should empty both L1 and L2."""
        cache = FourLayerCache(l1_size=10, l2_path=temp_db)
        key = CacheKey("Error", "msg", "hash")

        cache.set(key, self._make_result("test"))
        cache.clear()

        assert cache.get(key) is None
        assert len(cache.l1_memory) == 0

    def _make_result(self, error: str) -> ResolutionResult:
        return ResolutionResult(
            original_error=error,
            explanation="test",
            suggested_fix="test",
            confidence=ResolutionConfidence.HIGH,
            auto_applied=False,
            cache_hit=False,
            provider="test",
            execution_time_ms=100,
        )
