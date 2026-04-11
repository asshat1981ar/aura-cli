"""Tests for aura/error_resolution/cache.py — LRUCache, SQLiteCache, FourLayerCache."""

import time
import pytest

from aura.error_resolution.cache import LRUCache, SQLiteCache, FourLayerCache
from aura.error_resolution.types import CacheKey, ResolutionConfidence, ResolutionResult


def _result(fix="pip install foo", confidence=ResolutionConfidence.HIGH) -> ResolutionResult:
    return ResolutionResult(
        original_error="ImportError: foo",
        explanation="foo not installed",
        suggested_fix=fix,
        confidence=confidence,
        auto_applied=False,
        cache_hit=False,
        provider="test",
        execution_time_ms=50,
    )


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_get_missing_returns_none(self):
        cache = LRUCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = LRUCache()
        r = _result()
        cache.set("key1", r)
        assert cache.get("key1") is r

    def test_len_increases(self):
        cache = LRUCache()
        cache.set("a", _result())
        cache.set("b", _result())
        assert len(cache) == 2

    def test_evicts_lru_when_full(self):
        cache = LRUCache(maxsize=2)
        cache.set("a", _result("fix_a"))
        cache.set("b", _result("fix_b"))
        cache.set("c", _result("fix_c"))
        # "a" should be evicted (LRU)
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_promotes_to_mru(self):
        cache = LRUCache(maxsize=2)
        cache.set("a", _result())
        cache.set("b", _result())
        cache.get("a")  # access "a" to promote it
        cache.set("c", _result())  # should evict "b" now
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_overwrite_existing_key(self):
        cache = LRUCache()
        r1 = _result("fix1")
        r2 = _result("fix2")
        cache.set("k", r1)
        cache.set("k", r2)
        assert cache.get("k") is r2
        assert len(cache) == 1

    def test_clear_empties_cache(self):
        cache = LRUCache()
        cache.set("a", _result())
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None


# ---------------------------------------------------------------------------
# SQLiteCache
# ---------------------------------------------------------------------------

class TestSQLiteCache:
    def test_get_missing_returns_none(self):
        cache = SQLiteCache(path=":memory:")
        assert cache.get("missing") is None

    def test_set_and_get_roundtrip(self):
        cache = SQLiteCache(path=":memory:")
        r = _result()
        cache.set("k1", r)
        result = cache.get("k1")
        assert result is not None
        assert result.suggested_fix == r.suggested_fix
        assert result.confidence == r.confidence

    def test_clear_removes_all(self):
        cache = SQLiteCache(path=":memory:")
        cache.set("a", _result())
        cache.set("b", _result())
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_expired_entry_returns_none(self):
        # TTL of 0 means always expired
        cache = SQLiteCache(path=":memory:", ttl_seconds=0)
        cache.set("k", _result())
        assert cache.get("k") is None

    def test_cleanup_expired_removes_stale(self):
        cache = SQLiteCache(path=":memory:", ttl_seconds=1)
        cache.set("stale", _result())
        # Manually set created_at to far in the past
        cache._conn.execute(
            "UPDATE cache SET created_at = ? WHERE key = ?", (time.time() - 9999, "stale")
        )
        cache._conn.commit()
        cache.cleanup_expired()
        assert cache.get("stale") is None

    def test_overwrite_existing_key(self):
        cache = SQLiteCache(path=":memory:")
        cache.set("k", _result("old"))
        cache.set("k", _result("new"))
        result = cache.get("k")
        assert result.suggested_fix == "new"


# ---------------------------------------------------------------------------
# FourLayerCache
# ---------------------------------------------------------------------------

class TestFourLayerCache:
    def test_make_key_returns_cache_key(self):
        cache = FourLayerCache()
        key = cache.make_key(ValueError("bad input"))
        assert isinstance(key, CacheKey)
        assert key.error_type == "ValueError"
        assert "bad input" in key.error_message

    def test_make_key_deterministic(self):
        cache = FourLayerCache()
        err = RuntimeError("oops")
        k1 = cache.make_key(err, {"cmd": "pytest"})
        k2 = cache.make_key(err, {"cmd": "pytest"})
        assert k1 == k2

    def test_make_key_context_affects_hash(self):
        cache = FourLayerCache()
        err = RuntimeError("oops")
        k1 = cache.make_key(err, {"cmd": "pytest"})
        k2 = cache.make_key(err, {"cmd": "mypy"})
        assert k1 != k2

    def test_get_miss_returns_none(self):
        cache = FourLayerCache()
        key = CacheKey("ValueError", "msg", "hash")
        assert cache.get(key) is None

    def test_set_and_get(self):
        cache = FourLayerCache()
        key = CacheKey("ValueError", "msg", "hash")
        r = _result()
        cache.set(key, r)
        result = cache.get(key)
        assert result is not None
        assert result.suggested_fix == r.suggested_fix

    def test_l1_hit_after_set(self):
        cache = FourLayerCache()
        key = CacheKey("T", "m", "h")
        r = _result()
        cache.set(key, r)
        # L1 should have it
        assert cache.l1_memory.get(str(key)) is not None

    def test_l2_hit_promotes_to_l1(self):
        cache = FourLayerCache()
        key = CacheKey("T", "m", "h")
        r = _result()
        # Set only in L2, bypass L1
        cache.l2_disk.set(str(key), r)
        assert cache.l1_memory.get(str(key)) is None
        # Get should promote
        result = cache.get(key)
        assert result is not None
        assert cache.l1_memory.get(str(key)) is not None

    def test_clear_wipes_both_layers(self):
        cache = FourLayerCache()
        key = CacheKey("T", "m", "h")
        cache.set(key, _result())
        cache.clear()
        assert cache.get(key) is None
