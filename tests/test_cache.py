"""Tests for core/cache.py agent caching layer."""

import time
from unittest.mock import patch

import pytest

from core.cache import AgentCache, CacheEntry, get_cache, cached, invalidate_cache


class TestAgentCache:
    """Test AgentCache functionality."""

    def test_get_set(self):
        cache = AgentCache()
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_missing_key(self):
        cache = AgentCache()
        assert cache.get("nonexistent") is None

    def test_set_with_custom_ttl(self):
        cache = AgentCache()
        cache.set("k1", "v1", ttl=3600)
        assert cache.get("k1") == "v1"

    def test_ttl_expiration(self):
        cache = AgentCache(cleanup_interval=0)
        cache.set("k1", "v1", ttl=1)
        # Manually expire the entry
        cache._cache["k1"].expires_at = time.time() - 1
        assert cache.get("k1") is None

    def test_delete_existing_key(self):
        cache = AgentCache()
        cache.set("k1", "v1")
        assert cache.delete("k1") is True
        assert cache.get("k1") is None

    def test_delete_missing_key(self):
        cache = AgentCache()
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        cache = AgentCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_eviction_at_max_size(self):
        cache = AgentCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        # Oldest entry should have been evicted
        assert cache.get("k3") == "v3"
        assert cache.get_stats()["size"] == 2

    def test_get_stats(self):
        cache = AgentCache()
        cache.set("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert 0 <= stats["hit_rate"] <= 1

    def test_stats_hit_rate_zero_when_empty(self):
        cache = AgentCache()
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0


class TestGetCache:
    """Test global cache instance."""

    def test_get_cache_returns_instance(self):
        cache = get_cache()
        assert isinstance(cache, AgentCache)

    def test_get_cache_returns_same_instance(self):
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2


class TestCachedDecorator:
    """Test @cached decorator."""

    def test_caches_return_value(self):
        call_count = 0

        @cached(ttl=60)
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count == 1

    def test_different_args_different_keys(self):
        call_count = 0

        @cached(ttl=60)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        func(2)
        assert call_count == 2

    def test_custom_key_fn(self):
        call_count = 0

        @cached(ttl=60, key_fn=lambda x, **kw: f"fixed_{x}")
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        func(1)
        assert call_count == 1


class TestInvalidateCache:
    """Test cache invalidation."""

    def test_invalidate_all(self):
        cache = get_cache()
        cache.set("test_inv_1", "v1")
        count = invalidate_cache(pattern=None)
        assert count >= 1

    def test_invalidate_with_pattern_returns_zero(self):
        # Pattern matching not implemented — returns 0
        count = invalidate_cache(pattern="some:*")
        assert count == 0
