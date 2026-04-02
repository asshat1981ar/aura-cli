"""Tests for core/cache.py Redis caching layer."""

import pytest
import time
from unittest.mock import Mock, patch

from core.cache import (
    CacheClient,
    cached,
    cache_llm_response,
    get_cached_llm_response,
    cache_embedding,
    get_cached_embedding,
    cache_session,
    get_cached_session,
    invalidate_session,
    clear_all_cache,
    init_cache,
    get_cache,
    DEFAULT_TTL,
)


class TestCacheClient:
    """Test CacheClient functionality."""
    
    def test_init_without_redis(self):
        """Test CacheClient initializes with in-memory fallback."""
        cache = CacheClient(redis_url=None)
        assert cache._redis is None
        assert cache._enabled is True
    
    def test_get_set_memory_backend(self):
        """Test get/set with in-memory backend."""
        cache = CacheClient(redis_url=None)
        
        # Set value
        assert cache.set("test_key", "test_value", ttl=60) is True
        
        # Get value
        assert cache.get("test_key") == "test_value"
    
    def test_get_missing_key(self):
        """Test get returns None for missing key."""
        cache = CacheClient(redis_url=None)
        assert cache.get("nonexistent_key") is None
    
    def test_delete(self):
        """Test delete operation."""
        cache = CacheClient(redis_url=None)
        cache.set("delete_me", "value")
        assert cache.get("delete_me") == "value"
        
        cache.delete("delete_me")
        assert cache.get("delete_me") is None
    
    def test_exists(self):
        """Test exists check."""
        cache = CacheClient(redis_url=None)
        assert cache.exists("exists_key") is False
        
        cache.set("exists_key", "value")
        assert cache.exists("exists_key") is True
    
    def test_clear_pattern(self):
        """Test clearing keys by pattern."""
        cache = CacheClient(redis_url=None)
        cache.set("aura:test:1", "value1")
        cache.set("aura:test:2", "value2")
        cache.set("other:key", "value3")
        
        count = cache.clear_pattern("aura:test:*")
        assert count == 2
        assert cache.get("aura:test:1") is None
        assert cache.get("aura:test:2") is None
        assert cache.get("other:key") == "value3"
    
    def test_get_stats_memory(self):
        """Test get_stats with memory backend."""
        cache = CacheClient(redis_url=None)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        assert stats["backend"] == "memory"
        assert stats["key_count"] == 2
        assert stats["connected"] is True
    
    def test_disabled_cache(self):
        """Test cache operations when disabled."""
        cache = CacheClient(redis_url=None)
        cache._enabled = False
        
        assert cache.set("key", "value") is False
        assert cache.get("key") is None


class TestCachedDecorator:
    """Test @cached decorator."""
    
    def test_cached_function(self):
        """Test that cached decorator caches function results."""
        call_count = 0
        
        @cached(prefix="test", ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
    
    def test_cached_with_different_args(self):
        """Test cache keys are unique per arguments."""
        call_count = 0
        
        @cached(prefix="test", ttl=60)
        def func(x, y=0):
            nonlocal call_count
            call_count += 1
            return x + y
        
        func(1)
        func(1, y=0)
        func(2)
        
        # Should be 2 unique calls (1 and 2)
        assert call_count == 2


class TestLLMCache:
    """Test LLM response caching."""
    
    def test_cache_llm_response(self):
        """Test caching and retrieving LLM responses."""
        cache = CacheClient(redis_url=None)
        with patch('core.cache.get_cache', return_value=cache):
            # Cache response
            cache_llm_response("gpt-4", "prompt text", "response text")
            
            # Retrieve response
            cached = get_cached_llm_response("gpt-4", "prompt text")
            assert cached == "response text"
    
    def test_get_missing_llm_response(self):
        """Test getting non-existent LLM response."""
        cache = CacheClient(redis_url=None)
        with patch('core.cache.get_cache', return_value=cache):
            result = get_cached_llm_response("gpt-4", "unknown prompt")
            assert result is None


class TestEmbeddingCache:
    """Test embedding vector caching."""
    
    def test_cache_embedding(self):
        """Test caching and retrieving embeddings."""
        cache = CacheClient(redis_url=None)
        with patch('core.cache.get_cache', return_value=cache):
            embedding = [0.1, 0.2, 0.3, 0.4]
            cache_embedding("test text", embedding)
            
            cached = get_cached_embedding("test text")
            assert cached == embedding


class TestSessionCache:
    """Test session caching."""
    
    def test_cache_session(self):
        """Test caching and retrieving session data."""
        cache = CacheClient(redis_url=None)
        with patch('core.cache.get_cache', return_value=cache):
            session_data = {"user_id": "123", "preferences": {"theme": "dark"}}
            cache_session("session_123", session_data)
            
            cached = get_cached_session("session_123")
            assert cached == session_data
    
    def test_invalidate_session(self):
        """Test session invalidation."""
        cache = CacheClient(redis_url=None)
        with patch('core.cache.get_cache', return_value=cache):
            cache_session("session_456", {"data": "test"})
            assert get_cached_session("session_456") is not None
            
            invalidate_session("session_456")
            assert get_cached_session("session_456") is None


class TestGlobalCache:
    """Test global cache instance functions."""
    
    def test_init_cache(self):
        """Test initializing global cache."""
        cache = init_cache(redis_url=None)
        assert cache is not None
        assert get_cache() is cache
    
    def test_get_cache_creates_default(self):
        """Test get_cache creates default if not initialized."""
        import core.cache
        core.cache._cache_instance = None
        cache = get_cache()
        assert cache is not None


class TestClearAllCache:
    """Test clearing all cache."""
    
    def test_clear_all(self):
        """Test clearing all AURA cache entries."""
        cache = CacheClient(redis_url=None)
        with patch('core.cache.get_cache', return_value=cache):
            cache.set("aura:key1", "value1")
            cache.set("aura:key2", "value2")
            cache.set("other:key", "value3")
            
            count = clear_all_cache()
            assert count == 2
            assert cache.get("aura:key1") is None
            assert cache.get("other:key") == "value3"
