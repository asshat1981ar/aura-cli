"""Redis-backed caching layer for AURA CLI.

Provides distributed caching for:
- LLM response caching
- Embedding vector caching
- Session state caching
- Configuration caching
"""

from __future__ import annotations

import hashlib
import json
import pickle
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from core.logging_utils import log_json

T = TypeVar("T")

# Cache key prefixes
CACHE_PREFIX_LLM = "aura:llm:"
CACHE_PREFIX_EMBED = "aura:embed:"
CACHE_PREFIX_SESSION = "aura:session:"
CACHE_PREFIX_CONFIG = "aura:config:"
CACHE_PREFIX_TASK = "aura:task:"
DEFAULT_TTL = 3600  # 1 hour


class CacheError(Exception):
    """Cache operation error."""
    pass


class CacheClient:
    """Redis-backed cache client with fallback to in-memory."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self._redis = None
        self._memory: dict[str, Any] = {}
        self._enabled = True
        
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(
                    redis_url,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30,
                    decode_responses=False,
                )
                self._redis.ping()
                log_json("INFO", "cache_redis_connected", {"url": redis_url.split("@")[-1]})
            except Exception as e:
                log_json("WARN", "cache_redis_fallback", {"error": str(e)})
                self._redis = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._enabled:
            return None
            
        try:
            if self._redis:
                data = self._redis.get(key)
                if data:
                    return pickle.loads(data)
            else:
                entry = self._memory.get(key)
                if entry and entry["expires"] > 0:  # Simple TTL check
                    return entry["value"]
        except Exception as e:
            log_json("WARN", "cache_get_error", {"key": key, "error": str(e)})
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int = DEFAULT_TTL,
    ) -> bool:
        """Set value in cache with TTL."""
        if not self._enabled:
            return False
            
        try:
            if self._redis:
                data = pickle.dumps(value)
                self._redis.setex(key, ttl, data)
            else:
                import time
                self._memory[key] = {
                    "value": value,
                    "expires": time.time() + ttl,
                }
            return True
        except Exception as e:
            log_json("WARN", "cache_set_error", {"key": key, "error": str(e)})
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if self._redis:
                self._redis.delete(key)
            else:
                self._memory.pop(key, None)
            return True
        except Exception as e:
            log_json("WARN", "cache_delete_error", {"key": key, "error": str(e)})
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if self._redis:
                return bool(self._redis.exists(key))
            else:
                return key in self._memory
        except Exception:
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            if self._redis:
                keys = self._redis.scan_iter(match=pattern)
                count = 0
                for key in keys:
                    self._redis.delete(key)
                    count += 1
                return count
            else:
                # Simple prefix match for in-memory
                keys_to_delete = [k for k in self._memory if k.startswith(pattern.rstrip("*"))]
                for k in keys_to_delete:
                    del self._memory[k]
                return len(keys_to_delete)
        except Exception as e:
            log_json("WARN", "cache_clear_error", {"pattern": pattern, "error": str(e)})
            return 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            if self._redis:
                info = self._redis.info()
                return {
                    "backend": "redis",
                    "connected": True,
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "key_count": self._redis.dbsize(),
                    "hit_rate": info.get("keyspace_hits", 0) / max(
                        info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1
                    ),
                }
            else:
                return {
                    "backend": "memory",
                    "connected": True,
                    "key_count": len(self._memory),
                    "hit_rate": None,
                }
        except Exception as e:
            return {"backend": "unknown", "connected": False, "error": str(e)}


# Global cache instance
_cache_instance: Optional[CacheClient] = None


def init_cache(redis_url: Optional[str] = None) -> CacheClient:
    """Initialize global cache instance."""
    global _cache_instance
    _cache_instance = CacheClient(redis_url)
    return _cache_instance


def get_cache() -> CacheClient:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheClient()
    return _cache_instance


def cached(
    prefix: str = "aura",
    ttl: int = DEFAULT_TTL,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key from arguments
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = f"{prefix}:{key_func(*args, **kwargs)}"
            else:
                # Default: hash of function name and arguments
                key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                cache_key = f"{prefix}:{hashlib.sha256(key_data.encode()).hexdigest()[:16]}"
            
            # Try cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                log_json("DEBUG", "cache_hit", {"key": cache_key})
                return cached_value
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            log_json("DEBUG", "cache_miss", {"key": cache_key})
            return result
        
        return wrapper
    return decorator


def cache_llm_response(
    model: str,
    prompt: str,
    response: str,
    ttl: int = 3600,
) -> bool:
    """Cache LLM response."""
    key = f"{CACHE_PREFIX_LLM}{model}:{hashlib.sha256(prompt.encode()).hexdigest()[:32]}"
    return get_cache().set(key, {"response": response, "model": model}, ttl)


def get_cached_llm_response(model: str, prompt: str) -> Optional[str]:
    """Get cached LLM response."""
    key = f"{CACHE_PREFIX_LLM}{model}:{hashlib.sha256(prompt.encode()).hexdigest()[:32]}"
    cached = get_cache().get(key)
    if cached and isinstance(cached, dict):
        return cached.get("response")
    return None


def cache_embedding(text: str, embedding: list[float], ttl: int = 86400) -> bool:
    """Cache text embedding vector."""
    key = f"{CACHE_PREFIX_EMBED}{hashlib.sha256(text.encode()).hexdigest()[:32]}"
    return get_cache().set(key, embedding, ttl)


def get_cached_embedding(text: str) -> Optional[list[float]]:
    """Get cached embedding vector."""
    key = f"{CACHE_PREFIX_EMBED}{hashlib.sha256(text.encode()).hexdigest()[:32]}"
    return get_cache().get(key)


def cache_session(session_id: str, data: dict[str, Any], ttl: int = 86400) -> bool:
    """Cache session data."""
    key = f"{CACHE_PREFIX_SESSION}{session_id}"
    return get_cache().set(key, data, ttl)


def get_cached_session(session_id: str) -> Optional[dict[str, Any]]:
    """Get cached session data."""
    key = f"{CACHE_PREFIX_SESSION}{session_id}"
    return get_cache().get(key)


def invalidate_session(session_id: str) -> bool:
    """Invalidate cached session."""
    key = f"{CACHE_PREFIX_SESSION}{session_id}"
    return get_cache().delete(key)


def clear_all_cache() -> int:
    """Clear all AURA cache entries."""
    return get_cache().clear_pattern("aura:*")
