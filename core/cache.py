"""Agent result caching layer for performance optimization."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from threading import Lock

from core.logging_utils import log_json


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    expires_at: float
    hits: int = 0
    created_at: float = 0.0


class AgentCache:
    """Thread-safe cache for agent results."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        cleanup_interval: int = 300,  # 5 minutes
    ):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            self._maybe_cleanup()
            
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._stats["evictions"] += 1
                self._stats["misses"] += 1
                return None
            
            entry.hits += 1
            self._stats["hits"] += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        with self._lock:
            self._maybe_cleanup()
            
            # Evict oldest entries if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at,
                )
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
            
            now = time.time()
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=now + (ttl or self._default_ttl),
                created_at=now,
            )
            self._stats["size"] = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["size"] = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats["size"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            
            return {
                **self._stats,
                "hit_rate": round(hit_rate, 4),
                "size": len(self._cache),
                "max_size": self._max_size,
            }
    
    def _maybe_cleanup(self) -> None:
        """Clean up expired entries if cleanup interval has passed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        expired = [
            key for key, entry in self._cache.items()
            if now > entry.expires_at
        ]
        for key in expired:
            del self._cache[key]
            self._stats["evictions"] += 1
        
        self._last_cleanup = now
        self._stats["size"] = len(self._cache)


# Global cache instance
_global_cache: Optional[AgentCache] = None


def get_cache() -> AgentCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AgentCache()
    return _global_cache


def cached(
    ttl: Optional[int] = None,
    key_fn: Optional[Callable] = None,
):
    """Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_fn: Function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default: hash of function name and arguments
                key_data = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                }
                cache_key = hashlib.sha256(
                    json.dumps(key_data, sort_keys=True, default=str).encode()
                ).hexdigest()
            
            # Try cache first
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                log_json("DEBUG", "cache_hit", {
                    "key": cache_key[:16] + "...",
                    "func": func.__name__,
                })
                return cached_value
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            log_json("DEBUG", "cache_miss", {
                "key": cache_key[:16] + "...",
                "func": func.__name__,
            })
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(pattern: Optional[str] = None) -> int:
    """Invalidate cache entries.
    
    Args:
        pattern: Optional pattern to match keys (if None, clears all)
        
    Returns:
        Number of entries invalidated
    """
    cache = get_cache()
    
    if pattern is None:
        count = cache.get_stats()["size"]
        cache.clear()
        return count

    import fnmatch
    keys_to_delete = [
        key for key in list(cache._cache.keys())
        if fnmatch.fnmatch(key, pattern)
    ]
    for key in keys_to_delete:
        cache.delete(key)
    return len(keys_to_delete)
