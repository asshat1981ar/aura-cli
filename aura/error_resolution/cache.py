"""4-Layer Cache for Error Resolution.

L1: In-memory LRU cache (fastest, ephemeral)
L2: Disk cache via SQLite (persistent, local)
L3: Known fixes registry (curated solutions - separate module)
L4: AI provider (slowest, most capable - separate module)
"""

import hashlib
import json
import os
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .types import CacheKey, ResolutionResult


class LRUCache:
    """Simple in-memory LRU cache (L1)."""
    
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, ResolutionResult] = OrderedDict()
    
    def get(self, key: str) -> Optional[ResolutionResult]:
        """Get item from cache, moving to end if found (most recently used)."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: ResolutionResult):
        """Add item to cache, evicting oldest if at capacity."""
        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
        self._cache[key] = value
        
        # Evict oldest if over capacity
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
    
    def clear(self):
        """Clear all items from cache."""
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


class SQLiteCache:
    """Persistent disk cache using SQLite (L2)."""
    
    def __init__(self, path: str = "~/.aura/error_cache.db", ttl_seconds: int = 604800):
        """
        Initialize SQLite cache.
        
        Args:
            path: Path to SQLite database
            ttl_seconds: Time-to-live for cache entries (default 7 days)
        """
        self.path = Path(path).expanduser()
        self.ttl_seconds = ttl_seconds
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
            """)
            conn.commit()
    
    def get(self, key: str) -> Optional[ResolutionResult]:
        """Get item from cache if not expired."""
        with sqlite3.connect(str(self.path)) as conn:
            cursor = conn.execute(
                "SELECT value, created_at FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            value_json, created_at = row
            
            # Check TTL
            if time.time() - created_at > self.ttl_seconds:
                # Expired, delete it
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            
            # Parse and return
            return self._deserialize(value_json)
    
    def set(self, key: str, value: ResolutionResult, ttl_seconds: int | None = None):
        """Store item in cache."""
        ttl = ttl_seconds or self.ttl_seconds
        value_json = self._serialize(value)
        
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, created_at)
                VALUES (?, ?, ?)
                """,
                (key, value_json, time.time())
            )
            conn.commit()
    
    def clear(self):
        """Clear all items from cache."""
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
    
    def cleanup_expired(self):
        """Remove expired entries."""
        cutoff = time.time() - self.ttl_seconds
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            conn.commit()
    
    def _serialize(self, result: ResolutionResult) -> str:
        """Serialize ResolutionResult to JSON."""
        return json.dumps({
            "original_error": result.original_error,
            "explanation": result.explanation,
            "suggested_fix": result.suggested_fix,
            "confidence": result.confidence.value,
            "auto_applied": result.auto_applied,
            "cache_hit": result.cache_hit,
            "provider": result.provider,
            "execution_time_ms": result.execution_time_ms,
        })
    
    def _deserialize(self, json_str: str) -> ResolutionResult:
        """Deserialize JSON to ResolutionResult."""
        from .types import ResolutionConfidence
        data = json.loads(json_str)
        return ResolutionResult(
            original_error=data["original_error"],
            explanation=data["explanation"],
            suggested_fix=data["suggested_fix"],
            confidence=ResolutionConfidence(data["confidence"]),
            auto_applied=data["auto_applied"],
            cache_hit=data["cache_hit"],
            provider=data["provider"],
            execution_time_ms=data["execution_time_ms"],
        )


class FourLayerCache:
    """4-Layer cache for error resolution.
    
    L1: In-memory LRU (fastest)
    L2: SQLite disk cache (persistent)
    L3: Known fixes registry (curated) - in known_fixes.py
    L4: AI provider - in providers.py
    """
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_path: str = "~/.aura/error_cache.db",
        l2_ttl: int = 604800,  # 7 days
    ):
        self.l1_memory = LRUCache(maxsize=l1_size)
        self.l2_disk = SQLiteCache(path=l2_path, ttl_seconds=l2_ttl)
    
    def get(self, key: CacheKey) -> Optional[ResolutionResult]:
        """
        Get from cache, trying L1 then L2.
        
        Returns:
            Cached ResolutionResult or None
        """
        key_str = str(key)
        
        # Try L1 (memory)
        result = self.l1_memory.get(key_str)
        if result is not None:
            return result
        
        # Try L2 (disk)
        result = self.l2_disk.get(key_str)
        if result is not None:
            # Promote to L1
            self.l1_memory.set(key_str, result)
            return result
        
        return None
    
    def set(self, key: CacheKey, value: ResolutionResult, l2_ttl: int | None = None):
        """
        Store in both L1 and L2 caches.
        
        Args:
            key: Cache key
            value: ResolutionResult to cache
            l2_ttl: Optional TTL override for L2 cache
        """
        key_str = str(key)
        self.l1_memory.set(key_str, value)
        self.l2_disk.set(key_str, value, ttl_seconds=l2_ttl)
    
    def clear(self):
        """Clear all cache layers."""
        self.l1_memory.clear()
        self.l2_disk.clear()
    
    def make_key(self, error: Exception, context: dict | None = None) -> CacheKey:
        """Create a cache key from error and context."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Create hash of context for cache differentiation
        context_data = json.dumps(context or {}, sort_keys=True)
        command_hash = hashlib.md5(context_data.encode()).hexdigest()[:8]
        
        return CacheKey(
            error_type=error_type,
            error_message=error_message,
            command_hash=command_hash,
        )
