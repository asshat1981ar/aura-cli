"""Multi-layer cache for error resolution results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from typing import Optional

from aura.error_resolution.types import CacheKey, ResolutionResult


class LRUCache:
    """L1 in-memory LRU cache."""

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, ResolutionResult] = OrderedDict()

    def get(self, key: str) -> Optional[ResolutionResult]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: ResolutionResult) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class SQLiteCache:
    """L2 SQLite disk cache with TTL support."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS cache (
        key      TEXT PRIMARY KEY,
        value    TEXT NOT NULL,
        created_at REAL NOT NULL
    )
    """

    def __init__(self, path: str = ":memory:", ttl_seconds: int = 86400) -> None:
        self._path = path
        self._ttl = ttl_seconds
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(self._DDL)
        self._conn.commit()

    def get(self, key: str) -> Optional[ResolutionResult]:
        row = self._conn.execute(
            "SELECT value, created_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value_json, created_at = row
        if self._ttl == 0 or (time.time() - created_at) > self._ttl:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return ResolutionResult.from_dict(json.loads(value_json))

    def set(self, key: str, value: ResolutionResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(value.to_dict()), time.time()),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def cleanup_expired(self) -> None:
        cutoff = time.time() - self._ttl
        self._conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
        self._conn.commit()


class FourLayerCache:
    """Combined L1 (memory LRU) + L2 (SQLite) cache with automatic promotion."""

    def __init__(
        self,
        l1_size: int = 128,
        l2_path: str = ":memory:",
        l2_ttl: int = 86400,
    ) -> None:
        self.l1_memory = LRUCache(maxsize=l1_size)
        self.l2_disk = SQLiteCache(path=l2_path, ttl_seconds=l2_ttl)

    def make_key(self, error: Exception, context: Optional[dict] = None) -> CacheKey:
        """Build a deterministic CacheKey from an exception and optional context."""
        error_type = type(error).__name__
        error_message = str(error)
        context_str = json.dumps(context or {}, sort_keys=True)
        command_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
        return CacheKey(
            error_type=error_type,
            error_message=error_message,
            command_hash=command_hash,
        )

    def get(self, key: CacheKey) -> Optional[ResolutionResult]:
        str_key = str(key)

        # L1 hit
        result = self.l1_memory.get(str_key)
        if result is not None:
            return result

        # L2 hit — promote to L1
        result = self.l2_disk.get(str_key)
        if result is not None:
            self.l1_memory.set(str_key, result)
            return result

        return None

    def set(self, key: CacheKey, value: ResolutionResult) -> None:
        str_key = str(key)
        self.l1_memory.set(str_key, value)
        self.l2_disk.set(str_key, value)

    def clear(self) -> None:
        self.l1_memory.clear()
        self.l2_disk.clear()
