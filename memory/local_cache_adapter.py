"""
LocalCacheAdapter — zero-dependency, always-on drop-in for MomentoAdapter.

Provides the exact same interface as :class:`memory.momento_adapter.MomentoAdapter`
but runs entirely in-process:

- **Key-value cache**: in-process ``dict`` with per-key TTL tracking.
  Entries are lazily expired on ``cache_get`` — no background thread needed.
- **List operations**: in-process ``dict`` of lists with configurable max-size
  trimming (mirrors Momento's ``list_push_back`` + ``list_retain``).
- **Pub/Sub topics**: SQLite ``topic_events`` table (append-only log).
  ``publish()`` inserts a row; subscribers can poll or tail the table.

``is_available()`` always returns ``True`` — no API key or external service
required.  This makes it a perfect L1 cache for single-process AURA runs on
devices without network access (e.g. Android/Termux).

Usage::

    from memory.local_cache_adapter import LocalCacheAdapter
    adapter = LocalCacheAdapter()
    adapter.cache_set("aura-working-memory", "skill_weights:all", data, ttl_seconds=3600)
    value = adapter.cache_get("aura-working-memory", "skill_weights:all")
    adapter.publish("aura.cycle_complete", json.dumps(event))

Thread safety:
    All in-process operations are protected by a single ``threading.RLock``.
    SQLite writes use ``check_same_thread=False`` with a per-call connection.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

from core.logging_utils import log_json

# Default TTL for cache_set when caller passes 0 or omits ttl_seconds
DEFAULT_TTL_SECONDS = 3600

# Maximum list size when caller omits max_size
LIST_MAX_SIZE = 200

# SQLite file for topic event log (relative to project root)
_DEFAULT_DB_PATH = "memory/local_cache.db"


class LocalCacheAdapter:
    """In-process cache adapter with the same API as :class:`MomentoAdapter`.

    Args:
        db_path: Path to the SQLite file used for the topic event log.
            Defaults to ``memory/local_cache.db``.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self._lock = threading.RLock()
        # {namespaced_key: (value: str, expire_at: float|None)}
        self._kv: dict[str, tuple[str, Optional[float]]] = {}
        # {namespaced_key: list[str]}
        self._lists: dict[str, list[str]] = {}
        self._db_path = db_path
        self._init_db()
        log_json("INFO", "local_cache_adapter_initialized",
                 details={"db_path": db_path})

    # ── Availability ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Always True — no external service required."""
        return True

    # ── Cache operations ──────────────────────────────────────────────────────

    def cache_get(self, cache: str, key: str) -> Optional[str]:
        """Return the cached value or ``None`` on miss / expired entry."""
        k = _ns(cache, key)
        with self._lock:
            entry = self._kv.get(k)
            if entry is None:
                return None
            value, exp = entry
            if exp is not None and time.time() > exp:
                del self._kv[k]
                return None
            return value

    def cache_set(
        self,
        cache: str,
        key: str,
        value: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> bool:
        """Store *value* under *key* in *cache* with optional TTL."""
        k = _ns(cache, key)
        exp = time.time() + ttl_seconds if ttl_seconds > 0 else None
        with self._lock:
            self._kv[k] = (value, exp)
        return True

    def cache_delete(self, cache: str, key: str) -> bool:
        """Remove *key* from *cache*.  Returns ``True`` always."""
        k = _ns(cache, key)
        with self._lock:
            self._kv.pop(k, None)
        return True

    # ── List operations ───────────────────────────────────────────────────────

    def list_push(
        self,
        cache: str,
        key: str,
        value: str,
        ttl_seconds: int = 0,
        max_size: int = LIST_MAX_SIZE,
    ) -> bool:
        """Append *value* to the list at *key*, trimming to *max_size*."""
        k = _ns(cache, key)
        with self._lock:
            lst = self._lists.setdefault(k, [])
            lst.append(value)
            if len(lst) > max_size:
                # Keep the most recent max_size items (trim from front)
                self._lists[k] = lst[-max_size:]
        return True

    def list_range(
        self,
        cache: str,
        key: str,
        start: int = 0,
        end: int = -1,
    ) -> List[str]:
        """Return a slice of the list at *key*.

        *end* ``-1`` means "all remaining elements" (matches Redis/Momento
        convention).  Returned list is a copy — safe to mutate.
        """
        k = _ns(cache, key)
        with self._lock:
            lst = self._lists.get(k, [])
            if end == -1 or end is None:
                return list(lst[start:])
            return list(lst[start : end + 1])

    # ── Pub/Sub topics ────────────────────────────────────────────────────────

    def publish(self, topic: str, message: str) -> bool:
        """Append *message* to the ``topic_events`` log in SQLite."""
        try:
            with sqlite3.connect(self._db_path, check_same_thread=False) as conn:
                conn.execute(
                    "INSERT INTO topic_events (topic, message, ts) VALUES (?, ?, ?)",
                    (topic, message, time.time()),
                )
            return True
        except Exception as exc:
            log_json("WARN", "local_cache_publish_failed",
                     details={"topic": topic, "error": str(exc)})
            return False

    def read_events(
        self,
        topic: str,
        since_ts: float = 0.0,
        limit: int = 100,
    ) -> List[dict]:
        """Poll *topic* events newer than *since_ts*.

        Useful for improvement loops that want to react to cycle completions
        without a blocking subscribe.  Returns a list of
        ``{"id": int, "topic": str, "message": str, "ts": float}`` dicts.
        """
        try:
            with sqlite3.connect(self._db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, topic, message, ts FROM topic_events "
                    "WHERE topic = ? AND ts > ? ORDER BY ts ASC LIMIT ?",
                    (topic, since_ts, limit),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            log_json("WARN", "local_cache_read_events_failed",
                     details={"topic": topic, "error": str(exc)})
            return []

    # ── Compatibility stubs (match MomentoAdapter API exactly) ────────────────

    def ensure_caches(self) -> None:
        """No-op — local adapter has no named caches to create."""

    # ── Introspection / housekeeping ──────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of in-process cache statistics."""
        with self._lock:
            now = time.time()
            live = sum(
                1 for (_, exp) in self._kv.values()
                if exp is None or exp > now
            )
            expired = len(self._kv) - live
            list_items = sum(len(v) for v in self._lists.values())
        return {
            "kv_live": live,
            "kv_expired_pending_eviction": expired,
            "list_keys": len(self._lists),
            "list_items": list_items,
        }

    def evict_expired(self) -> int:
        """Eagerly remove expired KV entries.  Returns count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, (_, exp) in self._kv.items()
                if exp is not None and now > exp
            ]
            for k in expired_keys:
                del self._kv[k]
                removed += 1
        return removed

    def flush(self) -> None:
        """Clear all in-process state (KV + lists).  SQLite event log kept."""
        with self._lock:
            self._kv.clear()
            self._lists.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the SQLite event-log table if it doesn't exist."""
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._db_path, check_same_thread=False) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS topic_events (
                        id      INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic   TEXT    NOT NULL,
                        message TEXT    NOT NULL,
                        ts      REAL    NOT NULL
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_topic_ts "
                    "ON topic_events (topic, ts)"
                )
        except Exception as exc:
            log_json("WARN", "local_cache_db_init_failed", details={"error": str(exc)})


# ── Module-level helper ───────────────────────────────────────────────────────

def _ns(cache: str, key: str) -> str:
    """Namespace a key as ``cache:key`` (matches Momento's logical separation)."""
    return f"{cache}:{key}"
