from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class LocalCacheAdapter:
    """
    Lightweight in-memory cache used as the default fallback when no external
    cache (e.g., Momento) is configured. Thread-safe and file-system agnostic.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._kv: Dict[str, Dict[str, Tuple[Any, Optional[float]]]] = {}
        self._lists: Dict[str, Dict[str, List[Any]]] = {}
        self._events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Key-value operations
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        return True

    def _now(self) -> float:
        return time.time()

    def cache_set(self, cache: str, key: str, value: Any, ttl_seconds: float = 0) -> bool:
        expiry = None if not ttl_seconds or ttl_seconds <= 0 else self._now() + ttl_seconds
        with self._lock:
            self._kv.setdefault(cache, {})[key] = (value, expiry)
        return True

    def cache_get(self, cache: str, key: str) -> Any:
        with self._lock:
            entry = self._kv.get(cache, {}).get(key)
            if not entry:
                return None
            value, expiry = entry
            if expiry and expiry <= self._now():
                # Expired; remove lazily
                self.cache_delete(cache, key)
                return None
            return value

    def cache_delete(self, cache: str, key: str) -> None:
        with self._lock:
            cache_map = self._kv.get(cache)
            if not cache_map:
                return
            cache_map.pop(key, None)
            if not cache_map:
                self._kv.pop(cache, None)

    def evict_expired(self) -> int:
        removed = 0
        now = self._now()
        with self._lock:
            for cache, items in list(self._kv.items()):
                for key, (_, expiry) in list(items.items()):
                    if expiry and expiry <= now:
                        del items[key]
                        removed += 1
                if not items:
                    self._kv.pop(cache, None)
        return removed

    # ------------------------------------------------------------------ #
    # List operations
    # ------------------------------------------------------------------ #
    def list_push(self, cache: str, key: str, value: Any, max_size: Optional[int] = None) -> bool:
        with self._lock:
            lists = self._lists.setdefault(cache, {})
            lst = lists.setdefault(key, [])
            lst.append(value)
            if max_size and max_size > 0 and len(lst) > max_size:
                lst[:] = lst[-max_size:]
        return True

    def list_range(self, cache: str, key: str, start: int = 0, end: Optional[int] = None) -> List[Any]:
        with self._lock:
            lst = list(self._lists.get(cache, {}).get(key, []))
        if end is None or end == -1:
            return lst[start:]
        return lst[start:end + 1]

    # ------------------------------------------------------------------ #
    # Pub/Sub
    # ------------------------------------------------------------------ #
    def publish(self, topic: str, message: str) -> bool:
        with self._lock:
            self._events.append({"topic": topic, "message": message, "timestamp": self._now()})
        return True

    def read_events(self, topic: str, since_ts: float = 0.0) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                dict(evt)
                for evt in self._events
                if evt["topic"] == topic and evt["timestamp"] > since_ts
            ]

    # ------------------------------------------------------------------ #
    # Stats and maintenance
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        now = self._now()
        with self._lock:
            kv_live = kv_expired = 0
            for items in self._kv.values():
                for _, expiry in items.values():
                    if expiry and expiry <= now:
                        kv_expired += 1
                    else:
                        kv_live += 1

            list_keys = sum(len(cache_lists) for cache_lists in self._lists.values())
            list_items = sum(len(lst) for cache_lists in self._lists.values() for lst in cache_lists.values())

        return {
            "kv_live": kv_live,
            "kv_expired_pending_eviction": kv_expired,
            "list_keys": list_keys,
            "list_items": list_items,
        }

    def flush(self) -> None:
        with self._lock:
            self._kv.clear()
            self._lists.clear()
            self._events.clear()
