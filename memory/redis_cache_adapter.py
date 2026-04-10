"""
RedisCacheAdapter — optional Redis L0/L1 cache for AURA.

Falls back silently when the ``redis`` package is not installed or the
Redis server is unreachable.  No external dependency is required for the
module to be importable.

Usage::

    from memory.redis_cache_adapter import RedisCacheAdapter
    adapter = RedisCacheAdapter()          # REDIS_URL env var optional
    adapter.set("skill_weights:all", data, ttl_seconds=3600)
    value = adapter.get("skill_weights:all")  # None on miss / unavailable

Environment variables:
    REDIS_URL   Connection URL passed to ``redis.from_url()``.
                Defaults to ``redis://localhost:6379/0``.
"""
from __future__ import annotations

import json
import os
from typing import Any

from core.logging_utils import log_json

_DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class RedisCacheAdapter:
    """Optional Redis L0/L1 cache.  Falls back silently if redis not installed.

    All public methods are safe to call even when Redis is unavailable —
    they return ``None`` / no-op silently so callers never need to guard
    against a missing Redis connection.

    Values are JSON-serialised before storage so any JSON-compatible Python
    object can be round-tripped through the cache.

    Args:
        url: Redis connection URL.  Defaults to ``$REDIS_URL`` env var, or
             ``redis://localhost:6379/0`` if that is not set.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url: str = url or os.getenv("REDIS_URL", _DEFAULT_REDIS_URL)
        self._client: Any = None  # lazily initialised
        self._available: bool | None = None  # None = not yet probed

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` on miss / error.

        Args:
            key: Cache key string.

        Returns:
            The original Python object that was passed to :meth:`set`, or
            ``None`` if the key is absent, expired, or Redis is unavailable.
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            raw = client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            log_json(
                "WARN",
                "redis_get_failed",
                details={"key": key, "error": str(exc)},
            )
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Store *value* under *key* with an optional TTL.

        Args:
            key:         Cache key string.
            value:       JSON-serialisable Python object to cache.
            ttl_seconds: Time-to-live in seconds.  Pass ``0`` for no expiry.
        """
        client = self._get_client()
        if client is None:
            return
        try:
            serialised = json.dumps(value)
            if ttl_seconds > 0:
                client.setex(key, ttl_seconds, serialised)
            else:
                client.set(key, serialised)
        except Exception as exc:
            log_json(
                "WARN",
                "redis_set_failed",
                details={"key": key, "error": str(exc)},
            )

    def delete(self, key: str) -> None:
        """Remove *key* from the cache.  Silent no-op on error / unavailability.

        Args:
            key: Cache key string to remove.
        """
        client = self._get_client()
        if client is None:
            return
        try:
            client.delete(key)
        except Exception as exc:
            log_json(
                "WARN",
                "redis_delete_failed",
                details={"key": key, "error": str(exc)},
            )

    def clear(self) -> None:
        """Flush **all** keys from the current Redis database (FLUSHDB).

        Use with care.  Silent no-op on error / unavailability.
        """
        client = self._get_client()
        if client is None:
            return
        try:
            client.flushdb()
        except Exception as exc:
            log_json(
                "WARN",
                "redis_clear_failed",
                details={"error": str(exc)},
            )

    def is_available(self) -> bool:
        """Return ``True`` if a Redis connection was successfully established."""
        return self._get_client() is not None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Return a live Redis client, or ``None`` if unavailable.

        Uses lazy initialisation — the first call attempts to import
        ``redis`` and connect; subsequent calls reuse the cached result.
        """
        if self._client is not None:
            return self._client
        if self._available is False:
            # Already probed and failed — don't retry on every call.
            return None

        try:
            import redis  # noqa: PLC0415
        except ImportError:
            log_json(
                "WARN",
                "redis_not_installed",
                details={"hint": "pip install redis"},
            )
            self._available = False
            return None

        try:
            client = redis.from_url(self._url, socket_connect_timeout=2)
            client.ping()  # verify connectivity
            self._client = client
            self._available = True
            log_json(
                "INFO",
                "redis_cache_adapter_initialized",
                details={"url": self._url},
            )
            return self._client
        except Exception as exc:
            log_json(
                "WARN",
                "redis_connection_failed",
                details={"url": self._url, "error": str(exc)},
            )
            self._available = False
            return None
