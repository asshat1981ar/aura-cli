"""
Momento Adapter — unified client for Momento Cache + Topics.

Provides a single dependency-injection point for all Momento operations.
Falls back to no-op mode when the ``MOMENTO_API_KEY`` environment variable
is not set, so AURA works identically with or without a cloud account.

Usage::

    from memory.momento_adapter import MomentoAdapter
    adapter = MomentoAdapter()

    if adapter.is_available():
        adapter.cache_set("aura-working-memory", "skill_weights:all", json.dumps(weights))
        value = adapter.cache_get("aura-working-memory", "skill_weights:all")
        adapter.publish("aura.cycle_complete", json.dumps(event))
    else:
        # falls back to local storage automatically — no code change needed

Caches used by AURA:
    aura-working-memory   response cache, skill weights, planning hints
    aura-episodic-memory  cycle summaries, reflection reports, weaknesses

Topics used by AURA:
    aura.cycle_complete   published after every run_cycle()
    aura.weakness_found   published when a new weakness is recorded
    aura.goal_queued      published when propagation engine queues a goal
"""
from __future__ import annotations

import os
from datetime import timedelta
from typing import List, Optional

from core.logging_utils import log_json

# Cache names used across the system
WORKING_MEMORY_CACHE = "aura-working-memory"
EPISODIC_MEMORY_CACHE = "aura-episodic-memory"

# Topic names
TOPIC_CYCLE_COMPLETE = "aura.cycle_complete"
TOPIC_WEAKNESS_FOUND = "aura.weakness_found"
TOPIC_GOAL_QUEUED    = "aura.goal_queued"

# How many items to keep in each Momento list (rolling window)
LIST_MAX_SIZE = 200

# Default TTL for ephemeral working-memory keys
DEFAULT_TTL_SECONDS = int(os.getenv("MOMENTO_CACHE_TTL_SECONDS", "3600"))


class MomentoAdapter:
    """Lazy-init wrapper around Momento CacheClient + TopicsClient.

    All methods are safe to call even when Momento is unavailable — they
    return ``None`` / ``[]`` / ``False`` silently so callers never need to
    guard against a missing Momento connection.
    """

    def __init__(self):
        self._api_key: Optional[str] = os.getenv("MOMENTO_API_KEY")
        self._cache_client = None
        self._topics_client = None
        self._initialized = False
        self._available = False

    # ── Availability ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Momento is configured and the client initialised."""
        if not self._api_key:
            return False
        if not self._initialized:
            self._lazy_init()
        return self._available

    # ── Cache operations ──────────────────────────────────────────────────────

    def cache_get(self, cache: str, key: str) -> Optional[str]:
        """Return the cached string value, or None on miss/error."""
        if not self.is_available():
            return None
        try:
            from momento.responses import CacheGet
            resp = self._cache_client.get(cache, key)
            if isinstance(resp, CacheGet.Hit):
                return resp.value_string
            return None
        except Exception as exc:
            log_json("WARN", "momento_cache_get_failed",
                     details={"cache": cache, "key": key, "error": str(exc)})
            return None

    def cache_set(
        self,
        cache: str,
        key: str,
        value: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> bool:
        """Store a string value with TTL.  Returns True on success."""
        if not self.is_available():
            return False
        try:
            from momento.responses import CacheSet
            ttl = timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
            resp = self._cache_client.set(cache, key, value, ttl)
            return isinstance(resp, CacheSet.Success)
        except Exception as exc:
            log_json("WARN", "momento_cache_set_failed",
                     details={"cache": cache, "key": key, "error": str(exc)})
            return False

    def cache_delete(self, cache: str, key: str) -> bool:
        """Delete a key.  Returns True on success."""
        if not self.is_available():
            return False
        try:
            self._cache_client.delete(cache, key)
            return True
        except Exception as exc:
            log_json("WARN", "momento_cache_delete_failed",
                     details={"cache": cache, "key": key, "error": str(exc)})
            return False

    # ── List operations (rolling log patterns) ───────────────────────────────

    def list_push(
        self,
        cache: str,
        key: str,
        value: str,
        ttl_seconds: int = 0,
        max_size: int = LIST_MAX_SIZE,
    ) -> bool:
        """Append *value* to a Momento list, trimming to *max_size*."""
        if not self.is_available():
            return False
        try:
            from momento.requests import CollectionTtl
            from momento.responses import CacheListPushBack
            ttl = CollectionTtl.of(timedelta(seconds=ttl_seconds)) if ttl_seconds > 0 else CollectionTtl.from_cache_ttl()
            resp = self._cache_client.list_push_back(cache, key, value, ttl=ttl)
            # Trim front if over max_size
            self._cache_client.list_retain(cache, key, -max_size, -1)
            return isinstance(resp, CacheListPushBack.Success)
        except Exception as exc:
            log_json("WARN", "momento_list_push_failed",
                     details={"cache": cache, "key": key, "error": str(exc)})
            return False

    def list_range(
        self,
        cache: str,
        key: str,
        start: int = 0,
        end: int = -1,
    ) -> List[str]:
        """Fetch a range of elements from a Momento list."""
        if not self.is_available():
            return []
        try:
            from momento.responses import CacheListFetch
            resp = self._cache_client.list_fetch(cache, key, start_index=start, end_index=end)
            if isinstance(resp, CacheListFetch.Hit):
                return resp.values_string
            return []
        except Exception as exc:
            log_json("WARN", "momento_list_range_failed",
                     details={"cache": cache, "key": key, "error": str(exc)})
            return []

    # ── Topics (event bus) ───────────────────────────────────────────────────

    def publish(self, topic: str, message: str) -> bool:
        """Publish a message to a Momento topic.  Returns True on success."""
        if not self.is_available() or not self._topics_client:
            return False
        try:
            from momento.responses import TopicPublish
            resp = self._topics_client.publish(WORKING_MEMORY_CACHE, topic, message)
            return isinstance(resp, TopicPublish.Success)
        except Exception as exc:
            log_json("WARN", "momento_publish_failed",
                     details={"topic": topic, "error": str(exc)})
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def ensure_caches(self) -> None:
        """Create the standard AURA caches if they don't already exist."""
        if not self.is_available():
            return
        try:
            from momento.responses import CreateCache
            for cache_name in (WORKING_MEMORY_CACHE, EPISODIC_MEMORY_CACHE):
                resp = self._cache_client.create_cache(cache_name)
                if isinstance(resp, CreateCache.Success):
                    log_json("INFO", "momento_cache_created", details={"cache": cache_name})
        except Exception as exc:
            log_json("WARN", "momento_ensure_caches_failed", details={"error": str(exc)})

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _lazy_init(self) -> None:
        self._initialized = True
        try:
            from momento import (
                CacheClient,
                TopicClient,
                Configurations,
                CredentialProvider,
            )
            cred = CredentialProvider.from_string(self._api_key)
            self._cache_client = CacheClient(
                configuration=Configurations.Laptop.v1(),
                credential_provider=cred,
                default_ttl=timedelta(seconds=DEFAULT_TTL_SECONDS),
            )
            self._topics_client = TopicClient(
                configuration=Configurations.Laptop.v1(),
                credential_provider=cred,
            )
            self.ensure_caches()
            self._available = True
            log_json("INFO", "momento_adapter_initialized",
                     details={"ttl": DEFAULT_TTL_SECONDS})
        except ImportError:
            log_json("WARN", "momento_sdk_not_installed",
                     details={"hint": "pip install momento"})
            self._available = False
        except Exception as exc:
            log_json("WARN", "momento_init_failed", details={"error": str(exc)})
            self._available = False
