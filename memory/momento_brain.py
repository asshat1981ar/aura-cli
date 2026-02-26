"""
MomentoBrain — Brain with L1 (Momento Cache) + L2 (SQLite) write-through.

Extends the standard :class:`Brain` class with a hot Momento cache layer.
When Momento is available, frequently accessed data (recent memories,
weaknesses, response cache) is served from L1 for sub-ms reads.  Writes
always go to both L1 and L2 so the SQLite store remains the source of
truth.

When Momento is unavailable (no ``MOMENTO_API_KEY``), this class behaves
**identically** to the base ``Brain`` class — all L1 calls are no-ops.

Cache key layout (all in ``aura-working-memory``):
    memory:recent         JSON list — last 50 recall_all() results
    weaknesses:recent     JSON list — last 20 recall_weaknesses() results
    response:<hash>       String — cached LLM response (TTL = cache_ttl)

Usage::

    from memory.momento_brain import MomentoBrain
    from memory.momento_adapter import MomentoAdapter
    adapter = MomentoAdapter()
    brain = MomentoBrain(adapter)
    brain.remember({"key": "value"})   # writes to L1 + L2
    brain.recall_all()                 # reads from L1 (fast path)
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, List, Optional

from memory.brain import Brain
from memory.momento_adapter import MomentoAdapter, WORKING_MEMORY_CACHE
from core.logging_utils import log_json

# L1 TTLs
_MEMORY_TTL    = 12 * 3600   # 12 hours
_WEAKNESS_TTL  = 12 * 3600   # 12 hours
_RESPONSE_TTL  = 3600        # 1 hour (overridden by enable_cache ttl_seconds)

# Max items to cache in L1 lists
_MEMORY_LIST_MAX    = 50
_WEAKNESS_LIST_MAX  = 20

_MEMORY_KEY   = "memory:recent"
_WEAKNESS_KEY = "weaknesses:recent"


class MomentoBrain(Brain):
    """Write-through L1/L2 brain — Momento cache over the standard Brain.

    Args:
        momento: A :class:`MomentoAdapter` instance.  If not provided, a new
            one is created (it will be a no-op adapter if no API key is set).
    """

    def __init__(self, momento: Optional[MomentoAdapter] = None):
        super().__init__()
        self._m: MomentoAdapter = momento or MomentoAdapter()

    # ── Memory recall/store ───────────────────────────────────────────────────

    def remember(self, data: Any) -> None:
        """Store *data* in L2 (SQLite) and push to L1 list."""
        super().remember(data)
        if not self._m.is_available():
            return
        try:
            text = json.dumps(data) if isinstance(data, dict) else str(data)
            self._m.list_push(
                WORKING_MEMORY_CACHE, _MEMORY_KEY, text,
                ttl_seconds=_MEMORY_TTL,
                max_size=_MEMORY_LIST_MAX,
            )
        except Exception as exc:
            log_json("WARN", "momento_brain_remember_l1_failed",
                     details={"error": str(exc)})

    def recall_all(self) -> List[str]:
        """Return all memories — from L1 if available, else L2."""
        if self._m.is_available():
            items = self._m.list_range(WORKING_MEMORY_CACHE, _MEMORY_KEY)
            if items:
                log_json("DEBUG", "momento_brain_recall_l1_hit",
                         details={"count": len(items)})
                return items
            # L1 miss — load from L2 and backfill
            all_items = super().recall_all()
            self._backfill_memory_list(all_items)
            return all_items
        return super().recall_all()

    def recall_weaknesses(self) -> List[str]:
        """Return weaknesses — from L1 if available, else L2."""
        if self._m.is_available():
            items = self._m.list_range(WORKING_MEMORY_CACHE, _WEAKNESS_KEY)
            if items:
                return items
            # L1 miss — load from L2 and backfill
            weaknesses = super().recall_weaknesses()
            self._backfill_weakness_list(weaknesses)
            return weaknesses
        return super().recall_weaknesses()

    def add_weakness(self, weakness_description: str) -> None:
        """Add a weakness to L2 and push to L1."""
        super().add_weakness(weakness_description)
        if not self._m.is_available():
            return
        try:
            self._m.list_push(
                WORKING_MEMORY_CACHE, _WEAKNESS_KEY, weakness_description,
                ttl_seconds=_WEAKNESS_TTL,
                max_size=_WEAKNESS_LIST_MAX,
            )
        except Exception as exc:
            log_json("WARN", "momento_brain_weakness_l1_failed",
                     details={"error": str(exc)})

    # ── Response cache (L1 + L2) ──────────────────────────────────────────────

    def enable_cache(self, db_conn, ttl_seconds: int = 3600) -> None:
        """Enable L2 response cache (SQLite) and configure L1 TTL."""
        super().enable_cache(db_conn, ttl_seconds=ttl_seconds)
        self._response_ttl = ttl_seconds
        log_json("INFO", "momento_brain_cache_enabled",
                 details={"ttl": ttl_seconds, "l1_active": self._m.is_available()})

    def _get_cached_response(self, prompt: str):
        """Check L1 first; fall back to L2 SQLite."""
        if self._m.is_available():
            key = self._response_key(prompt)
            val = self._m.cache_get(WORKING_MEMORY_CACHE, key)
            if val is not None:
                log_json("INFO", "momento_brain_response_l1_hit",
                         details={"key": key[:16]})
                return val
        return super()._get_cached_response(prompt)

    def _save_to_cache(self, prompt: str, response: str) -> None:
        """Save to L2 and write-through to L1."""
        super()._save_to_cache(prompt, response)
        if not self._m.is_available():
            return
        try:
            key = self._response_key(prompt)
            ttl = getattr(self, "_response_ttl", _RESPONSE_TTL)
            self._m.cache_set(WORKING_MEMORY_CACHE, key, response, ttl_seconds=ttl)
        except Exception as exc:
            log_json("WARN", "momento_brain_response_l1_save_failed",
                     details={"error": str(exc)})

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _response_key(prompt: str) -> str:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return f"response:{h}"

    def _backfill_memory_list(self, items: List[str]) -> None:
        """Push last N memories into L1 list (one-time warm-up)."""
        try:
            for item in items[-_MEMORY_LIST_MAX:]:
                self._m.list_push(
                    WORKING_MEMORY_CACHE, _MEMORY_KEY, item,
                    ttl_seconds=_MEMORY_TTL,
                    max_size=_MEMORY_LIST_MAX,
                )
        except Exception as exc:
            log_json("WARN", "momento_brain_backfill_failed", details={"error": str(exc)})

    def _backfill_weakness_list(self, items: List[str]) -> None:
        """Push last N weaknesses into L1 list (one-time warm-up)."""
        try:
            for item in items[-_WEAKNESS_LIST_MAX:]:
                self._m.list_push(
                    WORKING_MEMORY_CACHE, _WEAKNESS_KEY, item,
                    ttl_seconds=_WEAKNESS_TTL,
                    max_size=_WEAKNESS_LIST_MAX,
                )
        except Exception as exc:
            log_json("WARN", "momento_brain_weakness_backfill_failed",
                     details={"error": str(exc)})
