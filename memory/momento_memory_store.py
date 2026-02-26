"""
MomentoMemoryStore — MemoryStore with Momento L1 hot cache + event publishing.

Extends :class:`MemoryStore` to:

1. **Hot-cache tier data** — ``put()`` and ``query()`` mirror data to/from
   Momento list keys in ``aura-episodic-memory``.  Reads hit L1 first and
   fall through to the JSON files only on a cache miss, keeping planning
   hint retrieval sub-millisecond.

2. **Publish cycle events** — ``append_log()`` writes the decision-log
   entry as usual AND publishes a compact event to the
   ``aura.cycle_complete`` Momento Topic.  This enables future improvement
   loops to subscribe reactively instead of polling.

Cache key layout (all in ``aura-episodic-memory``):
    tier:<tier_name>      Momento list — rolling window of tier records

Topic messages published by this store:
    aura.cycle_complete   ``{"cycle_id": "...", "goal_type": "...",
                             "verify_status": "...", "ts": 1234567890.0}``

Fallback behaviour (no Momento API key):
    Identical to base :class:`MemoryStore` — all L1 operations are no-ops.

Usage::

    from memory.momento_memory_store import MomentoMemoryStore
    from memory.momento_adapter import MomentoAdapter
    adapter = MomentoAdapter()
    store = MomentoMemoryStore(path, adapter)
    store.put("cycle_summaries", {"goal": "...", "status": "pass"})
    store.append_log(cycle_entry)   # also fires aura.cycle_complete
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.store import MemoryStore
from memory.momento_adapter import (
    MomentoAdapter,
    EPISODIC_MEMORY_CACHE,
    TOPIC_CYCLE_COMPLETE,
)
from core.logging_utils import log_json

# Rolling window kept in L1
_LIST_MAX = 100

# TTL for episodic memory lists (24 hours — long enough to survive restarts)
_EPISODIC_TTL = 24 * 3600


def _tier_key(tier: str) -> str:
    return f"tier:{tier}"


class MomentoMemoryStore(MemoryStore):
    """Write-through hot-cache MemoryStore backed by Momento.

    Args:
        root: Same as :class:`MemoryStore` — local directory for JSON files.
        momento: A :class:`MomentoAdapter` instance.
    """

    def __init__(self, root: Path, momento: Optional[MomentoAdapter] = None):
        super().__init__(root)
        self._m: MomentoAdapter = momento or MomentoAdapter()

    # ── Tier put/query ────────────────────────────────────────────────────────

    def put(self, tier: str, record: Dict[str, Any]) -> None:
        """Write *record* to JSON file (L2) and push to Momento list (L1)."""
        super().put(tier, record)
        if not self._m.is_available():
            return
        try:
            serialized = json.dumps(record)
            self._m.list_push(
                EPISODIC_MEMORY_CACHE,
                _tier_key(tier),
                serialized,
                ttl_seconds=_EPISODIC_TTL,
                max_size=_LIST_MAX,
            )
        except Exception as exc:
            log_json("WARN", "momento_memstore_put_l1_failed",
                     details={"tier": tier, "error": str(exc)})

    def query(self, tier: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Return last *limit* records — from L1 if available, else L2 JSON."""
        if self._m.is_available():
            try:
                raw_items = self._m.list_range(
                    EPISODIC_MEMORY_CACHE,
                    _tier_key(tier),
                    start=max(0, _LIST_MAX - limit),
                )
                if raw_items:
                    parsed = []
                    for item in raw_items[-limit:]:
                        try:
                            parsed.append(json.loads(item))
                        except json.JSONDecodeError:
                            pass
                    if parsed:
                        log_json("DEBUG", "momento_memstore_query_l1_hit",
                                 details={"tier": tier, "count": len(parsed)})
                        return parsed
                # L1 miss — load from L2 and backfill
                results = super().query(tier, limit=limit)
                self._backfill_tier(tier, results)
                return results
            except Exception as exc:
                log_json("WARN", "momento_memstore_query_l1_failed",
                         details={"tier": tier, "error": str(exc)})
        return super().query(tier, limit=limit)

    # ── Decision log + topic publish ──────────────────────────────────────────

    def append_log(self, entry: Dict[str, Any]) -> None:
        """Write JSONL entry (L2) and publish cycle event to Topics (L1)."""
        super().append_log(entry)
        if not self._m.is_available():
            return
        try:
            # Compact event payload — only what subscribers need
            po = entry.get("phase_outputs", {})
            verify_status = po.get("verification", {}).get("status", "unknown")
            event = {
                "cycle_id":     entry.get("cycle_id", ""),
                "goal_type":    entry.get("goal_type", ""),
                "verify_status": verify_status,
                "stop_reason":  entry.get("stop_reason"),
                "ts":           time.time(),
            }
            self._m.publish(TOPIC_CYCLE_COMPLETE, json.dumps(event))
        except Exception as exc:
            log_json("WARN", "momento_memstore_publish_failed",
                     details={"error": str(exc)})

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _backfill_tier(self, tier: str, records: List[Dict]) -> None:
        """Push existing L2 records into L1 (one-time warm-up on L1 miss)."""
        try:
            for record in records[-_LIST_MAX:]:
                self._m.list_push(
                    EPISODIC_MEMORY_CACHE,
                    _tier_key(tier),
                    json.dumps(record),
                    ttl_seconds=_EPISODIC_TTL,
                    max_size=_LIST_MAX,
                )
        except Exception as exc:
            log_json("WARN", "momento_memstore_backfill_failed",
                     details={"tier": tier, "error": str(exc)})
