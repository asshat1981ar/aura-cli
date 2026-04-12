"""Working memory module -- in-memory ring buffer for session-scoped context.

Stores the most recent observations, thoughts, and intermediate results for the
current task.  When the buffer is full the oldest entries are evicted; an
optional *summarize_fn* callback can condense evicted entries into a single
summary before they are lost.
"""

import threading
import time
import uuid
from typing import Callable, Optional

from core.logging_utils import log_json
from memory.memory_module import MemoryEntry, MemoryModule


class WorkingMemory(MemoryModule):
    """Thread-safe, in-memory ring buffer implementing :class:`MemoryModule`.

    Args:
        max_entries: Maximum number of entries the buffer will hold before
            evicting the oldest.  Defaults to 50.
        summarize_fn: An optional callback ``(list[MemoryEntry]) -> str`` that
            receives evicted entries and returns a condensed summary.  The
            summary is re-inserted as a single entry so context is not
            entirely lost.
    """

    def __init__(
        self,
        max_entries: int = 50,
        summarize_fn: Optional[Callable[[list[MemoryEntry]], str]] = None,
    ):
        self._max_entries = max_entries
        self._summarize_fn = summarize_fn
        self._entries: list[MemoryEntry] = []
        self._lock = threading.Lock()
        log_json(
            "INFO",
            "working_memory_init",
            details={"max_entries": max_entries},
        )

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def write(self, content: str, metadata: Optional[dict] = None) -> str:
        """Append an entry to the ring buffer, evicting oldest if full."""
        entry_id = uuid.uuid4().hex[:12]
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        with self._lock:
            self._entries.append(entry)
            self._maybe_evict()
        log_json(
            "INFO",
            "working_memory_write",
            details={"entry_id": entry_id, "content_snippet": content[:80]},
        )
        return entry_id

    def read(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Return recent entries whose content contains *query* (substring match).

        If *query* is empty, the most recent *top_k* entries are returned.
        """
        query_lower = query.lower()
        with self._lock:
            if not query_lower:
                matches = list(self._entries[-top_k:])
            else:
                matches = [
                    e for e in self._entries if query_lower in e.content.lower()
                ][-top_k:]
        # Assign simple recency-based scores (newest = highest).
        for idx, entry in enumerate(matches):
            entry.score = (idx + 1) / max(len(matches), 1)
        return matches

    def search(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """Alias for :meth:`read` -- working memory uses substring matching."""
        return self.read(query, top_k=top_k)

    def delete(self, entry_id: str) -> bool:
        """Remove a single entry by ID."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.id != entry_id]
            removed = len(self._entries) < before
        if removed:
            log_json(
                "INFO",
                "working_memory_delete",
                details={"entry_id": entry_id},
            )
        return removed

    def clear(self) -> None:
        """Wipe the buffer -- typically called at the start of a new task."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        log_json(
            "INFO",
            "working_memory_clear",
            details={"entries_cleared": count},
        )

    def stats(self) -> dict:
        """Return buffer utilisation statistics."""
        with self._lock:
            count = len(self._entries)
        return {
            "type": "working",
            "count": count,
            "max_entries": self._max_entries,
            "utilization_pct": round(count / self._max_entries * 100, 1)
            if self._max_entries
            else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evict oldest entries when the buffer exceeds *max_entries*.

        Must be called while *self._lock* is held.
        """
        if len(self._entries) <= self._max_entries:
            return

        overflow = len(self._entries) - self._max_entries
        evicted = self._entries[:overflow]
        self._entries = self._entries[overflow:]

        log_json(
            "INFO",
            "working_memory_evict",
            details={"evicted_count": len(evicted)},
        )

        # Optionally summarize the evicted entries and re-insert a condensed
        # version so context is not entirely lost.
        if self._summarize_fn is not None:
            try:
                summary = self._summarize_fn(evicted)
                if summary:
                    condensed = MemoryEntry(
                        id=uuid.uuid4().hex[:12],
                        content=summary,
                        metadata={"type": "eviction_summary", "evicted_count": len(evicted)},
                        timestamp=time.time(),
                    )
                    self._entries.insert(0, condensed)
            except Exception as exc:
                log_json(
                    "WARN",
                    "working_memory_summarize_failed",
                    details={"error": str(exc)},
                )
