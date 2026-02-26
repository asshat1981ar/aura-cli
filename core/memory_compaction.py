"""
Memory Compaction Loop.

Prevents ``memory/decision_log.jsonl`` from growing unboundedly.  When the
log exceeds ``MAX_ENTRIES``, old entries (everything outside the ``KEEP_RECENT``
most recent) are compressed into per-goal-type rolling summary records and
appended to ``memory/compacted_history.jsonl``.  Only the recent entries are
written back to the live log.

Usage::

    from core.memory_compaction import MemoryCompactionLoop
    compactor = MemoryCompactionLoop(memory_store)
    compactor.on_cycle_complete(cycle_entry)   # auto-trigger
    result = compactor.run()                   # or trigger manually
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from core.logging_utils import log_json

MAX_ENTRIES: int = 200   # trigger compaction when log exceeds this
KEEP_RECENT: int = 50    # keep this many entries in the live log after compaction
COMPACTED_LOG_NAME: str = "compacted_history.jsonl"


class MemoryCompactionLoop:
    """Compact the decision log to prevent unbounded growth."""

    def __init__(self, memory_store):
        self.memory = memory_store
        self._compacted_path = Path(memory_store.root) / COMPACTED_LOG_NAME

    # ── Public API ───────────────────────────────────────────────────────────

    def on_cycle_complete(self, _cycle_entry: Dict[str, Any]) -> None:
        """Check log size and compact if necessary.  Never raises."""
        try:
            entries = self.memory.read_log(limit=MAX_ENTRIES + 1)
            if len(entries) > MAX_ENTRIES:
                self.run()
        except Exception as exc:
            log_json("ERROR", "memory_compaction_check_failed", details={"error": str(exc)})

    def run(self) -> Dict[str, Any]:
        """Compact old entries.  Never raises."""
        try:
            return self._run()
        except Exception as exc:
            log_json("ERROR", "memory_compaction_failed", details={"error": str(exc)})
            return {"error": str(exc)}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self) -> Dict[str, Any]:
        all_entries = self.memory.read_log(limit=10_000)
        if len(all_entries) <= KEEP_RECENT:
            return {"skipped": True, "reason": "nothing_to_compact"}

        recent = all_entries[-KEEP_RECENT:]
        old = all_entries[:-KEEP_RECENT]

        summaries = self._compress(old)
        self._write_compacted(summaries)
        self._rewrite_log(recent)

        log_json(
            "INFO", "memory_compaction_complete",
            details={
                "entries_compacted": len(old),
                "entries_kept": len(recent),
                "summaries_written": len(summaries),
            },
        )
        return {
            "entries_compacted": len(old),
            "entries_kept": len(recent),
            "summaries_written": len(summaries),
        }

    def _compress(self, entries: List[Dict]) -> List[Dict]:
        """Compress a batch of entries into per-goal-type summary records."""
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for e in entries:
            gt = e.get("goal_type", "unknown")
            by_type[gt].append(e)

        summaries: List[Dict] = []
        for goal_type, group in by_type.items():
            total = len(group)
            passed = sum(
                1 for e in group
                if e.get("phase_outputs", {}).get("verification", {}).get("status")
                in ("pass", "skip")
            )

            # Collect unique learnings
            learnings: List[str] = []
            for e in group:
                for l in e.get("phase_outputs", {}).get("reflection", {}).get("learnings", []):
                    if l and l not in learnings:
                        learnings.append(l)

            summaries.append({
                "type": "compacted_summary",
                "goal_type": goal_type,
                "cycles": total,
                "pass_rate": round(passed / max(total, 1), 3),
                "unique_learnings": learnings[:20],  # cap at 20
                "compacted_at": time.time(),
            })
        return summaries

    def _write_compacted(self, summaries: List[Dict]) -> None:
        self._compacted_path.parent.mkdir(parents=True, exist_ok=True)
        with self._compacted_path.open("a", encoding="utf-8") as fh:
            for s in summaries:
                fh.write(json.dumps(s) + "\n")

    def _rewrite_log(self, recent: List[Dict]) -> None:
        """Overwrite the live log with only the recent entries."""
        with self.memory.log_path.open("w", encoding="utf-8") as fh:
            for e in recent:
                fh.write(json.dumps(e) + "\n")
