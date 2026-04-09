"""Tracks the goal currently executing to enable crash recovery.

When the goal run loop pops a goal from the queue, it calls
``InFlightTracker.write()`` before entering the orchestrator. If the process is
killed between dequeue and archive (SIGKILL, OOM, power loss), the record
survives on disk and can be recovered with ``python3 main.py goal resume``.

The record is written atomically (write to ``.tmp`` then ``os.replace``) so a
kill during the write itself cannot produce a partial/corrupt file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_PATH = Path("memory") / "in_flight_goal.json"


class InFlightTracker:
    """Persists the currently-executing goal to ``memory/in_flight_goal.json``.

    Example usage in ``run_goals_loop()``::

        tracker = InFlightTracker()
        goal = goal_queue.next()
        tracker.write(goal, cycle_limit)
        try:
            result = orchestrator.run_loop(goal, ...)
            goal_archive.record(goal, score)
        finally:
            tracker.clear()
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _DEFAULT_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, goal: str, cycle_limit: int, phase: str = "ingest") -> None:
        """Record *goal* as in-flight.

        Writes atomically: data is first written to a ``.tmp`` file, then
        renamed over the target path via :func:`os.replace`.  This guarantees
        that a process kill during the write cannot leave a partial file.

        Args:
            goal: The goal string exactly as dequeued from ``GoalQueue``.
            cycle_limit: The cycle limit in use for this execution.
            phase: The current pipeline phase (reserved for future phase-aware
                resume; always ``"ingest"`` at write time for now).
        """
        data = {
            "goal": goal,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cycle_limit": cycle_limit,
            "phase": phase,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def read(self) -> Optional[dict]:
        """Return the in-flight record, or ``None`` if no interrupted goal exists.

        Returns ``None`` (rather than raising) if the file is absent, empty, or
        contains invalid JSON.
        """
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        """Remove the in-flight record.

        Safe to call even when the file does not exist (e.g. in dry-run mode or
        when called from a ``finally`` block that was never preceded by
        ``write``).
        """
        self._path.unlink(missing_ok=True)

    def exists(self) -> bool:
        """Return ``True`` if an in-flight record is present on disk."""
        return self._path.exists()

    def get_summary(self) -> Optional[dict]:
        """Get a human-readable summary of the in-flight goal.
        
        Returns:
            Summary dict with formatted timestamps and elapsed time,
            or None if no goal is in-flight.
        """
        record = self.read()
        if not record:
            return None
        
        from datetime import datetime
        
        started_at = record.get("started_at", "")
        goal = record.get("goal", "Unknown")
        
        try:
            started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
            
            # Format elapsed time
            if elapsed < 60:
                elapsed_str = f"{int(elapsed)}s"
            elif elapsed < 3600:
                elapsed_str = f"{int(elapsed / 60)}m {int(elapsed % 60)}s"
            else:
                hours = int(elapsed / 3600)
                minutes = int((elapsed % 3600) / 60)
                elapsed_str = f"{hours}h {minutes}m"
            
            return {
                "goal": goal,
                "started_at": started_at,
                "started_at_formatted": started_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_formatted": elapsed_str,
                "cycle_limit": record.get("cycle_limit", 10),
            }
        except (ValueError, TypeError):
            return {
                "goal": goal,
                "started_at": started_at,
                "started_at_formatted": started_at,
                "elapsed_seconds": 0,
                "elapsed_formatted": "unknown",
                "cycle_limit": record.get("cycle_limit", 10),
            }


# Global instance helpers
_tracker: Optional[InFlightTracker] = None


def get_tracker() -> InFlightTracker:
    """Get global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = InFlightTracker()
    return _tracker
