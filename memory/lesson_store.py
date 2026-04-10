"""LessonStore — persistent cycle lessons for the AURA orchestrator.

Records outcomes from each orchestrator cycle and makes injectable
lessons available to the planner at the start of subsequent cycles.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class LessonStore:
    """Stores and retrieves lessons learned from orchestrator cycles.

    Lessons are persisted as JSONL in ``memory/lessons.jsonl``.  Each entry
    records the goal, outcome, failure reasons, and any insights that the
    reflector or skill dispatcher surfaced.

    Args:
        store_path: Path to the JSONL file. Defaults to ``memory/lessons.jsonl``.
        max_injectable: Maximum number of recent lessons to inject into the
            planner prompt.  Older lessons are still stored but not injected.
    """

    def __init__(self, store_path: Optional[Path] = None, max_injectable: int = 5):
        self.store_path = Path(store_path or "memory/lessons.jsonl")
        self.max_injectable = max_injectable
        self._lessons: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing lessons from disk."""
        if not self.store_path.exists():
            return
        try:
            for line in self.store_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._lessons.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass  # Start fresh on corruption

    def _save_entry(self, entry: Dict[str, Any]) -> None:
        """Append a single entry to the JSONL file."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def record_cycle(self, cycle_result: Dict[str, Any]) -> None:
        """Record a lesson from a completed cycle.

        Extracts the goal, stop_reason, phase outcomes, and any insights
        from the cycle result dict.

        Args:
            cycle_result: The dict returned by ``LoopOrchestrator.run_cycle()``.
        """
        entry = {
            "timestamp": time.time(),
            "goal": cycle_result.get("goal", ""),
            "goal_type": cycle_result.get("goal_type", ""),
            "stop_reason": cycle_result.get("stop_reason", ""),
            "status": cycle_result.get("status", ""),
            "verify_status": cycle_result.get("phase_outputs", {}).get("verify", {}).get("status", ""),
            "cycle_confidence": cycle_result.get("phase_outputs", {}).get("cycle_confidence", 0),
            "failure_context": cycle_result.get("phase_outputs", {}).get("_failure_context", ""),
        }
        self._lessons.append(entry)
        self._save_entry(entry)

    def injectable_lessons(self) -> List[Dict[str, Any]]:
        """Return recent lessons suitable for injection into planner context.

        Returns the most recent ``max_injectable`` lessons, formatted
        for inclusion in the planner's input data.
        """
        recent = self._lessons[-self.max_injectable :]
        return [
            {
                "goal": l.get("goal", ""),
                "outcome": l.get("stop_reason", ""),
                "confidence": l.get("cycle_confidence", 0),
                "lesson": _derive_lesson(l),
            }
            for l in recent
        ]

    def count(self) -> int:
        """Return the total number of stored lessons."""
        return len(self._lessons)


def _derive_lesson(entry: Dict[str, Any]) -> str:
    """Derive a human-readable lesson from a cycle entry."""
    stop = entry.get("stop_reason", "")
    verify = entry.get("verify_status", "")
    goal = entry.get("goal", "unknown")

    if stop == "MAX_CYCLES":
        return f"Goal '{goal[:50]}' hit max cycles — consider decomposing into smaller steps."
    if verify == "fail":
        return f"Goal '{goal[:50]}' failed verification — check test expectations."
    if verify == "pass":
        return f"Goal '{goal[:50]}' succeeded — similar approach may work for related goals."
    return f"Goal '{goal[:50]}' completed with stop_reason={stop}."
