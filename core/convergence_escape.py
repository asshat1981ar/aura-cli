"""
Convergence Escape Loop.

Detects when AURA is stuck in a local minimum: the same goal producing the
same verification failure for N consecutive cycles.  When stuck, selects and
applies an escape strategy rather than looping indefinitely.

Escape strategies
-----------------
overwrite         Switch act phase to overwrite mode (old_code_not_found errors).
different_model   Request a higher-quality model for next generation attempt.
skip_and_log      Record the goal as externally blocked and remove from queue.
decompose         Re-enqueue the goal prefixed with [DECOMPOSE] for sub-goal splitting.

Usage::

    from core.convergence_escape import ConvergenceEscapeLoop
    escape = ConvergenceEscapeLoop(memory_store, goal_queue)

    # After every run_cycle():
    action = escape.check_and_escape(goal, cycle_entry)
    if action:
        # action is a dict: {"strategy": "...", "hint": {...}}
        # Pass hint into next run_cycle() call as override
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json

# How many consecutive same-failure cycles before triggering escape
STUCK_THRESHOLD: int = 3

# Minimum history to evaluate
MIN_HISTORY: int = 3


class ConvergenceEscapeLoop:
    """Detect stuck loops and apply escape strategies."""

    def __init__(self, memory_store, goal_queue):
        self.memory = memory_store
        self.queue = goal_queue

    # ── Public API ───────────────────────────────────────────────────────────

    def check_and_escape(
        self, goal: str, cycle_entry: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyse recent history for *goal* and escape if stuck.

        Returns:
            A strategy dict ``{"strategy": str, "hint": dict}`` if an escape
            was triggered, else ``None``.
        """
        try:
            return self._check(goal, cycle_entry)
        except Exception as exc:
            log_json("ERROR", "convergence_escape_failed", details={"error": str(exc)})
            return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check(self, goal: str, cycle_entry: Dict) -> Optional[Dict]:
        recent = self.memory.read_log(limit=20)

        # Filter to entries that belong to this goal
        goal_lower = goal.lower()
        goal_cycles = [
            e for e in recent
            if goal_lower in str(e.get("phase_outputs", {})).lower()
            or goal_lower in str(e.get("goal", "")).lower()
        ]

        if len(goal_cycles) < MIN_HISTORY:
            return None

        # Collect the last STUCK_THRESHOLD verification failure signatures
        failure_signatures: List[str] = []
        for entry in goal_cycles[-STUCK_THRESHOLD:]:
            verif = entry.get("phase_outputs", {}).get("verification", {})
            if isinstance(verif, dict) and verif.get("status") == "fail":
                failures = verif.get("failures", [])
                sig = failures[0] if failures else "unknown_failure"
                failure_signatures.append(str(sig)[:80])

        if len(failure_signatures) < STUCK_THRESHOLD:
            return None  # not stuck — not enough consecutive failures

        # All signatures identical → stuck
        unique_sigs = set(failure_signatures)
        if len(unique_sigs) > 1:
            return None  # failures are changing — still making progress

        stuck_sig = failure_signatures[-1]
        strategy = self._select_strategy(stuck_sig)

        log_json(
            "WARN", "convergence_escape_triggered",
            details={
                "goal": goal[:80],
                "stuck_signature": stuck_sig,
                "consecutive_failures": STUCK_THRESHOLD,
                "strategy": strategy,
            },
        )

        return self._apply_strategy(goal, strategy, stuck_sig)

    def _select_strategy(self, failure_signature: str) -> str:
        sig = failure_signature.lower()
        if "old_code_not_found" in sig or "not found" in sig:
            return "overwrite"
        if "syntax" in sig or "parse" in sig or "invalid" in sig:
            return "different_model"
        if "import" in sig or "module" in sig or "dependency" in sig or "env" in sig:
            return "skip_and_log"
        # Default: goal is too hard as-is — break it down
        return "decompose"

    def _apply_strategy(
        self, goal: str, strategy: str, stuck_sig: str
    ) -> Dict[str, Any]:
        hint: Dict[str, Any] = {}

        if strategy == "overwrite":
            # Signal to act phase to use overwrite_file=True
            hint = {"force_overwrite": True}

        elif strategy == "different_model":
            # Signal model adapter to use quality model
            hint = {"model_override": "quality"}

        elif strategy == "skip_and_log":
            self.memory.put("skipped_goals", {
                "goal": goal,
                "reason": "external_dependency_stuck",
                "failure_signature": stuck_sig,
                "timestamp": time.time(),
            })
            log_json("INFO", "convergence_escape_goal_skipped",
                     details={"goal": goal[:80], "reason": stuck_sig})
            hint = {"skip": True}

        elif strategy == "decompose":
            # Re-enqueue prefixed — GoalDecomposer will handle it
            decompose_goal = f"[DECOMPOSE] {goal}"
            self.queue.add(decompose_goal)
            log_json("INFO", "convergence_escape_decompose_queued",
                     details={"original_goal": goal[:80]})
            hint = {"decomposed": True}

        return {"strategy": strategy, "hint": hint, "stuck_signature": stuck_sig}
