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
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from core.logging_utils import log_json

# How many consecutive same-failure cycles before triggering escape
STUCK_THRESHOLD: int = 3

# Number of verify scores tracked for oscillation detection
OSCILLATION_WINDOW: int = 8


class OscillationDetector:
    """Detect alternating pass/fail patterns in verify scores.

    Tracks the last *window* verify scores and flags when a pass→fail→pass
    alternation has repeated at least *min_alternations* times.  Oscillation
    suggests the current strategy is bouncing around a saddle point rather
    than converging; the caller should switch strategies.

    Usage::

        detector = OscillationDetector()
        detector.record(0.8)   # pass
        detector.record(0.3)   # fail
        detector.record(0.9)   # pass
        if detector.is_oscillating():
            strategy = detector.suggest_strategy()
    """

    PASS_THRESHOLD: float = 0.5

    def __init__(
        self,
        window: int = OSCILLATION_WINDOW,
        min_alternations: int = 3,
    ):
        """Initialise the detector.

        Args:
            window: Maximum number of recent scores to retain.
            min_alternations: Number of direction changes required before
                :meth:`is_oscillating` returns ``True``.
        """
        self._window = window
        self._min_alternations = min_alternations
        self._scores: Deque[float] = deque(maxlen=window)

    # ── Public API ───────────────────────────────────────────────────────────

    def record(self, score: float) -> None:
        """Append a new verify *score* to the sliding window.

        Args:
            score: Numeric verification score in [0, 1] where values above
                :attr:`PASS_THRESHOLD` count as pass.
        """
        self._scores.append(float(score))

    def is_oscillating(self) -> bool:
        """Return ``True`` if the recorded scores show an alternating pattern.

        An *alternation* is a transition between pass (> threshold) and fail
        (<= threshold) or vice-versa.  Returns ``True`` once the number of
        such transitions meets or exceeds *min_alternations*.
        """
        scores = list(self._scores)
        if len(scores) < self._min_alternations + 1:
            return False

        passes = [s > self.PASS_THRESHOLD for s in scores]
        alternations = sum(
            1 for i in range(1, len(passes)) if passes[i] != passes[i - 1]
        )
        return alternations >= self._min_alternations

    def suggest_strategy(self) -> str:
        """Return a recommended escape strategy when oscillating.

        Heuristic: if the most recent score is a *pass* (the cycle just
        succeeded) prefer ``"vary_prompt"`` to slightly diversify the
        generation; if it was a *fail* prefer ``"replan"`` to start fresh.

        Returns:
            One of ``"vary_prompt"`` or ``"replan"``.
        """
        if self._scores and self._scores[-1] > self.PASS_THRESHOLD:
            return "vary_prompt"
        return "replan"

    def reset(self) -> None:
        """Clear all recorded scores."""
        self._scores.clear()

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
