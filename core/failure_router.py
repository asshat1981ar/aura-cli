"""Failure routing logic extracted from the orchestrator.

This module provides :class:`FailureRouter`, a dedicated component that
decides how the orchestrator should respond to a failed phase — retry the
act step, escalate to a full re-plan, skip (environmental issue), or abort.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional


class FailureAction(Enum):
    """Recommended action after a phase failure."""

    RETRY_ACT = auto()
    """Recoverable code-level error — retry the act phase."""

    REPLAN = auto()
    """Structural or design error — re-plan from scratch."""

    SKIP = auto()
    """External / environmental issue that cannot be self-fixed."""

    ABORT = auto()
    """Fatal error — stop the current cycle entirely."""


# Signals that indicate an external / environmental cause rather than a
# code-level bug that the agent can fix.
_ENVIRONMENTAL_SIGNALS = (
    "network",
    "connection",
    "timeout",
    "timed out",
    "dns",
    "certificate",
    "ssl",
)

# Pattern that indicates some tests are still passing (used to distinguish
# partial failures from total failures in the verify phase).
_PASSING_TEST_PATTERN = "passed"


class FailureRouter:
    """Route a phase failure to the appropriate recovery action.

    Args:
        max_act_retries: Maximum number of act-phase retries before
            escalating to a re-plan.  Defaults to 3.
    """

    def __init__(self, max_act_retries: int = 3) -> None:
        self.max_act_retries = max_act_retries

    def route_failure(
        self,
        phase: str,
        attempt: int,
        error: str,
        verification_output: Optional[str] = None,
    ) -> FailureAction:
        """Determine how the orchestrator should recover from a failure.

        Args:
            phase: The pipeline phase that failed (e.g. ``"sandbox"``,
                ``"verify"``).
            attempt: Current attempt number (1-based).  When this reaches
                *max_act_retries* the router escalates to REPLAN.
            error: Error message or summary string from the failed phase.
            verification_output: Optional output from the verify phase (e.g.
                pytest stdout).  When this contains passing-test indicators
                the failure is treated as a partial/recoverable failure.

        Returns:
            A :class:`FailureAction` value.
        """
        error_lower = (error or "").lower()

        # Environmental errors cannot be fixed by the agent — skip.
        if any(signal in error_lower for signal in _ENVIRONMENTAL_SIGNALS):
            return FailureAction.SKIP

        if phase == "sandbox":
            if attempt < self.max_act_retries:
                return FailureAction.RETRY_ACT
            return FailureAction.REPLAN

        if phase == "verify":
            has_passing = (
                verification_output is not None
                and _PASSING_TEST_PATTERN in (verification_output or "").lower()
                and self._has_some_passing(verification_output)
            )
            if has_passing:
                if attempt < self.max_act_retries:
                    return FailureAction.RETRY_ACT
                return FailureAction.REPLAN
            return FailureAction.REPLAN

        # Default for unknown phases — skip rather than loop forever.
        return FailureAction.SKIP

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _has_some_passing(verification_output: str) -> bool:
        """Return True when *verification_output* contains a non-zero passed count.

        Handles common pytest-style summaries like ``"3 passed, 2 failed"``.
        """
        import re

        match = re.search(r"(\d+)\s+passed", verification_output, re.IGNORECASE)
        if match:
            return int(match.group(1)) > 0
        return False
