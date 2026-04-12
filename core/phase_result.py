"""Phase result with confidence scoring for smart orchestrator routing.

Each orchestrator phase returns a PhaseResult with a confidence score (0.0-1.0)
and a suggested next action. The ConfidenceRouter uses these scores to make
data-driven routing decisions instead of hard-coded pass/fail logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NextAction(str, Enum):
    """Suggested next action after a phase completes."""

    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    ESCALATE = "escalate"
    SKIP = "skip"
    DECOMPOSE = "decompose"


@dataclass
class PhaseResult:
    """Result from any orchestrator phase with confidence scoring."""

    phase: str
    output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    suggested_next: NextAction = NextAction.CONTINUE
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.3

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "confidence": self.confidence,
            "suggested_next": self.suggested_next.value,
            "reasoning": self.reasoning,
            "output": self.output,
            "metadata": self.metadata,
        }


class ConfidenceRouter:
    """Routes orchestrator flow based on phase confidence scores.

    Tracks confidence across phases within a cycle and detects patterns
    like declining confidence (suggesting escalation) or consistently
    high confidence (suggesting optional phase skipping).
    """

    OPTIONAL_PHASES = {"critique", "skill_dispatch"}

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = thresholds or {
            "replan_below": 0.3,
            "escalate_below": 0.2,
            "retry_below": 0.4,
            "skip_above": 0.9,
        }
        self.phase_history: list[PhaseResult] = []

    def record(self, result: PhaseResult):
        """Record a phase result for trend analysis."""
        self.phase_history.append(result)

    def should_replan(self, result: PhaseResult) -> bool:
        if result.suggested_next == NextAction.REPLAN:
            return True
        if result.phase in ("plan", "critique") and result.confidence < self.thresholds["replan_below"]:
            return True
        return False

    def should_escalate(self, result: PhaseResult) -> bool:
        if result.suggested_next == NextAction.ESCALATE:
            return True
        if result.confidence < self.thresholds["escalate_below"]:
            return True
        # Escalate if confidence is trending down across 3+ phases
        if len(self.phase_history) >= 3:
            recent = [r.confidence for r in self.phase_history[-3:]]
            if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                if recent[-1] < self.thresholds["retry_below"]:
                    return True
        return False

    def should_retry(self, result: PhaseResult) -> bool:
        if result.suggested_next == NextAction.RETRY:
            return True
        if result.confidence < self.thresholds["retry_below"]:
            return True
        return False

    def should_skip_optional(self, result: PhaseResult, next_phase: str) -> bool:
        """Check if an optional phase can be skipped due to high confidence."""
        if next_phase in self.OPTIONAL_PHASES and result.confidence > self.thresholds["skip_above"]:
            return True
        return False

    def get_cycle_confidence(self) -> float:
        """Get overall confidence for the current cycle."""
        if not self.phase_history:
            return 0.5
        return sum(r.confidence for r in self.phase_history) / len(self.phase_history)

    def reset(self):
        """Reset for a new cycle."""
        self.phase_history.clear()
