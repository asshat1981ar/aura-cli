"""Structured learning artifacts for the AURA Autonomous Learning Loop (PRD-003).

LearningArtifact is the canonical unit of what AURA has learned from a cycle:
it captures the insight, its severity, the originating cycle, and whether it
has been converted into a goal queue entry.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Valid artifact types
ARTIFACT_TYPES = frozenset(
    {
        "phase_failure",  # a pipeline phase is failing at high rate
        "skill_weakness",  # a skill produces low-signal output
        "quality_regression",  # a quality metric dropped below threshold
        "cycle_learning",  # a per-cycle learning string from ReflectorAgent
        "success_pattern",  # a pattern observed in successful cycles
    }
)

# Severity levels ordered from least to most urgent
SEVERITIES = ("low", "medium", "high", "critical")


@dataclass
class LearningArtifact:
    """A single structured unit of what AURA learned.

    Attributes:
        artifact_id:    Unique hex identifier (auto-generated).
        cycle_id:       ID of the cycle this was derived from.
        goal:           Goal text of that cycle.
        goal_type:      Classified goal type (e.g. "feature", "bug_fix").
        artifact_type:  Category — see ARTIFACT_TYPES.
        insight:        Human-readable lesson or observation.
        evidence:       Raw supporting data (alert fields, insight dict, etc.).
        suggested_goal: A remediation goal string to enqueue, if applicable.
        severity:       Urgency — one of SEVERITIES.
        created_at:     Unix timestamp of creation.
        acted_on:       True when suggested_goal has been enqueued.
    """

    artifact_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cycle_id: str = ""
    goal: str = ""
    goal_type: str = ""
    artifact_type: str = "cycle_learning"
    insight: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_goal: Optional[str] = None
    severity: str = "low"
    created_at: float = field(default_factory=time.time)
    acted_on: bool = False

    def is_actionable(self) -> bool:
        """True if this artifact has a suggested goal that hasn't been acted on."""
        return bool(self.suggested_goal) and not self.acted_on

    def mark_acted_on(self) -> None:
        """Mark the suggested goal as having been enqueued."""
        self.acted_on = True
