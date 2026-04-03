"""Failure routing logic extracted from orchestrator (#317).

Provides type-safe failure classification and routing decisions
for the AURA orchestration loop.
"""
from enum import Enum, auto
from typing import Dict, List


class FailureAction(Enum):
    """Action to take after a phase failure."""
    RETRY_ACT = auto()   # Retry the act phase (code-level fix)
    REPLAN = auto()      # Replan from scratch (structural/design issue)
    SKIP = auto()        # Skip due to external/environmental issue
    ABORT = auto()       # Abort the cycle (unrecoverable)


# Environmental/external failure keywords
ENVIRONMENTAL_KEYWORDS: List[str] = [
    "dependency",
    "network",
    "env",
    "environment",
    "permission",
    "not found",
    "no module",
    "import error",
    "connection",
    "timeout",
    "dns",
    "certificate",
]

# Structural/design failure keywords
STRUCTURAL_KEYWORDS: List[str] = [
    "architecture",
    "circular",
    "api_breaking",
    "breaking_change",
    "design",
    "interface",
    "contract",
]


class FailureRouter:
    """Routes phase failures to appropriate recovery actions.
    
    Extracted from LoopOrchestrator to reduce coupling and enable
    independent testing.
    """

    def __init__(self, max_act_retries: int = 3):
        self.max_act_retries = max_act_retries

    def route_failure(
        self,
        verification: Dict,
        attempt: int = 1,
    ) -> FailureAction:
        """Classify a verification failure and return the recommended action.

        Args:
            verification: The dict returned by the verify phase.
            attempt: Current retry attempt number (1-indexed).

        Returns:
            FailureAction indicating how to proceed.
        """
        failures = " ".join(str(f) for f in verification.get("failures", []))
        logs = str(verification.get("logs", ""))
        combined = (failures + " " + logs).lower()

        # Check for external/environmental issues first
        if any(kw in combined for kw in ENVIRONMENTAL_KEYWORDS):
            return FailureAction.SKIP

        # Check for structural/design issues
        if any(kw in combined for kw in STRUCTURAL_KEYWORDS):
            return FailureAction.REPLAN

        # Default: retry act phase if under max retries
        if attempt < self.max_act_retries:
            return FailureAction.RETRY_ACT
        return FailureAction.REPLAN

    def route(self, verification: Dict) -> str:
        """Legacy-compatible routing returning string literals.
        
        Returns:
            One of: "act", "plan", "skip"
        """
        action = self.route_failure(verification, attempt=1)
        mapping = {
            FailureAction.RETRY_ACT: "act",
            FailureAction.REPLAN: "plan",
            FailureAction.SKIP: "skip",
            FailureAction.ABORT: "skip",
        }
        return mapping.get(action, "act")
