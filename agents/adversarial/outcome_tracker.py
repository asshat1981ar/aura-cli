"""Outcome tracker for adversarial critique validation."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json


@dataclass
class CritiqueOutcome:
    """Outcome of a critique validation."""
    critique_id: str
    was_validated: bool
    actual_severity: Optional[str]
    notes: Optional[str]
    validated_by: Optional[str] = None
    validation_timestamp: float = field(default_factory=time.time)


@dataclass
class TrackedCritique:
    """A critique being tracked for outcome."""
    critique_id: str
    target_type: str
    findings_count: int
    risk_score: float
    strategies_used: List[str]
    created_at: float
    outcome: Optional[CritiqueOutcome] = None


class CritiqueOutcomeTracker:
    """Tracks adversarial critiques and their validation outcomes."""
    
    def __init__(self, max_tracked: int = 1000):
        """
        Initialize the tracker.
        
        Args:
            max_tracked: Maximum number of critiques to track in memory
        """
        self._tracked: Dict[str, TrackedCritique] = {}
        self._max_tracked = max_tracked
        
    async def start_tracking(
        self,
        critique_id: str,
        critique: Any  # AdversarialCritique
    ) -> TrackedCritique:
        """
        Start tracking a critique for outcome validation.
        
        Args:
            critique_id: ID of the critique
            critique: The critique to track
            
        Returns:
            TrackedCritique object
        """
        tracked = TrackedCritique(
            critique_id=critique_id,
            target_type=critique.target_type.value,
            findings_count=len(critique.findings),
            risk_score=critique.risk_score,
            strategies_used=list(critique.strategy_results.keys()),
            created_at=critique.timestamp
        )
        
        self._tracked[critique_id] = tracked
        
        # Trim if needed
        self._trim_if_needed()
        
        log_json("DEBUG", "critique_tracking_started", {
            "critique_id": critique_id,
            "findings_count": tracked.findings_count
        })
        
        return tracked
    
    async def record_outcome(
        self,
        critique_id: str,
        was_validated: bool,
        actual_severity: Optional[str] = None,
        notes: Optional[str] = None,
        validated_by: Optional[str] = None
    ) -> Optional[TrackedCritique]:
        """
        Record the outcome of a critique validation.
        
        Args:
            critique_id: ID of the critique
            was_validated: Whether findings were validated
            actual_severity: Actual severity of issues found
            notes: Additional notes
            validated_by: Who validated the critique
            
        Returns:
            Updated TrackedCritique or None if not found
        """
        tracked = self._tracked.get(critique_id)
        if not tracked:
            log_json("WARN", "critique_not_found_for_outcome", {
                "critique_id": critique_id
            })
            return None
        
        outcome = CritiqueOutcome(
            critique_id=critique_id,
            was_validated=was_validated,
            actual_severity=actual_severity,
            notes=notes,
            validated_by=validated_by,
            validation_timestamp=time.time()
        )
        
        tracked.outcome = outcome
        
        log_json("INFO", "critique_outcome_recorded", {
            "critique_id": critique_id,
            "was_validated": was_validated,
            "time_to_validation": outcome.validation_timestamp - tracked.created_at
        })
        
        return tracked
    
    def get_tracked(self, critique_id: str) -> Optional[TrackedCritique]:
        """Get a tracked critique by ID."""
        return self._tracked.get(critique_id)
    
    def get_pending_validation(self) -> List[TrackedCritique]:
        """Get critiques awaiting validation."""
        return [
            t for t in self._tracked.values()
            if t.outcome is None
        ]
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = len(self._tracked)
        validated = sum(1 for t in self._tracked.values() if t.outcome)
        validated_true = sum(
            1 for t in self._tracked.values()
            if t.outcome and t.outcome.was_validated
        )
        validated_false = validated - validated_true
        
        pending = total - validated
        
        return {
            "total_tracked": total,
            "validated": validated,
            "validated_true": validated_true,
            "validated_false": validated_false,
            "pending": pending,
            "validation_rate": validated / total if total > 0 else 0,
            "accuracy_rate": validated_true / validated if validated > 0 else 0
        }
    
    def get_stats_by_strategy(self) -> Dict[str, Dict[str, Any]]:
        """Get validation statistics grouped by strategy."""
        stats: Dict[str, Dict[str, Any]] = {}
        
        for tracked in self._tracked.values():
            if not tracked.outcome:
                continue
            
            for strategy in tracked.strategies_used:
                if strategy not in stats:
                    stats[strategy] = {
                        "total": 0,
                        "validated_true": 0,
                        "validated_false": 0
                    }
                
                stats[strategy]["total"] += 1
                if tracked.outcome.was_validated:
                    stats[strategy]["validated_true"] += 1
                else:
                    stats[strategy]["validated_false"] += 1
        
        # Calculate rates
        for strategy in stats:
            total = stats[strategy]["total"]
            stats[strategy]["accuracy"] = (
                stats[strategy]["validated_true"] / total
                if total > 0 else 0
            )
        
        return stats
    
    def get_recent_outcomes(
        self,
        limit: int = 50,
        validated_only: bool = False
    ) -> List[TrackedCritique]:
        """Get recent critique outcomes."""
        critiques = list(self._tracked.values())
        
        if validated_only:
            critiques = [c for c in critiques if c.outcome]
        
        # Sort by creation time
        critiques.sort(key=lambda c: c.created_at, reverse=True)
        
        return critiques[:limit]
    
    def _trim_if_needed(self):
        """Trim old tracked critiques if over limit."""
        if len(self._tracked) <= self._max_tracked:
            return
        
        # Sort by creation time and keep newest
        sorted_ids = sorted(
            self._tracked.keys(),
            key=lambda k: self._tracked[k].created_at,
            reverse=True
        )
        
        # Keep max_tracked newest
        to_remove = sorted_ids[self._max_tracked:]
        for cid in to_remove:
            del self._tracked[cid]
        
        log_json("DEBUG", "critique_tracker_trimmed", {
            "removed": len(to_remove),
            "remaining": len(self._tracked)
        })
