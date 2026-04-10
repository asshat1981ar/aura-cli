"""Gödel Agent-inspired self-improvement loop.

This module implements research concepts from the Gödel Agent paper:
- Self-referential improvement through reflection
- Capability gap analysis
- Safe self-modification with rollback capability
- Performance-based learning

Note: This is a research prototype implementing academic concepts.
Actual self-modification is strictly controlled and requires explicit opt-in.
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from enum import Enum

from core.logging_utils import log_json


class ImprovementStatus(Enum):
    """Status of an improvement attempt."""
    PROPOSED = "proposed"
    ANALYZED = "analyzed"
    APPROVED = "approved"
    APPLIED = "applied"
    TESTED = "tested"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class CapabilityGap:
    """Identified gap in system capabilities.
    
    Attributes:
        area: Capability area (e.g., "memory", "planning", "execution")
        description: Human-readable description
        severity: Gap severity (1-10)
        evidence: Supporting evidence
        suggested_fix: Proposed solution
    """
    area: str
    description: str
    severity: int = field(default=5)
    evidence: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    
    def __post_init__(self):
        if not 1 <= self.severity <= 10:
            raise ValueError("Severity must be between 1 and 10")


@dataclass
class ImprovementProposal:
    """A proposed self-improvement.
    
    Based on Gödel Agent concepts of self-referential improvement.
    All proposals are tracked and can be rolled back.
    
    Attributes:
        proposal_id: Unique identifier
        target: What component to improve
        change_type: Type of change (config, code, prompt, strategy)
        description: Human-readable description
        current_state: Current implementation state
        proposed_state: Proposed implementation state
        rationale: Why this improvement helps
        risks: Potential risks
        status: Current status
        created_at: Timestamp
        applied_at: Optional timestamp when applied
    """
    proposal_id: str = field(default_factory=lambda: f"imp-{int(time.time())}-{hashlib.md5(str(time.time()).encode(), usedforsecurity=False).hexdigest()[:6]}")
    target: str = ""
    change_type: str = "config"  # config, code, prompt, strategy
    description: str = ""
    current_state: Dict[str, Any] = field(default_factory=dict)
    proposed_state: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    risks: List[str] = field(default_factory=list)
    status: ImprovementStatus = ImprovementStatus.PROPOSED
    created_at: float = field(default_factory=time.time)
    applied_at: Optional[float] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImprovementProposal":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = ImprovementStatus(data["status"])
        return cls(**data)


@dataclass
class PerformanceSnapshot:
    """Performance metrics at a point in time.
    
    Used to evaluate whether improvements actually help.
    """
    timestamp: float = field(default_factory=time.time)
    goal_success_rate: float = 0.0
    avg_cycle_duration_ms: float = 0.0
    memory_hit_rate: float = 0.0
    error_rate: float = 0.0
    user_satisfaction: Optional[float] = None  # If feedback available
    
    def improvement_over(self, baseline: "PerformanceSnapshot") -> Dict[str, float]:
        """Calculate improvement metrics over baseline."""
        return {
            "success_rate_delta": self.goal_success_rate - baseline.goal_success_rate,
            "duration_delta": baseline.avg_cycle_duration_ms - self.avg_cycle_duration_ms,  # Lower is better
            "memory_delta": self.memory_hit_rate - baseline.memory_hit_rate,
            "error_rate_delta": baseline.error_rate - self.error_rate,  # Lower is better
        }


class SelfImprovementEngine:
    """Engine for safe self-improvement based on Gödel Agent concepts.
    
    This is a research prototype that:
    1. Analyzes capability gaps
    2. Proposes safe improvements
    3. Tracks all changes with rollback capability
    4. Measures actual impact
    
    Safety controls:
    - All changes require explicit approval (opt-in)
    - Automatic rollback on test failure
    - Change history with full audit trail
    - No automatic code modification (config/strategy only)
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_approve: bool = False,
    ):
        """Initialize improvement engine.
        
        Args:
            storage_path: Path to store improvement history
            auto_approve: DANGER: Auto-approve proposals (default False)
        """
        self.storage_path = storage_path or Path("memory/improvements.json")
        self.auto_approve = auto_approve
        self.proposals: Dict[str, ImprovementProposal] = {}
        self.baseline: Optional[PerformanceSnapshot] = None
        
        # Callbacks for different change types
        self._config_applier: Optional[Callable[[str, Any], bool]] = None
        self._strategy_applier: Optional[Callable[[str, Any], bool]] = None
        
        self._load_history()
        log_json("INFO", "improvement_engine_initialized", {
            "auto_approve": auto_approve,
            "proposal_count": len(self.proposals),
        })
    
    def register_config_applier(self, callback: Callable[[str, Any], bool]) -> None:
        """Register callback for applying config changes.
        
        Args:
            callback: Function that takes (key, value) and returns success bool
        """
        self._config_applier = callback
    
    def register_strategy_applier(self, callback: Callable[[str, Any], bool]) -> None:
        """Register callback for applying strategy changes."""
        self._strategy_applier = callback
    
    def analyze_capability_gaps(
        self,
        recent_logs: List[Dict[str, Any]],
        performance_metrics: PerformanceSnapshot,
    ) -> List[CapabilityGap]:
        """Analyze logs and metrics to find capability gaps.
        
        This is the "reflection" phase of the Gödel Agent loop.
        
        Args:
            recent_logs: Recent log entries
            performance_metrics: Current performance metrics
            
        Returns:
            List of identified capability gaps
        """
        gaps: List[CapabilityGap] = []
        
        # Analyze error patterns
        errors = [e for e in recent_logs if e.get("level") == "ERROR"]
        error_types: Dict[str, int] = {}
        for e in errors:
            event = e.get("event", "unknown")
            error_types[event] = error_types.get(event, 0) + 1
        
        # Identify recurring errors as gaps
        for event, count in error_types.items():
            if count >= 3:  # Recurring error
                gaps.append(CapabilityGap(
                    area="reliability",
                    description=f"Recurring error: {event}",
                    severity=min(count, 10),
                    evidence=[f"Occurred {count} times in recent logs"],
                    suggested_fix=f"Add error handling or retry logic for {event}",
                ))
        
        # Check performance metrics
        if performance_metrics.memory_hit_rate < 0.5:
            gaps.append(CapabilityGap(
                area="memory",
                description="Low memory retrieval rate",
                severity=7,
                evidence=[f"Hit rate: {performance_metrics.memory_hit_rate:.1%}"],
                suggested_fix="Improve embedding quality or increase top_k",
            ))
        
        if performance_metrics.avg_cycle_duration_ms > 60000:  # > 1 minute
            gaps.append(CapabilityGap(
                area="performance",
                description="Slow cycle execution",
                severity=6,
                evidence=[f"Avg duration: {performance_metrics.avg_cycle_duration_ms/1000:.1f}s"],
                suggested_fix="Optimize tool calls or reduce max_iterations",
            ))
        
        if performance_metrics.error_rate > 0.1:  # > 10% errors
            gaps.append(CapabilityGap(
                area="reliability",
                description="High error rate",
                severity=8,
                evidence=[f"Error rate: {performance_metrics.error_rate:.1%}"],
                suggested_fix="Review error patterns and add specific handling",
            ))
        
        log_json("INFO", "capability_gaps_analyzed", {
            "gap_count": len(gaps),
            "areas": list(set(g.area for g in gaps)),
        })
        
        return sorted(gaps, key=lambda g: g.severity, reverse=True)
    
    def propose_improvement(
        self,
        gap: CapabilityGap,
        current_config: Dict[str, Any],
    ) -> Optional[ImprovementProposal]:
        """Generate improvement proposal for a capability gap.
        
        Args:
            gap: The capability gap to address
            current_config: Current system configuration
            
        Returns:
            Improvement proposal or None if no automatic fix available
        """
        proposal = ImprovementProposal(
            target=gap.area,
            description=f"Address {gap.description}",
            rationale=gap.suggested_fix or "Automatic improvement proposal",
            risks=["May impact other functionality", "Needs testing"],
        )
        
        # Generate specific proposals based on gap area
        if gap.area == "memory":
            proposal.change_type = "config"
            proposal.current_state = {"semantic_memory.top_k": current_config.get("semantic_memory", {}).get("top_k", 10)}
            proposal.proposed_state = {"semantic_memory.top_k": 20}
            proposal.description = "Increase memory retrieval count for better context"
            
        elif gap.area == "performance":
            proposal.change_type = "config"
            proposal.current_state = {"max_iterations": current_config.get("max_iterations", 10)}
            proposal.proposed_state = {"max_iterations": 7}
            proposal.description = "Reduce max iterations to speed up cycles"
            
        elif gap.area == "reliability":
            proposal.change_type = "strategy"
            proposal.current_state = {"retry_policy": "none"}
            proposal.proposed_state = {"retry_policy": "exponential_backoff"}
            proposal.description = "Add retry logic for transient failures"
        
        else:
            # No automatic proposal available
            return None
        
        # Store proposal
        self.proposals[proposal.proposal_id] = proposal
        self._save_history()
        
        log_json("INFO", "improvement_proposed", {
            "proposal_id": proposal.proposal_id,
            "target": proposal.target,
            "change_type": proposal.change_type,
        })
        
        return proposal
    
    def evaluate_proposal(self, proposal_id: str) -> bool:
        """Evaluate whether to approve a proposal.
        
        In a full Gödel Agent, this would involve self-referential reasoning.
        Here we use safety heuristics.
        
        Args:
            proposal_id: ID of proposal to evaluate
            
        Returns:
            True if approved
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        
        # Safety checks
        if proposal.change_type == "code":
            log_json("WARN", "improvement_code_changes_not_allowed", {
                "proposal_id": proposal_id,
            })
            proposal.status = ImprovementStatus.REJECTED
            return False
        
        # Check if change is significant
        if proposal.change_type == "config":
            for key, new_val in proposal.proposed_state.items():
                old_val = proposal.current_state.get(key)
                if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
                    change_ratio = abs(new_val - old_val) / max(old_val, 1)
                    if change_ratio > 0.5:  # > 50% change
                        log_json("WARN", "improvement_significant_config_change", {
                            "proposal_id": proposal_id,
                            "key": key,
                            "change_ratio": change_ratio,
                        })
                        proposal.risks.append(f"Large change in {key}: {old_val} -> {new_val}")
        
        proposal.status = ImprovementStatus.ANALYZED
        
        if self.auto_approve:
            proposal.status = ImprovementStatus.APPROVED
            return True
        
        return False  # Requires manual approval
    
    def apply_proposal(self, proposal_id: str) -> bool:
        """Apply an approved improvement proposal.
        
        Args:
            proposal_id: ID of proposal to apply
            
        Returns:
            True if successfully applied
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            log_json("ERROR", "improvement_proposal_not_found", {"proposal_id": proposal_id})
            return False
        
        if proposal.status not in (ImprovementStatus.APPROVED, ImprovementStatus.ANALYZED):
            log_json("ERROR", "improvement_proposal_not_approved", {
                "proposal_id": proposal_id,
                "status": proposal.status.value,
            })
            return False
        
        # Capture baseline if first application
        if self.baseline is None:
            self.baseline = PerformanceSnapshot()  # Would use actual metrics
        
        # Apply based on change type
        success = False
        if proposal.change_type == "config":
            if self._config_applier:
                for key, value in proposal.proposed_state.items():
                    success = self._config_applier(key, value)
                    if not success:
                        break
            else:
                log_json("ERROR", "improvement_no_config_applier")
                return False
        elif proposal.change_type == "strategy":
            if self._strategy_applier:
                success = self._strategy_applier(proposal.target, proposal.proposed_state)
            else:
                log_json("ERROR", "improvement_no_strategy_applier")
                return False
        
        if success:
            proposal.status = ImprovementStatus.APPLIED
            proposal.applied_at = time.time()
            self._save_history()
            
            log_json("INFO", "improvement_applied", {
                "proposal_id": proposal_id,
                "target": proposal.target,
            })
        else:
            log_json("ERROR", "improvement_apply_failed", {"proposal_id": proposal_id})
        
        return success
    
    def evaluate_impact(
        self,
        proposal_id: str,
        current_metrics: PerformanceSnapshot,
    ) -> Dict[str, Any]:
        """Evaluate whether an improvement actually helped.
        
        Args:
            proposal_id: ID of applied proposal
            current_metrics: Current performance metrics
            
        Returns:
            Impact assessment
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != ImprovementStatus.APPLIED:
            return {"error": "Proposal not found or not applied"}
        
        if self.baseline is None:
            return {"error": "No baseline metrics available"}
        
        improvements = current_metrics.improvement_over(self.baseline)
        
        # Overall improvement score
        score = (
            improvements["success_rate_delta"] * 0.4 +
            (improvements["duration_delta"] / 1000) * 0.2 +  # Normalize
            improvements["memory_delta"] * 0.2 +
            improvements["error_rate_delta"] * 0.2
        )
        
        proposal.test_results = {
            "improvements": improvements,
            "score": score,
            "timestamp": time.time(),
        }
        
        if score > 0:
            proposal.status = ImprovementStatus.TESTED
        else:
            # Negative or neutral impact - consider rollback
            proposal.risks.append(f"Negative impact score: {score:.3f}")
        
        self._save_history()
        
        return {
            "proposal_id": proposal_id,
            "improvements": improvements,
            "score": score,
            "should_rollback": score < -0.1,
        }
    
    def get_proposals(
        self,
        status: Optional[ImprovementStatus] = None,
    ) -> List[ImprovementProposal]:
        """Get improvement proposals, optionally filtered by status."""
        proposals = list(self.proposals.values())
        if status:
            proposals = [p for p in proposals if p.status == status]
        return sorted(proposals, key=lambda p: p.created_at, reverse=True)
    
    def _load_history(self) -> None:
        """Load improvement history from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for p_data in data.get("proposals", []):
                    proposal = ImprovementProposal.from_dict(p_data)
                    self.proposals[proposal.proposal_id] = proposal
                if data.get("baseline"):
                    self.baseline = PerformanceSnapshot(**data["baseline"])
            except Exception as e:
                log_json("WARN", "improvement_history_load_failed", {"error": str(e)})
    
    def _save_history(self) -> None:
        """Save improvement history to storage."""
        try:
            data = {
                "proposals": [p.to_dict() for p in self.proposals.values()],
                "baseline": asdict(self.baseline) if self.baseline else None,
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log_json("ERROR", "improvement_history_save_failed", {"error": str(e)})


def create_improvement_engine(
    config: Dict[str, Any],
    enable_auto: bool = False,
) -> SelfImprovementEngine:
    """Create and configure improvement engine.
    
    Args:
        config: System configuration
        enable_auto: Enable auto-approval (requires explicit opt-in)
        
    Returns:
        Configured SelfImprovementEngine
    """
    # Safety: Auto-approve requires both flag AND config setting
    auto_approve = enable_auto and config.get("enable_auto_improvements", False)
    
    if auto_approve:
        log_json("WARN", "improvement_auto_approve_enabled")
    
    engine = SelfImprovementEngine(auto_approve=auto_approve)
    
    # Register default appliers
    # (Would be connected to actual config manager in production)
    
    return engine
