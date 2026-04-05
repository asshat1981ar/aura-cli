"""Learning module for adversarial strategies."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.adversarial.agent import TargetType
from core.logging_utils import log_json


@dataclass
class StrategyEffectiveness:
    """Track effectiveness of an adversarial strategy."""
    strategy: str
    target_type: str
    total_critiques: int = 0
    validated_findings: int = 0
    false_positives: int = 0
    average_severity: str = "medium"
    success_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class SuccessfulPattern:
    """A pattern that led to successful critique."""
    strategy: str
    target_type: TargetType
    notes: str
    timestamp: float


@dataclass
class FailurePattern:
    """A pattern that led to false positive."""
    strategy: str
    target_type: TargetType
    notes: str
    timestamp: float


class AdversarialLearner:
    """Learns from critique outcomes to improve adversarial capabilities."""
    
    def __init__(self):
        """Initialize the learner."""
        self.effectiveness: Dict[str, StrategyEffectiveness] = {}
        self.pattern_memory: List[SuccessfulPattern] = []
        self.failure_patterns: List[FailurePattern] = []
        self._max_patterns = 100
        
    def record_feedback(
        self,
        strategy: str,
        target_type: TargetType,
        was_validated: bool,
        severity: str,
        notes: str
    ):
        """
        Record feedback about a critique's accuracy.
        
        Args:
            strategy: Strategy name
            target_type: Type of target critiqued
            was_validated: Whether the critique was validated
            severity: Severity of actual issue
            notes: Additional notes
        """
        key = f"{strategy}:{target_type.value}"
        
        # Initialize or get existing stats
        if key not in self.effectiveness:
            self.effectiveness[key] = StrategyEffectiveness(
                strategy=strategy,
                target_type=target_type.value
            )
        
        stats = self.effectiveness[key]
        stats.total_critiques += 1
        
        if was_validated:
            stats.validated_findings += 1
            self.pattern_memory.append(SuccessfulPattern(
                strategy=strategy,
                target_type=target_type,
                notes=notes,
                timestamp=time.time()
            ))
        else:
            stats.false_positives += 1
            self.failure_patterns.append(FailurePattern(
                strategy=strategy,
                target_type=target_type,
                notes=notes,
                timestamp=time.time()
            ))
        
        # Update success rate
        stats.success_rate = (
            stats.validated_findings / stats.total_critiques
            if stats.total_critiques > 0 else 0.0
        )
        
        # Update average severity
        severity_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        # This is simplified - would track actual distribution in production
        
        stats.last_updated = time.time()
        
        # Trim old patterns
        self._trim_old_patterns()
        
        log_json("DEBUG", "adversarial_feedback_recorded", {
            "strategy": strategy,
            "target_type": target_type.value,
            "was_validated": was_validated,
            "success_rate": stats.success_rate
        })
    
    def get_relevant_notes(
        self,
        target_type: TargetType,
        limit: int = 10
    ) -> List[str]:
        """
        Get learning notes relevant to a target type.
        
        Args:
            target_type: Type of target
            limit: Maximum number of notes
            
        Returns:
            List of relevant notes
        """
        # Find successful patterns for similar targets
        cutoff_time = time.time() - (30 * 24 * 3600)  # 30 days
        
        relevant = [
            p for p in self.pattern_memory
            if p.target_type == target_type
            and p.timestamp > cutoff_time
        ]
        
        # Sort by recency
        relevant.sort(key=lambda p: p.timestamp, reverse=True)
        
        return [p.notes for p in relevant[:limit]]
    
    def recommend_strategies(
        self,
        target_type: TargetType,
        min_success_rate: float = 0.5,
        limit: int = 5
    ) -> List[str]:
        """
        Recommend strategies based on historical effectiveness.
        
        Args:
            target_type: Type of target
            min_success_rate: Minimum success rate
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended strategy names
        """
        recommendations = []
        
        for key, stats in self.effectiveness.items():
            if stats.target_type == target_type.value:
                if stats.success_rate >= min_success_rate:
                    recommendations.append((
                        stats.strategy,
                        stats.success_rate,
                        stats.total_critiques
                    ))
        
        # Sort by success rate, then by sample size
        recommendations.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return [r[0] for r in recommendations[:limit]]
    
    def get_strategy_recommendations(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy recommendations.
        
        Returns:
            Dictionary with recommendations
        """
        if not self.effectiveness:
            return {
                "recommended": [],
                "avoid": [],
                "experimental": []
            }
        
        # Categorize strategies
        recommended = []
        avoid = []
        experimental = []
        
        for key, stats in self.effectiveness.items():
            if stats.total_critiques < 5:
                experimental.append({
                    "strategy": stats.strategy,
                    "target_type": stats.target_type,
                    "samples": stats.total_critiques
                })
            elif stats.success_rate >= 0.7:
                recommended.append({
                    "strategy": stats.strategy,
                    "target_type": stats.target_type,
                    "success_rate": stats.success_rate
                })
            elif stats.success_rate < 0.3:
                avoid.append({
                    "strategy": stats.strategy,
                    "target_type": stats.target_type,
                    "success_rate": stats.success_rate
                })
        
        return {
            "recommended": recommended,
            "avoid": avoid,
            "experimental": experimental
        }
    
    def get_performance_stats(
        self,
        target_type: Optional[TargetType] = None
    ) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            target_type: Optional filter by target type
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_strategies": len(self.effectiveness),
            "total_patterns": len(self.pattern_memory),
            "total_failures": len(self.failure_patterns)
        }
        
        if target_type:
            # Filter by target type
            filtered = {
                k: v for k, v in self.effectiveness.items()
                if v.target_type == target_type.value
            }
        else:
            filtered = self.effectiveness
        
        if filtered:
            avg_success = sum(e.success_rate for e in filtered.values()) / len(filtered)
            stats["average_success_rate"] = round(avg_success, 3)
            
            # Best and worst performers
            sorted_by_success = sorted(
                filtered.values(),
                key=lambda x: x.success_rate,
                reverse=True
            )
            
            stats["best_performer"] = {
                "strategy": sorted_by_success[0].strategy,
                "success_rate": sorted_by_success[0].success_rate
            } if sorted_by_success else None
            
            stats["worst_performer"] = {
                "strategy": sorted_by_success[-1].strategy,
                "success_rate": sorted_by_success[-1].success_rate
            } if sorted_by_success else None
        
        return stats
    
    def adapt_prompts(
        self,
        strategy: str,
        recent_failures: List[FailurePattern]
    ) -> str:
        """
        Adapt prompts based on failure patterns.
        
        Args:
            strategy: Strategy name
            recent_failures: Recent failure patterns
            
        Returns:
            Adapted prompt guidance
        """
        if not recent_failures:
            return self._get_base_guidance(strategy)
        
        # Analyze common failure modes
        common_issues = self._analyze_failure_patterns(recent_failures)
        
        # Build adapted guidance
        guidance = self._get_base_guidance(strategy)
        
        if "overlooked_context" in common_issues:
            guidance += "\n\nImportant: Always check the full context including related code and dependencies."
        
        if "false_positives" in common_issues:
            guidance += "\n\nBe more conservative. Only flag issues you are highly confident about."
        
        if "vague_findings" in common_issues:
            guidance += "\n\nProvide specific, actionable findings with clear evidence."
        
        return guidance
    
    def _get_base_guidance(self, strategy: str) -> str:
        """Get base prompt guidance for a strategy."""
        guidance_map = {
            "devils_advocate": "Argue against the solution constructively. Find real weaknesses.",
            "edge_case_hunter": "Find specific edge cases that could break the solution.",
            "assumption_challenge": "Identify and challenge hidden assumptions.",
            "worst_case": "Identify catastrophic failure modes.",
            "security_mindset": "Focus on security vulnerabilities and risks.",
            "scalability_focus": "Identify performance and scalability limitations.",
        }
        return guidance_map.get(strategy, "Provide thorough adversarial critique.")
    
    def _analyze_failure_patterns(
        self,
        failures: List[FailurePattern]
    ) -> List[str]:
        """Analyze common failure patterns."""
        issues = []
        
        # Simple heuristic analysis
        notes_combined = " ".join(f.notes.lower() for f in failures)
        
        if "context" in notes_combined:
            issues.append("overlooked_context")
        
        if "false" in notes_combined or "not valid" in notes_combined:
            issues.append("false_positives")
        
        if "vague" in notes_combined or "unclear" in notes_combined:
            issues.append("vague_findings")
        
        return issues
    
    def _trim_old_patterns(self):
        """Trim old patterns to manage memory."""
        cutoff = time.time() - (90 * 24 * 3600)  # 90 days
        
        self.pattern_memory = [
            p for p in self.pattern_memory
            if p.timestamp > cutoff
        ]
        
        self.failure_patterns = [
            p for p in self.failure_patterns
            if p.timestamp > cutoff
        ]
        
        # Also enforce max size
        if len(self.pattern_memory) > self._max_patterns:
            self.pattern_memory = self.pattern_memory[-self._max_patterns:]
        
        if len(self.failure_patterns) > self._max_patterns:
            self.failure_patterns = self.failure_patterns[-self._max_patterns:]
