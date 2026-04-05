"""Adversarial Agent for red-team critique and stress testing."""

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json


class TargetType(Enum):
    """Types of targets for adversarial critique."""
    CODE = "code"
    PLAN = "plan"
    DECISION = "decision"
    DESIGN = "design"
    ARCHITECTURE = "architecture"
    API = "api"
    TEST = "test"
    DOCUMENTATION = "documentation"


class AdversarialStrategy(Enum):
    """Available adversarial strategies."""
    DEVILS_ADVOCATE = "devils_advocate"
    WORST_CASE = "worst_case"
    ASSUMPTION_CHALLENGE = "assumption_challenge"
    EDGE_CASE_HUNTER = "edge_case_hunter"
    SECURITY_MINDSET = "security_mindset"
    SCALABILITY_FOCUS = "scalability_focus"


@dataclass
class Finding:
    """A single finding from adversarial critique."""
    category: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    evidence: str
    recommendation: str
    confidence: float = 0.8


@dataclass
class StrategyResult:
    """Result from running an adversarial strategy."""
    strategy: str
    findings: List[Finding]
    confidence: float = 0.0
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class AdversarialCritique:
    """Structured adversarial critique output."""
    critique_id: str
    target_summary: str
    target_type: TargetType
    overall_assessment: str
    confidence: float
    findings: List[Finding]
    suggested_tests: List[str]
    counterarguments: List[str]
    strategy_results: Dict[str, StrategyResult]
    learning_notes: List[str]
    timestamp: float
    risk_score: float = 0.0
    validation_status: Optional[str] = None


class AdversarialAgent:
    """
    Red-team agent that provides adversarial critique and learns from outcomes.
    
    Uses multiple adversarial strategies:
    - Devil's Advocate: Argue against the proposed solution
    - Worst-Case Analysis: Identify catastrophic failure modes
    - Assumption Challenging: Question underlying assumptions
    - Edge Case Hunter: Find boundary conditions that break the solution
    - Security Mindset: Identify security vulnerabilities
    """
    
    capabilities = [
        "adversarial_critique",
        "red_team",
        "stress_test",
        "devils_advocate",
        "assumption_challenging",
        "edge_case_analysis",
        "security_review"
    ]
    
    def __init__(self, brain=None, model=None):
        """
        Initialize the adversarial agent.
        
        Args:
            brain: Optional memory/brain instance
            model: Optional model adapter
        """
        self.brain = brain
        self.model = model
        
        # Initialize learning components
        from agents.adversarial.learning import AdversarialLearner
        from agents.adversarial.outcome_tracker import CritiqueOutcomeTracker
        
        self.learner = AdversarialLearner()
        self.outcome_tracker = CritiqueOutcomeTracker()
        
        # Initialize strategies
        from agents.adversarial.strategies import (
            DevilsAdvocateStrategy,
            EdgeCaseHunterStrategy,
            AssumptionChallengeStrategy,
            WorstCaseStrategy,
            SecurityMindsetStrategy,
            ScalabilityFocusStrategy,
        )
        
        self.strategies = {
            AdversarialStrategy.DEVILS_ADVOCATE: DevilsAdvocateStrategy(model),
            AdversarialStrategy.WORST_CASE: WorstCaseStrategy(model),
            AdversarialStrategy.ASSUMPTION_CHALLENGE: AssumptionChallengeStrategy(model),
            AdversarialStrategy.EDGE_CASE_HUNTER: EdgeCaseHunterStrategy(model),
            AdversarialStrategy.SECURITY_MINDSET: SecurityMindsetStrategy(model),
            AdversarialStrategy.SCALABILITY_FOCUS: ScalabilityFocusStrategy(model),
        }
        
        self._active_critiques: Dict[str, AdversarialCritique] = {}
    
    async def critique(
        self,
        target: str,
        target_type: TargetType,
        context: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[AdversarialStrategy]] = None,
        intensity: float = 0.8
    ) -> AdversarialCritique:
        """
        Provide adversarial critique of a target.
        
        Args:
            target: The code/plan/decision to critique
            target_type: Type of target
            context: Additional context
            strategies: Which adversarial strategies to use
            intensity: How aggressive to be (0-1)
            
        Returns:
            Structured adversarial critique
        """
        critique_id = str(uuid.uuid4())[:8]
        context = context or {}
        
        # Use recommended strategies if none specified
        if strategies is None:
            strategies = self.learner.recommend_strategies(target_type)
            if not strategies:
                strategies = [
                    AdversarialStrategy.DEVILS_ADVOCATE,
                    AdversarialStrategy.ASSUMPTION_CHALLENGE,
                    AdversarialStrategy.EDGE_CASE_HUNTER,
                ]
        
        log_json("INFO", "adversarial_critique_started", {
            "critique_id": critique_id,
            "target_type": target_type.value,
            "strategies": [s.value for s in strategies],
            "intensity": intensity
        })
        
        # Run selected strategies
        strategy_results = await self._run_strategies(
            target, target_type, context, strategies, intensity
        )
        
        # Synthesize results
        synthesized = self._synthesize_critiques(strategy_results)
        
        # Calculate confidence based on finding overlap
        confidence = self._calculate_confidence(strategy_results)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(synthesized.findings)
        
        # Get learning notes
        learning_notes = self.learner.get_relevant_notes(target_type)
        
        critique = AdversarialCritique(
            critique_id=critique_id,
            target_summary=target[:100] if isinstance(target, str) else str(target)[:100],
            target_type=target_type,
            overall_assessment=synthesized.assessment,
            confidence=confidence,
            findings=synthesized.findings,
            suggested_tests=synthesized.suggested_tests,
            counterarguments=synthesized.counterarguments,
            strategy_results=strategy_results,
            learning_notes=learning_notes,
            timestamp=time.time(),
            risk_score=risk_score
        )
        
        # Track for learning
        self._active_critiques[critique_id] = critique
        await self.outcome_tracker.start_tracking(critique_id, critique)
        
        log_json("INFO", "adversarial_critique_completed", {
            "critique_id": critique_id,
            "findings_count": len(critique.findings),
            "risk_score": risk_score,
            "confidence": confidence
        })
        
        return critique
    
    async def critique_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None
    ) -> AdversarialCritique:
        """Convenience method for code critique."""
        ctx = context or {}
        ctx["language"] = language
        ctx["code_length"] = len(code)
        
        return await self.critique(
            target=code,
            target_type=TargetType.CODE,
            context=ctx,
            strategies=[
                AdversarialStrategy.EDGE_CASE_HUNTER,
                AdversarialStrategy.SECURITY_MINDSET,
                AdversarialStrategy.ASSUMPTION_CHALLENGE,
            ]
        )
    
    async def critique_plan(
        self,
        plan: str,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AdversarialCritique:
        """Convenience method for plan critique."""
        ctx = context or {}
        ctx["goal"] = goal
        
        return await self.critique(
            target=plan,
            target_type=TargetType.PLAN,
            context=ctx,
            strategies=[
                AdversarialStrategy.DEVILS_ADVOCATE,
                AdversarialStrategy.WORST_CASE,
                AdversarialStrategy.ASSUMPTION_CHALLENGE,
            ]
        )
    
    async def learn_from_outcome(
        self,
        critique_id: str,
        was_validated: bool,
        actual_severity: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """
        Learn from whether the critique was validated.
        
        Args:
            critique_id: ID of the critique
            was_validated: Whether the critique findings were validated
            actual_severity: Actual severity of issues found
            notes: Additional notes
        """
        critique = self._active_critiques.get(critique_id)
        if not critique:
            log_json("WARN", "critique_not_found_for_learning", {
                "critique_id": critique_id
            })
            return
        
        # Update learning for each strategy
        for strategy_name, result in critique.strategy_results.items():
            self.learner.record_feedback(
                strategy=strategy_name,
                target_type=critique.target_type,
                was_validated=was_validated,
                severity=actual_severity or "medium",
                notes=notes or ""
            )
        
        # Update critique status
        critique.validation_status = "validated" if was_validated else "false_positive"
        
        # Adapt strategies based on effectiveness
        await self._adapt_strategies()
        
        log_json("INFO", "adversarial_learning_updated", {
            "critique_id": critique_id,
            "was_validated": was_validated,
            "strategies_updated": len(critique.strategy_results)
        })
    
    async def _run_strategies(
        self,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        strategies: List[AdversarialStrategy],
        intensity: float
    ) -> Dict[str, StrategyResult]:
        """Run all selected adversarial strategies."""
        tasks = []
        for strategy_enum in strategies:
            strategy = self.strategies.get(strategy_enum)
            if strategy:
                task = self._run_strategy_with_timeout(
                    strategy, target, target_type, context, intensity
                )
                tasks.append((strategy_enum.value, task))
        
        results = {}
        for strategy_name, task in tasks:
            try:
                result = await task
                results[strategy_name] = result
            except asyncio.TimeoutError:
                results[strategy_name] = StrategyResult(
                    strategy=strategy_name,
                    findings=[],
                    error="Timeout",
                    confidence=0.0
                )
            except Exception as e:
                log_json("ERROR", "strategy_execution_failed", {
                    "strategy": strategy_name,
                    "error": str(e)
                })
                results[strategy_name] = StrategyResult(
                    strategy=strategy_name,
                    findings=[],
                    error=str(e),
                    confidence=0.0
                )
        
        return results
    
    async def _run_strategy_with_timeout(
        self,
        strategy,
        target: str,
        target_type: TargetType,
        context: Dict[str, Any],
        intensity: float,
        timeout: float = 30.0
    ) -> StrategyResult:
        """Run a strategy with timeout."""
        return await asyncio.wait_for(
            strategy.execute(target, target_type, context, intensity),
            timeout=timeout
        )
    
    def _synthesize_critiques(
        self,
        strategy_results: Dict[str, StrategyResult]
    ) -> Any:
        """Synthesize results from multiple strategies."""
        all_findings = []
        all_tests = []
        all_counterarguments = []
        
        for result in strategy_results.values():
            all_findings.extend(result.findings)
            
            # Extract suggested tests from findings
            for finding in result.findings:
                if "test" in finding.recommendation.lower():
                    all_tests.append(finding.recommendation)
            
            # Add counterarguments for devil's advocate
            if "devil" in result.strategy.lower():
                for finding in result.findings:
                    all_counterarguments.append(finding.description)
        
        # Determine overall assessment
        critical_count = sum(1 for f in all_findings if f.severity == "critical")
        high_count = sum(1 for f in all_findings if f.severity == "high")
        
        if critical_count > 0:
            assessment = f"Critical issues identified ({critical_count} critical, {high_count} high)"
        elif high_count > 0:
            assessment = f"Significant concerns ({high_count} high severity findings)"
        elif all_findings:
            assessment = f"Minor issues identified ({len(all_findings)} findings)"
        else:
            assessment = "No significant issues found"
        
        # Create synthesized result
        @dataclass
        class SynthesizedResult:
            assessment: str
            findings: List[Finding]
            suggested_tests: List[str]
            counterarguments: List[str]
        
        return SynthesizedResult(
            assessment=assessment,
            findings=all_findings,
            suggested_tests=all_tests,
            counterarguments=all_counterarguments
        )
    
    def _calculate_confidence(
        self,
        strategy_results: Dict[str, StrategyResult]
    ) -> float:
        """Calculate overall confidence based on strategy results."""
        if not strategy_results:
            return 0.0
        
        confidences = [r.confidence for r in strategy_results.values() if r.confidence > 0]
        
        if not confidences:
            return 0.5
        
        # Average confidence weighted by number of findings
        weighted_sum = sum(
            r.confidence * len(r.findings) 
            for r in strategy_results.values()
        )
        total_findings = sum(len(r.findings) for r in strategy_results.values())
        
        if total_findings == 0:
            return sum(confidences) / len(confidences)
        
        return min(1.0, weighted_sum / total_findings)
    
    def _calculate_risk_score(self, findings: List[Finding]) -> float:
        """Calculate overall risk score from findings."""
        if not findings:
            return 0.0
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }
        
        total_risk = sum(
            severity_weights.get(f.severity, 0.0) * f.confidence
            for f in findings
        )
        
        # Normalize by number of findings (more findings = higher max risk)
        max_possible = len(findings) * 1.0
        return min(1.0, total_risk / max_possible) if max_possible > 0 else 0.0
    
    async def _adapt_strategies(self):
        """Adapt strategies based on learning feedback."""
        # Get recommendations from learner
        recommendations = self.learner.get_strategy_recommendations()
        
        log_json("INFO", "strategies_adapted", {
            "recommendations": recommendations
        })
    
    def get_strategy_performance(
        self,
        target_type: Optional[TargetType] = None
    ) -> Dict[str, Any]:
        """Get performance statistics for strategies."""
        return self.learner.get_performance_stats(target_type)
    
    def get_active_critiques(self) -> List[str]:
        """Get list of active critique IDs."""
        return list(self._active_critiques.keys())
