"""Voting engine for multi-LLM consensus decisions."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.logging_utils import log_json

if TYPE_CHECKING:
    from core.voting.consensus import ConsensusAnalysis


class VotingStrategy(Enum):
    """Available voting strategies."""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED = "weighted"
    BORDA_COUNT = "borda_count"
    CONDORCET = "condorcet"
    EXPERT_PANEL = "expert_panel"


@dataclass
class Vote:
    """A single vote from a model."""
    model_id: str
    selection: str
    confidence: float = 0.8
    reasoning: str = ""
    alternatives: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoteConfig:
    """Configuration for a voting round."""
    models: List[str] = field(default_factory=list)
    strategy: VotingStrategy = VotingStrategy.WEIGHTED
    min_consensus: float = 0.6
    max_disagreement: float = 0.3
    timeout_seconds: float = 60.0
    require_unanimity: bool = False
    break_ties_with: Optional[str] = None
    max_retries: int = 2
    domain: Optional[str] = None


@dataclass
class DisagreementAnalysis:
    """Analysis of disagreement in votes."""
    disagreement_level: float  # 0-1
    primary_cluster: str
    secondary_clusters: List[str]
    key_differences: List[str]
    suggested_resolution: Optional[str]


@dataclass
class VoteResult:
    """Result from a voting round."""
    vote_id: str
    winner: str
    winner_confidence: float
    all_votes: Dict[str, Vote]
    consensus_level: float
    disagreement_analysis: Optional[DisagreementAnalysis]
    confidence_breakdown: Dict[str, float]
    strategy_used: VotingStrategy
    runtime_seconds: float
    tie_broken: bool = False
    resolution_method: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AggregationResult:
    """Result from vote aggregation."""
    winner: str
    confidence: float
    breakdown: Dict[str, float]
    all_rankings: Dict[str, List[str]]
    tie_detected: bool = False


class VotingEngine:
    """Orchestrates multi-LLM voting on decisions."""
    
    def __init__(self, model_adapter=None):
        """
        Initialize the voting engine.
        
        Args:
            model_adapter: Model adapter for calling LLMs
        """
        self.model_adapter = model_adapter
        self.history: List[VoteResult] = []
        self._model_weights: Dict[str, float] = {}
        self._accuracy_history: Dict[str, List[bool]] = {}
        
        # Initialize strategies
        from core.voting.strategies import (
            SimpleMajorityStrategy,
            WeightedConfidenceStrategy,
            BordaCountStrategy,
            ExpertPanelStrategy,
        )
        
        self.strategies = {
            VotingStrategy.SIMPLE_MAJORITY: SimpleMajorityStrategy(),
            VotingStrategy.WEIGHTED: WeightedConfidenceStrategy(self._get_model_weights),
            VotingStrategy.BORDA_COUNT: BordaCountStrategy(),
            VotingStrategy.EXPERT_PANEL: ExpertPanelStrategy(),
        }
    
    async def vote(
        self,
        prompt: str,
        options: List[str],
        config: Optional[VoteConfig] = None
    ) -> VoteResult:
        """
        Run a voting round with multiple models.
        
        Args:
            prompt: The decision prompt
            options: Available options to vote on
            config: Voting configuration
            
        Returns:
            VoteResult with winner and analysis
        """
        config = config or VoteConfig()
        start_time = time.time()
        vote_id = str(uuid.uuid4())[:8]
        
        log_json("INFO", "voting_started", {
            "vote_id": vote_id,
            "models": config.models,
            "options_count": len(options),
            "strategy": config.strategy.value
        })
        
        if not config.models:
            return VoteResult(
                vote_id=vote_id,
                winner="",
                winner_confidence=0.0,
                all_votes={},
                consensus_level=0.0,
                disagreement_analysis=None,
                confidence_breakdown={},
                strategy_used=config.strategy,
                runtime_seconds=0.0
            )
        
        # 1. Collect votes from all models
        votes = await self._collect_votes(prompt, options, config)
        
        if not votes:
            return VoteResult(
                vote_id=vote_id,
                winner=options[0] if options else "",
                winner_confidence=0.0,
                all_votes={},
                consensus_level=0.0,
                disagreement_analysis=None,
                confidence_breakdown={opt: 0.0 for opt in options},
                strategy_used=config.strategy,
                runtime_seconds=time.time() - start_time
            )
        
        # 2. Apply voting strategy
        strategy = self.strategies[config.strategy]
        aggregation = strategy.aggregate(votes, options, config.domain)
        
        # 3. Handle ties
        tie_broken = False
        resolution_method = None
        
        if aggregation.tie_detected and config.break_ties_with:
            winner = await self._break_tie(
                votes, options, config.break_ties_with
            )
            aggregation.winner = winner
            tie_broken = True
            resolution_method = f"tie_breaker:{config.break_ties_with}"
        
        # 4. Analyze consensus
        from core.voting.consensus import ConsensusAnalyzer
        consensus = ConsensusAnalyzer().analyze(votes, aggregation.winner)
        
        # 5. Handle low consensus
        if consensus.consensus_level < config.min_consensus and not config.require_unanimity:
            # Could trigger additional discussion round
            log_json("INFO", "low_consensus_detected", {
                "vote_id": vote_id,
                "consensus": consensus.consensus_level,
                "threshold": config.min_consensus
            })
        
        # 6. Build disagreement analysis if needed
        disagreement = None
        if consensus.consensus_level < 0.8:
            disagreement = self._analyze_disagreement(votes, consensus)
        
        runtime = time.time() - start_time
        
        result = VoteResult(
            vote_id=vote_id,
            winner=aggregation.winner,
            winner_confidence=aggregation.confidence,
            all_votes=votes,
            consensus_level=consensus.consensus_level,
            disagreement_analysis=disagreement,
            confidence_breakdown=aggregation.breakdown,
            strategy_used=config.strategy,
            runtime_seconds=runtime,
            tie_broken=tie_broken,
            resolution_method=resolution_method
        )
        
        self.history.append(result)
        
        log_json("INFO", "voting_completed", {
            "vote_id": vote_id,
            "winner": result.winner,
            "confidence": result.winner_confidence,
            "consensus": result.consensus_level,
            "runtime": runtime
        })
        
        return result
    
    async def vote_on_plan(
        self,
        plans: List[Dict[str, Any]],
        evaluation_criteria: List[str],
        config: Optional[VoteConfig] = None
    ) -> VoteResult:
        """
        Vote on multiple plan candidates.
        
        Args:
            plans: List of plan dictionaries
            evaluation_criteria: Criteria for evaluating plans
            config: Voting configuration
            
        Returns:
            VoteResult with winning plan
        """
        # Build prompt for plan evaluation
        plan_descriptions = []
        for i, plan in enumerate(plans):
            desc = f"Plan {i+1}: {plan.get('name', 'Unnamed')}\n"
            desc += f"Description: {plan.get('description', 'No description')}\n"
            desc += f"Steps: {len(plan.get('steps', []))}\n"
            plan_descriptions.append(desc)
        
        prompt = f"""Evaluate these plans based on the following criteria:
{chr(10).join(f"- {c}" for c in evaluation_criteria)}

{chr(10).join(plan_descriptions)}

Select the best plan (respond with "Plan 1", "Plan 2", etc.) and explain why.
"""
        
        options = [f"Plan {i+1}" for i in range(len(plans))]
        
        return await self.vote(prompt, options, config)
    
    async def vote_on_code_review(
        self,
        code: str,
        context: str,
        review_aspects: List[str],
        config: Optional[VoteConfig] = None
    ) -> VoteResult:
        """
        Vote on code review decisions.
        
        Args:
            code: Code to review
            context: Context about the code
            review_aspects: Aspects to review (e.g., ["security", "performance"])
            config: Voting configuration
            
        Returns:
            VoteResult with review decision
        """
        prompt = f"""Review this code for the following aspects:
{chr(10).join(f"- {a}" for a in review_aspects)}

Context: {context}

Code:
```
{code}
```

Should this code be approved, approved with changes, or rejected?
"""
        
        options = ["approve", "approve_with_changes", "reject"]
        
        return await self.vote(prompt, options, config)
    
    async def _collect_votes(
        self,
        prompt: str,
        options: List[str],
        config: VoteConfig
    ) -> Dict[str, Vote]:
        """Collect votes from all configured models."""
        tasks = []
        for model_id in config.models:
            task = self._get_model_vote_with_retry(
                model_id, prompt, options, config
            )
            tasks.append((model_id, task))
        
        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
        
        votes = {}
        for (model_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                log_json("WARN", "model_vote_failed", {
                    "model": model_id,
                    "error": str(result)
                })
            else:
                votes[model_id] = result
        
        return votes
    
    async def _get_model_vote_with_retry(
        self,
        model_id: str,
        prompt: str,
        options: List[str],
        config: VoteConfig
    ) -> Vote:
        """Get vote from a model with retry logic."""
        last_error = None
        
        for attempt in range(config.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._get_model_vote(model_id, prompt, options),
                    timeout=config.timeout_seconds / (config.max_retries + 1)
                )
            except Exception as e:
                last_error = e
                if attempt < config.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        
        raise last_error or RuntimeError(f"Failed to get vote from {model_id}")
    
    async def _get_model_vote(
        self,
        model_id: str,
        prompt: str,
        options: List[str]
    ) -> Vote:
        """Get a vote from a specific model."""
        # Build voting prompt
        voting_prompt = f"""{prompt}

You must select one of the following options:
{chr(10).join(f"- {opt}" for opt in options)}

Respond with your selection and confidence (0.0-1.0) in this format:
SELECTION: <option>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""
        
        # Call model (this is a placeholder - actual implementation would use model_adapter)
        if self.model_adapter:
            response = await self._call_model(model_id, voting_prompt)
        else:
            # Simulated response for testing
            response = self._simulate_vote(model_id, options)
        
        # Parse response
        return self._parse_vote_response(model_id, response, options)
    
    async def _call_model(self, model_id: str, prompt: str) -> str:
        """Call a model to get response."""
        # This would integrate with the actual model adapter
        # For now, return a simulated response
        return self._simulate_vote(model_id, ["option_a", "option_b"])
    
    def _simulate_vote(self, model_id: str, options: List[str]) -> str:
        """Simulate a vote for testing."""
        import random
        selection = random.choice(options)
        confidence = random.uniform(0.6, 0.95)
        return f"SELECTION: {selection}\nCONFIDENCE: {confidence:.2f}\nREASONING: Simulated vote"
    
    def _parse_vote_response(
        self,
        model_id: str,
        response: str,
        options: List[str]
    ) -> Vote:
        """Parse vote from model response."""
        lines = response.strip().split('\n')
        
        selection = None
        confidence = 0.8
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('SELECTION:'):
                selection = line.split(':', 1)[1].strip().lower()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 0.8
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        # Validate selection
        if selection not in [opt.lower() for opt in options]:
            # Try to find closest match
            for opt in options:
                if opt.lower() in selection or selection in opt.lower():
                    selection = opt.lower()
                    break
            else:
                selection = options[0].lower() if options else ""
        
        # Normalize selection to original case
        selection = next(
            (opt for opt in options if opt.lower() == selection),
            options[0] if options else ""
        )
        
        return Vote(
            model_id=model_id,
            selection=selection,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning
        )
    
    async def _break_tie(
        self,
        votes: Dict[str, Vote],
        options: List[str],
        tie_breaker_model: str
    ) -> str:
        """Break a tie using a designated model."""
        log_json("INFO", "breaking_tie", {"model": tie_breaker_model})
        
        # Get tie-breaking vote
        try:
            tie_breaker_vote = await self._get_model_vote(
                tie_breaker_model,
                "Break the tie between equally voted options.",
                options
            )
            return tie_breaker_vote.selection
        except Exception as e:
            log_json("WARN", "tie_break_failed", {"error": str(e)})
            # Fall back to first option
            return options[0] if options else ""
    
    def _analyze_disagreement(
        self,
        votes: Dict[str, Vote],
        consensus: "ConsensusAnalysis"
    ) -> DisagreementAnalysis:
        """Analyze disagreement in votes."""
        # Find key differences between clusters
        primary_votes = [v for v in votes.values() if v.selection == consensus.primary_cluster]
        secondary_selections = consensus.secondary_clusters
        
        key_differences = []
        
        if len(secondary_selections) > 0:
            secondary_votes = [v for v in votes.values() if v.selection == secondary_selections[0]]
            
            # Compare reasoning
            primary_reasoning = [v.reasoning for v in primary_votes if v.reasoning]
            secondary_reasoning = [v.reasoning for v in secondary_votes if v.reasoning]
            
            if primary_reasoning and secondary_reasoning:
                key_differences.append(
                    f"Different evaluation criteria: "
                    f"'{consensus.primary_cluster}' supporters focus on different aspects "
                    f"than '{secondary_selections[0]}' supporters"
                )
        
        # Suggest resolution
        suggestion = None
        if consensus.consensus_level < 0.5:
            suggestion = "Consider gathering more information or refining the options"
        elif consensus.consensus_level < 0.7:
            suggestion = "Consider a hybrid approach combining elements of top choices"
        
        return DisagreementAnalysis(
            disagreement_level=1.0 - consensus.consensus_level,
            primary_cluster=consensus.primary_cluster,
            secondary_clusters=consensus.secondary_clusters,
            key_differences=key_differences,
            suggested_resolution=suggestion
        )
    
    def record_outcome(
        self,
        vote_id: str,
        actual_best_option: str,
        was_successful: bool
    ):
        """Record actual outcome for learning."""
        # Find vote result
        result = next((r for r in self.history if r.vote_id == vote_id), None)
        if not result:
            return
        
        # Update accuracy history for each model
        for model_id, vote in result.all_votes.items():
            if model_id not in self._accuracy_history:
                self._accuracy_history[model_id] = []
            
            was_correct = vote.selection == actual_best_option
            self._accuracy_history[model_id].append(was_correct)
            
            # Keep only recent history
            self._accuracy_history[model_id] = self._accuracy_history[model_id][-100:]
        
        log_json("INFO", "vote_outcome_recorded", {
            "vote_id": vote_id,
            "predicted": result.winner,
            "actual": actual_best_option,
            "match": result.winner == actual_best_option,
            "was_successful": was_successful
        })
    
    def _get_model_weights(self) -> Dict[str, float]:
        """Get current model weights based on accuracy."""
        weights = {}
        for model_id, history in self._accuracy_history.items():
            if history:
                accuracy = sum(history) / len(history)
                weights[model_id] = 0.5 + accuracy  # Weight between 0.5 and 1.5
            else:
                weights[model_id] = 1.0
        return weights
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for each model."""
        performance = {}
        
        for model_id, history in self._accuracy_history.items():
            if history:
                accuracy = sum(history) / len(history)
                performance[model_id] = {
                    "total_votes": len(history),
                    "correct_votes": sum(history),
                    "accuracy": accuracy,
                    "weight": 0.5 + accuracy
                }
        
        return performance
    
    def get_voting_history(
        self,
        limit: int = 100,
        min_consensus: Optional[float] = None
    ) -> List[VoteResult]:
        """Get voting history with optional filtering."""
        results = self.history
        
        if min_consensus is not None:
            results = [r for r in results if r.consensus_level >= min_consensus]
        
        return results[-limit:]
