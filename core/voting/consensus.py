"""Consensus analysis for voting results."""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json
from core.voting.engine import Vote


@dataclass
class ConsensusAnalysis:
    """Analysis of consensus level in votes."""

    consensus_level: float  # 0-1
    primary_cluster: str  # Most voted option
    secondary_clusters: List[str]  # Other significant options
    entropy: float  # Normalized Shannon entropy (0-1)
    is_unanimous: bool
    requires_discussion: bool
    vote_distribution: Dict[str, float] = field(default_factory=dict)
    confidence_variance: float = 0.0


class ConsensusAnalyzer:
    """Analyzes voting patterns for consensus detection."""

    def __init__(self):
        self.low_consensus_threshold = 0.5
        self.medium_consensus_threshold = 0.7
        self.high_consensus_threshold = 0.9

    def analyze(self, votes: Dict[str, Vote], winner: str) -> ConsensusAnalysis:
        """
        Analyze consensus level from votes.

        Args:
            votes: Dictionary of model votes
            winner: The winning option

        Returns:
            ConsensusAnalysis with metrics
        """
        if not votes:
            return ConsensusAnalysis(consensus_level=0.0, primary_cluster="", secondary_clusters=[], entropy=1.0, is_unanimous=False, requires_discussion=True, vote_distribution={}, confidence_variance=0.0)

        # Count votes per option
        counts = Counter(v.selection for v in votes.values())
        total_votes = len(votes)

        # Calculate vote distribution
        distribution = {opt: count / total_votes for opt, count in counts.items()}

        # Calculate entropy
        entropy = self._calculate_entropy(distribution.values())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Consensus is inverse of entropy (adjusted)
        # High entropy = low consensus, Low entropy = high consensus
        consensus_level = 1.0 - normalized_entropy

        # Adjust consensus by confidence variance
        confidence_variance = self._calculate_confidence_variance(votes)
        # Higher variance reduces consensus
        consensus_level *= 1.0 - confidence_variance * 0.3

        # Determine clusters
        sorted_options = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
        primary = sorted_options[0] if sorted_options else ""
        secondary = sorted_options[1:] if len(sorted_options) > 1 else []

        # Check unanimity
        is_unanimous = len(counts) == 1 and total_votes > 0

        # Determine if discussion is required
        requires_discussion = consensus_level < self.medium_consensus_threshold

        return ConsensusAnalysis(
            consensus_level=round(consensus_level, 3), primary_cluster=primary, secondary_clusters=secondary, entropy=round(normalized_entropy, 3), is_unanimous=is_unanimous, requires_discussion=requires_discussion, vote_distribution=distribution, confidence_variance=round(confidence_variance, 3)
        )

    def calculate_strength(self, analysis: ConsensusAnalysis, weighted_by_confidence: bool = True) -> str:
        """
        Calculate consensus strength label.

        Args:
            analysis: Consensus analysis
            weighted_by_confidence: Whether to weight by confidence

        Returns:
            Strength label ("strong", "moderate", "weak", "none")
        """
        level = analysis.consensus_level

        if level >= self.high_consensus_threshold:
            return "strong"
        elif level >= self.medium_consensus_threshold:
            return "moderate"
        elif level >= self.low_consensus_threshold:
            return "weak"
        else:
            return "none"

    def identify_dissenters(self, votes: Dict[str, Vote], winner: str, threshold: float = 0.5) -> List[str]:
        """
        Identify models that voted against the winner.

        Args:
            votes: Dictionary of votes
            winner: The winning option
            threshold: Minimum confidence to be considered a dissenter

        Returns:
            List of model IDs that dissented
        """
        dissenters = []

        for model_id, vote in votes.items():
            if vote.selection != winner and vote.confidence >= threshold:
                dissenters.append(model_id)

        return dissenters

    def analyze_confidence_distribution(self, votes: Dict[str, Vote]) -> Dict[str, Any]:
        """
        Analyze the distribution of confidence scores.

        Args:
            votes: Dictionary of votes

        Returns:
            Dictionary with confidence statistics
        """
        if not votes:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        confidences = [v.confidence for v in votes.values()]

        mean_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)

        # Calculate median
        sorted_conf = sorted(confidences)
        mid = len(sorted_conf) // 2
        if len(sorted_conf) % 2 == 0:
            median_conf = (sorted_conf[mid - 1] + sorted_conf[mid]) / 2
        else:
            median_conf = sorted_conf[mid]

        # Calculate standard deviation
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_conf = math.sqrt(variance)

        return {
            "mean": round(mean_conf, 3),
            "median": round(median_conf, 3),
            "std": round(std_conf, 3),
            "min": round(min_conf, 3),
            "max": round(max_conf, 3),
        }

    def recommend_action(self, analysis: ConsensusAnalysis, vote_count: int) -> Optional[str]:
        """
        Recommend next action based on consensus analysis.

        Args:
            analysis: Consensus analysis
            vote_count: Number of votes cast

        Returns:
            Recommended action string or None
        """
        strength = self.calculate_strength(analysis)

        if strength == "strong":
            return "Proceed with confidence. High consensus achieved."

        elif strength == "moderate":
            if len(analysis.secondary_clusters) > 0:
                return f"Moderate consensus on '{analysis.primary_cluster}'. Consider reviewing concerns about alternatives: {', '.join(analysis.secondary_clusters[:2])}"
            return "Moderate consensus achieved. Consider additional validation."

        elif strength == "weak":
            if analysis.entropy > 0.7:
                return "High disagreement detected. Consider breaking decision into smaller components or gathering additional information."
            return "Weak consensus. Recommend discussion to understand differing perspectives before proceeding."

        else:  # none
            return "No consensus achieved. Options are too contentious. Recommend: (1) Reframe the question, (2) Add more context, or (3) Seek expert input on specific分歧 areas."

    def _calculate_entropy(self, probabilities) -> float:
        """
        Calculate Shannon entropy.

        Args:
            probabilities: Iterable of probability values

        Returns:
            Shannon entropy in bits
        """
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _calculate_confidence_variance(self, votes: Dict[str, Vote]) -> float:
        """
        Calculate normalized variance in confidence scores.

        Args:
            votes: Dictionary of votes

        Returns:
            Normalized variance (0-1)
        """
        if len(votes) < 2:
            return 0.0

        confidences = [v.confidence for v in votes.values()]
        mean = sum(confidences) / len(confidences)

        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)

        # Normalize by max possible variance (0.25 for [0,1] range)
        return min(1.0, variance / 0.25)


class ConsensusBuilder:
    """Builds consensus through iterative refinement."""

    def __init__(self, analyzer: Optional[ConsensusAnalyzer] = None):
        """
        Initialize the consensus builder.

        Args:
            analyzer: Consensus analyzer instance
        """
        self.analyzer = analyzer or ConsensusAnalyzer()
        self.iteration_history: List[Dict] = []

    async def build_consensus(self, voting_engine, prompt: str, options: List[str], config, max_iterations: int = 3, target_consensus: float = 0.8) -> Dict[str, Any]:
        """
        Build consensus through iterative voting and refinement.

        Args:
            voting_engine: Voting engine instance
            prompt: Base decision prompt
            options: Available options
            config: Voting configuration
            max_iterations: Maximum iterations
            target_consensus: Target consensus level

        Returns:
            Dictionary with final result and iteration history
        """
        best_result = None
        best_consensus = 0.0

        for iteration in range(max_iterations):
            # Run vote
            result = await voting_engine.vote(prompt, options, config)

            # Analyze consensus
            analysis = self.analyzer.analyze(result.all_votes, result.winner)

            # Record iteration
            self.iteration_history.append({"iteration": iteration + 1, "winner": result.winner, "consensus": analysis.consensus_level, "entropy": analysis.entropy, "vote_count": len(result.all_votes)})

            # Track best result
            if analysis.consensus_level > best_consensus:
                best_consensus = analysis.consensus_level
                best_result = result

            # Check if target achieved
            if analysis.consensus_level >= target_consensus:
                log_json("INFO", "consensus_target_achieved", {"iteration": iteration + 1, "consensus": analysis.consensus_level})
                break

            # Refine for next iteration
            if iteration < max_iterations - 1:
                prompt, options = self._refine_for_next_iteration(result, analysis, prompt, options)

        return {"final_result": best_result, "iterations": len(self.iteration_history), "best_consensus": best_consensus, "history": self.iteration_history}

    def _refine_for_next_iteration(self, result, analysis: ConsensusAnalysis, prompt: str, options: List[str]) -> tuple:
        """
        Refine prompt and options for next iteration.

        Args:
            result: Voting result
            analysis: Consensus analysis
            prompt: Current prompt
            options: Current options

        Returns:
            Tuple of (new_prompt, new_options)
        """
        # Add context about disagreement
        dissenters = [model_id for model_id, vote in result.all_votes.items() if vote.selection != result.winner]

        if dissenters:
            refinement = f"\n\nNote: Previous vote had {len(dissenters)} dissenting model(s). Please reconsider your position carefully."
            new_prompt = prompt + refinement
        else:
            new_prompt = prompt

        # Could also filter to top options
        if len(options) > 2 and analysis.consensus_level < 0.5:
            # Keep only top 2 options
            top_options = [analysis.primary_cluster, analysis.secondary_clusters[0] if analysis.secondary_clusters else options[0]]
            new_options = [opt for opt in options if opt in top_options]
            if len(new_options) < 2:
                new_options = options[:2]
        else:
            new_options = options

        return new_prompt, new_options
