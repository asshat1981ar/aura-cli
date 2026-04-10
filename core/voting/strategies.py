"""Voting strategies for aggregating LLM votes."""

from collections import defaultdict
from typing import Callable, Dict, List, Optional

from core.logging_utils import log_json
from core.voting.engine import AggregationResult, Vote


class VotingStrategy:
    """Base class for voting strategies."""

    def aggregate(self, votes: Dict[str, Vote], options: List[str], domain: Optional[str] = None) -> AggregationResult:
        """
        Aggregate votes and determine winner.

        Args:
            votes: Dictionary of model_id -> Vote
            options: List of available options
            domain: Optional domain context

        Returns:
            AggregationResult with winner and confidence
        """
        raise NotImplementedError


class SimpleMajorityStrategy(VotingStrategy):
    """Simple majority voting - most votes wins."""

    def aggregate(self, votes: Dict[str, Vote], options: List[str], domain: Optional[str] = None) -> AggregationResult:
        """Aggregate votes by simple majority."""
        if not votes:
            return AggregationResult(winner=options[0] if options else "", confidence=0.0, breakdown={opt: 0.0 for opt in options}, all_rankings={})

        # Count votes
        counts = defaultdict(float)
        for vote in votes.values():
            counts[vote.selection] += 1.0

        # Find winner
        total_votes = sum(counts.values())
        winner = max(counts.keys(), key=lambda k: counts[k])
        winner_count = counts[winner]

        # Check for tie
        max_count = max(counts.values())
        tied_options = [opt for opt, count in counts.items() if count == max_count]
        tie_detected = len(tied_options) > 1

        # Calculate confidence
        confidence = winner_count / total_votes if total_votes > 0 else 0.0

        # Build breakdown
        breakdown = {opt: counts.get(opt, 0.0) / total_votes if total_votes > 0 else 0.0 for opt in options}

        # Build rankings (all equal in simple majority)
        all_rankings = {model_id: [vote.selection] for model_id, vote in votes.items()}

        return AggregationResult(winner=winner, confidence=confidence, breakdown=breakdown, all_rankings=all_rankings, tie_detected=tie_detected)


class WeightedConfidenceStrategy(VotingStrategy):
    """Votes weighted by model confidence and historical accuracy."""

    def __init__(self, weight_provider: Optional[Callable[[], Dict[str, float]]] = None):
        """
        Initialize with optional weight provider.

        Args:
            weight_provider: Function that returns model weights
        """
        self.weight_provider = weight_provider

    def aggregate(self, votes: Dict[str, Vote], options: List[str], domain: Optional[str] = None) -> AggregationResult:
        """Aggregate votes with confidence weighting."""
        if not votes:
            return AggregationResult(winner=options[0] if options else "", confidence=0.0, breakdown={opt: 0.0 for opt in options}, all_rankings={})

        # Get model weights
        model_weights = {}
        if self.weight_provider:
            try:
                model_weights = self.weight_provider()
            except Exception as e:
                log_json("WARN", "weight_provider_failed", {"error": str(e)})

        # Calculate weighted scores
        scores = defaultdict(float)
        all_rankings = {}

        for model_id, vote in votes.items():
            # Calculate weight
            base_weight = model_weights.get(model_id, 1.0)
            weight = base_weight * vote.confidence

            # Add to option score
            scores[vote.selection] += weight

            # Record ranking
            all_rankings[model_id] = [vote.selection]

        # Find winner
        total_weight = sum(scores.values())
        winner = max(scores.keys(), key=lambda k: scores[k])
        winner_score = scores[winner]

        # Check for tie
        max_score = max(scores.values())
        tied_options = [opt for opt, score in scores.items() if abs(score - max_score) < 0.001]
        tie_detected = len(tied_options) > 1

        # Calculate confidence
        confidence = winner_score / total_weight if total_weight > 0 else 0.0

        # Normalize breakdown
        breakdown = {opt: scores.get(opt, 0.0) / total_weight if total_weight > 0 else 0.0 for opt in options}

        return AggregationResult(winner=winner, confidence=confidence, breakdown=breakdown, all_rankings=all_rankings, tie_detected=tie_detected)


class BordaCountStrategy(VotingStrategy):
    """Borda count voting - ranks all options, points for position."""

    def aggregate(self, votes: Dict[str, Vote], options: List[str], domain: Optional[str] = None) -> AggregationResult:
        """Aggregate votes using Borda count method."""
        if not votes:
            return AggregationResult(winner=options[0] if options else "", confidence=0.0, breakdown={opt: 0.0 for opt in options}, all_rankings={})

        # For each model, rank all options
        # In simple case where we only have top choice,
        # we'll use confidence as proxy for ranking

        borda_scores = defaultdict(float)
        all_rankings = {}

        num_options = len(options)

        for model_id, vote in votes.items():
            # Create ranking: selected option first, others by decreasing confidence
            ranking = [vote.selection]

            # Add other options in arbitrary order (could be extended with full rankings)
            for opt in options:
                if opt != vote.selection:
                    ranking.append(opt)

            all_rankings[model_id] = ranking

            # Assign Borda points
            for position, opt in enumerate(ranking):
                points = (num_options - position - 1) * vote.confidence
                borda_scores[opt] += points

        # Find winner
        total_points = sum(borda_scores.values())
        winner = max(borda_scores.keys(), key=lambda k: borda_scores[k])
        winner_score = borda_scores[winner]

        # Check for tie
        max_score = max(borda_scores.values())
        tied_options = [opt for opt, score in borda_scores.items() if abs(score - max_score) < 0.001]
        tie_detected = len(tied_options) > 1

        # Calculate confidence
        confidence = winner_score / total_points if total_points > 0 else 0.0

        # Normalize breakdown
        breakdown = {opt: borda_scores.get(opt, 0.0) / total_points if total_points > 0 else 0.0 for opt in options}

        return AggregationResult(winner=winner, confidence=confidence, breakdown=breakdown, all_rankings=all_rankings, tie_detected=tie_detected)


class ExpertPanelStrategy(VotingStrategy):
    """Different models act as experts in different domains."""

    # Define expertise domains for different model types
    EXPERTISE_DOMAINS = {
        "claude": ["architecture", "design", "refactoring", "code_review"],
        "claude-sonnet": ["architecture", "design", "refactoring", "code_review"],
        "claude-opus": ["complex_reasoning", "architecture", "design_patterns"],
        "gpt-4": ["code_generation", "debugging", "testing", "implementation"],
        "gpt-4o": ["code_generation", "debugging", "testing", "implementation"],
        "gemini": ["documentation", "analysis", "review", "summarization"],
        "gemini-pro": ["documentation", "analysis", "review", "summarization"],
        "codex": ["implementation", "coding_patterns", "syntax"],
        "local": ["privacy_sensitive", "offline", "low_latency"],
    }

    def aggregate(self, votes: Dict[str, Vote], options: List[str], domain: Optional[str] = None) -> AggregationResult:
        """Aggregate votes weighted by domain expertise."""
        if not votes:
            return AggregationResult(winner=options[0] if options else "", confidence=0.0, breakdown={opt: 0.0 for opt in options}, all_rankings={})

        # Calculate expert-weighted scores
        scores = defaultdict(float)
        all_rankings = {}

        for model_id, vote in votes.items():
            # Determine expertise weight
            expertise_weight = self._get_expertise_weight(model_id, domain)

            # Calculate final weight
            weight = expertise_weight * vote.confidence

            scores[vote.selection] += weight
            all_rankings[model_id] = [vote.selection]

        # Find winner
        total_weight = sum(scores.values())

        if total_weight == 0:
            # Fall back to simple majority
            return SimpleMajorityStrategy().aggregate(votes, options, domain)

        winner = max(scores.keys(), key=lambda k: scores[k])
        winner_score = scores[winner]

        # Check for tie
        max_score = max(scores.values())
        tied_options = [opt for opt, score in scores.items() if abs(score - max_score) < 0.001]
        tie_detected = len(tied_options) > 1

        # Calculate confidence
        confidence = winner_score / total_weight

        # Normalize breakdown
        breakdown = {opt: scores.get(opt, 0.0) / total_weight for opt in options}

        return AggregationResult(winner=winner, confidence=confidence, breakdown=breakdown, all_rankings=all_rankings, tie_detected=tie_detected)

    def _get_expertise_weight(self, model_id: str, domain: Optional[str]) -> float:
        """Get expertise weight for a model in a domain."""
        if not domain:
            return 1.0

        domain_lower = domain.lower()

        # Check if model has explicit expertise
        for model_pattern, domains in self.EXPERTISE_DOMAINS.items():
            if model_pattern in model_id.lower():
                if domain_lower in [d.lower() for d in domains]:
                    return 2.0  # Expert in this domain
                return 0.5  # Not expert, reduced weight

        # Unknown model, neutral weight
        return 1.0


class CondorcetStrategy(VotingStrategy):
    """Condorcet method - pairwise comparison winner."""

    def aggregate(self, votes: Dict[str, Vote], options: List[str], domain: Optional[str] = None) -> AggregationResult:
        """
        Aggregate votes using Condorcet pairwise comparisons.

        Note: This is a simplified implementation. Full Condorcet
        would require complete rankings from all voters.
        """
        if not votes or len(options) < 2:
            return SimpleMajorityStrategy().aggregate(votes, options, domain)

        # Build pairwise comparison matrix
        # Matrix[i][j] = number of voters who prefer option i over option j
        pairwise = defaultdict(lambda: defaultdict(float))

        for vote in votes.values():
            selected = vote.selection
            # Selected beats all others in pairwise comparison
            for other in options:
                if other != selected:
                    pairwise[selected][other] += vote.confidence
                    pairwise[other][selected] += 0  # Implicit loss

        # Find Condorcet winner (beats all others in pairwise)
        condorcet_winner = None
        scores = defaultdict(float)

        for opt_a in options:
            beats_all = True
            for opt_b in options:
                if opt_a != opt_b:
                    if pairwise[opt_a][opt_b] <= pairwise[opt_b][opt_a]:
                        beats_all = False
                        break
                    scores[opt_a] += pairwise[opt_a][opt_b]

            if beats_all:
                condorcet_winner = opt_a
                break

        # If no Condorcet winner, use Copeland method (most pairwise wins)
        if condorcet_winner is None:
            copeland_scores = {}
            for opt in options:
                wins = sum(1 for other in options if other != opt and pairwise[opt][other] > pairwise[other][opt])
                copeland_scores[opt] = wins

            condorcet_winner = max(copeland_scores.keys(), key=lambda k: copeland_scores[k])
            scores = copeland_scores

        # Calculate confidence based on margin of victory
        winner_score = scores[condorcet_winner]
        total_score = sum(scores.values())

        # Check for tie
        max_score = max(scores.values())
        tied_options = [opt for opt, score in scores.items() if score == max_score]
        tie_detected = len(tied_options) > 1

        confidence = winner_score / total_score if total_score > 0 else 0.0

        # Normalize breakdown
        breakdown = {opt: scores.get(opt, 0.0) / total_score if total_score > 0 else 0.0 for opt in options}

        all_rankings = {model_id: [vote.selection] for model_id, vote in votes.items()}

        return AggregationResult(winner=condorcet_winner, confidence=confidence, breakdown=breakdown, all_rankings=all_rankings, tie_detected=tie_detected)


class EnsembleStrategy(VotingStrategy):
    """Combine multiple voting strategies."""

    def __init__(self, strategies: Optional[List[VotingStrategy]] = None):
        """
        Initialize with list of strategies to ensemble.

        Args:
            strategies: List of voting strategies
        """
        self.strategies = strategies or [
            SimpleMajorityStrategy(),
            WeightedConfidenceStrategy(),
            BordaCountStrategy(),
        ]

    def aggregate(self, votes: Dict[str, Vote], options: List[str], domain: Optional[str] = None) -> AggregationResult:
        """Aggregate using ensemble of strategies."""
        # Get results from all strategies
        results = []
        for strategy in self.strategies:
            try:
                result = strategy.aggregate(votes, options, domain)
                results.append(result)
            except Exception as e:
                log_json("WARN", "strategy_failed", {"strategy": type(strategy).__name__, "error": str(e)})

        if not results:
            return SimpleMajorityStrategy().aggregate(votes, options, domain)

        # Combine results (average confidence scores)
        combined_scores = defaultdict(float)

        for result in results:
            for opt, score in result.breakdown.items():
                combined_scores[opt] += score

        # Average
        num_strategies = len(results)
        for opt in combined_scores:
            combined_scores[opt] /= num_strategies

        # Find winner
        winner = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        confidence = combined_scores[winner]

        # Check agreement
        winners = [r.winner for r in results]
        agreement = len(set(winners)) == 1

        all_rankings = results[0].all_rankings if results else {}

        return AggregationResult(winner=winner, confidence=confidence, breakdown=dict(combined_scores), all_rankings=all_rankings, tie_detected=not agreement)
