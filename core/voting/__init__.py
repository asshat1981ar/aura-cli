"""LLM Voting System for multi-model consensus decisions."""

from core.voting.engine import VotingEngine, VoteConfig, VoteResult, Vote, VotingStrategy
from core.voting.strategies import (
    SimpleMajorityStrategy,
    WeightedConfidenceStrategy,
    ExpertPanelStrategy,
    BordaCountStrategy,
)
from core.voting.consensus import ConsensusAnalyzer, ConsensusAnalysis

__all__ = [
    "VotingEngine",
    "VoteConfig",
    "VoteResult",
    "Vote",
    "VotingStrategy",
    "SimpleMajorityStrategy",
    "WeightedConfidenceStrategy",
    "ExpertPanelStrategy",
    "BordaCountStrategy",
    "ConsensusAnalyzer",
    "ConsensusAnalysis",
]
