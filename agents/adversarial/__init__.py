"""Adversarial Agent for red-team critique and stress testing."""

from agents.adversarial.agent import AdversarialAgent, AdversarialCritique, AdversarialStrategy, Finding, StrategyResult, TargetType
from agents.adversarial.strategies import (
    DevilsAdvocateStrategy,
    EdgeCaseHunterStrategy,
    AssumptionChallengeStrategy,
    WorstCaseStrategy,
    SecurityMindsetStrategy,
    ScalabilityFocusStrategy,
)
from agents.adversarial.learning import (
    AdversarialLearner,
    StrategyEffectiveness,
)
from agents.adversarial.outcome_tracker import CritiqueOutcomeTracker

__all__ = [
    "AdversarialAgent",
    "AdversarialCritique",
    "AdversarialStrategy",
    "Finding",
    "StrategyResult",
    "TargetType",
    "DevilsAdvocateStrategy",
    "EdgeCaseHunterStrategy",
    "AssumptionChallengeStrategy",
    "WorstCaseStrategy",
    "SecurityMindsetStrategy",
    "ScalabilityFocusStrategy",
    "AdversarialLearner",
    "StrategyEffectiveness",
    "CritiqueOutcomeTracker",
]
