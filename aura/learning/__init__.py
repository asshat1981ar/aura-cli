"""Autonomous Learning Loop for AURA.

Enables AURA to learn from past successes and failures to improve
future performance.
"""

from .feedback import FeedbackCollector, ExecutionOutcome
from .patterns import PatternRecognizer, SuccessPattern
from .optimization import PromptOptimizer

__all__ = [
    "FeedbackCollector",
    "ExecutionOutcome",
    "PatternRecognizer",
    "SuccessPattern",
    "PromptOptimizer",
]
