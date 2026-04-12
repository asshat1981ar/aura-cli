import math
from typing import Dict


class FitnessFunction:
    """
    A multi-objective fitness function to evaluate AURA's meta-performance.
    Weighs success rate, efficiency (tokens), and code quality (complexity delta).
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {"success_rate": 0.5, "token_efficiency": 0.3, "complexity_penalty": 0.2}

    def calculate(self, metrics: dict) -> float:
        """
        Calculates a fitness score between 0.0 and 1.0.

        Args:
            metrics: A dictionary containing:
                - success_rate (float): 0.0 to 1.0
                - tokens_per_action (int): actual tokens used
                - baseline_tokens (int): expected/budgeted tokens
                - complexity_delta (float): change in complexity (0.0 to 1.0)
        """
        sr = metrics.get("success_rate", 0.0)

        # Token efficiency: exponential decay as usage exceeds baseline
        tokens = metrics.get("tokens_per_action", 10000)
        baseline = metrics.get("baseline_tokens", 5000)
        efficiency = math.exp(-max(0, tokens - baseline) / baseline)

        # Quality: 1.0 minus the complexity delta (clamped)
        complexity_delta = metrics.get("complexity_delta", 0.0)
        quality = 1.0 - max(0.0, min(1.0, complexity_delta))

        score = self.weights["success_rate"] * sr + self.weights["token_efficiency"] * efficiency + self.weights["complexity_penalty"] * quality
        return round(score, 4)
