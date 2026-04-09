"""Decomposition strategies for goal breakdown.

Provides multiple strategies for breaking down complex goals:
- Sequential: Simple step-by-step decomposition
- Parallel: Independent subtasks
- TreeOfThoughts: Advanced reasoning with multiple paths
"""

from core.decomposition.base import DecompositionStrategy, DecompositionResult
from core.decomposition.sequential import SequentialDecomposition
from core.decomposition.parallel import ParallelDecomposition
from core.decomposition.tree_of_thoughts import TreeOfThoughtsDecomposition

__all__ = [
    "DecompositionStrategy",
    "DecompositionResult",
    "SequentialDecomposition",
    "ParallelDecomposition",
    "TreeOfThoughtsDecomposition",
]
