"""Deprecated RSI prototype scaffold.

This module used to hold a second self-improvement planning surface. The
canonical RSI path now lives in:

- ``core/recursive_improvement.py``
- ``core/fitness.py``
- ``conductor/tracks/recursive_self_improvement_20260301/``
"""

from __future__ import annotations


CANONICAL_RSI_PATHS = (
    "core/recursive_improvement.py",
    "core/fitness.py",
    "conductor/tracks/recursive_self_improvement_20260301/",
)


def evolve_aura_system() -> None:
    raise RuntimeError(
        "core.evolution_plan is retired. Use core/recursive_improvement.py, "
        "core/fitness.py, and conductor/tracks/recursive_self_improvement_20260301/ "
        "as the canonical RSI path."
    )
