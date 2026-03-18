from __future__ import annotations

import pytest

from core.evolution_plan import CANONICAL_RSI_PATHS, evolve_aura_system


def test_evolution_plan_redirects_to_canonical_rsi_path():
    assert "core/recursive_improvement.py" in CANONICAL_RSI_PATHS

    with pytest.raises(RuntimeError) as excinfo:
        evolve_aura_system()

    assert "core.evolution_plan is retired" in str(excinfo.value)
    assert "core/recursive_improvement.py" in str(excinfo.value)
