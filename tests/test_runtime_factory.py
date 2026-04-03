"""Tests for aura_cli.runtime_factory internals."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


def test_runtime_factory_module_imports() -> None:
    from aura_cli.runtime_factory import RuntimeFactory

    assert RuntimeFactory is not None


def test_weakness_remediator_loop_fires_every_5() -> None:
    from aura_cli.runtime_factory import _WeaknessRemediatorLoop

    remediator = MagicMock()
    brain = MagicMock()
    goal_queue = MagicMock()
    loop = _WeaknessRemediatorLoop(remediator, brain, goal_queue)

    for _ in range(10):
        loop.on_cycle_complete({})

    assert remediator.run.call_count == 2


def test_weakness_remediator_loop_does_not_fire_at_cycle_3() -> None:
    from aura_cli.runtime_factory import _WeaknessRemediatorLoop

    remediator = MagicMock()
    brain = MagicMock()
    goal_queue = MagicMock()
    loop = _WeaknessRemediatorLoop(remediator, brain, goal_queue)

    for _ in range(3):
        loop.on_cycle_complete({})

    remediator.run.assert_not_called()


def test_convergence_escape_loop_fires_every_cycle() -> None:
    from aura_cli.runtime_factory import _ConvergenceEscapeLoop

    escape = MagicMock()
    loop = _ConvergenceEscapeLoop(escape)

    entry = {"phase_outputs": {"context": {"goal": "Build feature X"}}}
    loop.on_cycle_complete(entry)

    escape.check_and_escape.assert_called_once_with("Build feature X", entry)


def test_convergence_escape_skips_entry_without_goal() -> None:
    from aura_cli.runtime_factory import _ConvergenceEscapeLoop

    escape = MagicMock()
    loop = _ConvergenceEscapeLoop(escape)

    entry = {"phase_outputs": {"context": {}}}
    loop.on_cycle_complete(entry)

    escape.check_and_escape.assert_not_called()


def test_build_runtime_config_merges_overrides() -> None:
    from aura_cli.runtime_factory import _build_runtime_config

    result = _build_runtime_config({"max_cycles": 99})

    assert result["max_cycles"] == 99


def test_build_runtime_config_returns_dict() -> None:
    from aura_cli.runtime_factory import _build_runtime_config

    result = _build_runtime_config()

    assert isinstance(result, dict)


def test_resolve_runtime_paths_returns_all_keys(tmp_path: Path) -> None:
    from aura_cli.runtime_factory import _resolve_runtime_paths

    result = _resolve_runtime_paths(tmp_path)

    assert "goal_queue_path" in result
