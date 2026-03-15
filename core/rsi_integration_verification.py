"""Shared RSI runtime helpers for CLI and verification scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


DEFAULT_EVOLVE_GOAL = "evolve and improve the AURA system"
DEFAULT_RSI_VERIFICATION_GOAL = (
    "evolve and improve the AURA system via recursive self-improvement"
)


def load_optional_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def ensure_rsi_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    target.setdefault("AGENT_API_TOKEN", "rsi-dev-token")
    target.setdefault("AGENT_API_ENABLE_RUN", "1")
    return target


def build_evolution_loop(
    runtime: dict[str, Any],
    project_root: Path,
    *,
    improvement_service: Any = None,
    default_agents_factory: Callable[[Any, Any], dict[str, Any]] | None = None,
    git_tools_cls: Any = None,
):
    from agents.mutator import MutatorAgent
    from core.evolution_loop import EvolutionLoop
    from core.git_tools import GitTools
    from core.vector_store import VectorStore
    from memory.brain import Brain

    brain = runtime.get("brain") or Brain()
    model = runtime.get("model_adapter")
    planner = runtime.get("planner") or runtime.get("plan")
    coder = runtime.get("act") or runtime.get("coder")
    critic = runtime.get("critique") or runtime.get("critic")

    if planner is None or coder is None or critic is None:
        if default_agents_factory is None:
            from aura_cli.cli_main import default_agents as default_agents_factory
        agents = default_agents_factory(brain, model)
        planner = planner or agents.get("plan")
        coder = coder or agents.get("act")
        critic = critic or agents.get("critique")

    if planner is None or coder is None or critic is None:
        raise ValueError("RSI requires planner, coder, and critic agents.")
    if model is None:
        raise ValueError("RSI requires a model adapter in runtime['model_adapter'].")

    if git_tools_cls is None:
        git_tools_cls = GitTools

    git_tools = runtime.get("git_tools") or git_tools_cls(repo_path=str(project_root))
    mutator = runtime.get("mutator") or MutatorAgent(project_root)
    vector_store = runtime.get("vector_store") or VectorStore(model, brain)
    improvement_service = improvement_service or runtime.get("recursive_improvement_service")

    return EvolutionLoop(
        planner,
        coder,
        critic,
        brain,
        vector_store,
        git_tools,
        mutator,
        improvement_service=improvement_service,
    )


def build_rsi_cycle_entry(goal: str, cycle_number: int, result: dict[str, Any]) -> dict[str, Any]:
    validation = result.get("validation", {}) if isinstance(result, dict) else {}
    if not isinstance(validation, dict):
        validation = {}
    mutation_applied = bool(result.get("mutation_applied")) if isinstance(result, dict) else False
    verification_status = "pass" if mutation_applied else "fail"

    return {
        "cycle_id": f"rsi_cycle_{cycle_number}",
        "goal": goal,
        "verification_status": verification_status,
        "phase_outputs": {
            "verification": {
                "status": verification_status,
                "decision": validation.get("decision"),
                "confidence_score": validation.get("confidence_score"),
            },
            "retry_count": 0,
            "rsi": {
                "validation": validation,
                "mutation_applied": mutation_applied,
                "tasks": result.get("tasks", []) if isinstance(result, dict) else [],
                "hypothesis": result.get("hypothesis") if isinstance(result, dict) else None,
            },
        },
    }


def run_rsi_verification(
    loop,
    *,
    goal: str = DEFAULT_RSI_VERIFICATION_GOAL,
    max_cycles: int = 1,
    on_cycle_complete: Callable[[int, dict[str, Any], list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    if max_cycles < 1:
        raise ValueError("RSI verification requires at least one cycle.")

    cycles = []
    proposal_count = 0

    for cycle_number in range(1, max_cycles + 1):
        result = loop.run(goal)
        cycle_entry = build_rsi_cycle_entry(goal, cycle_number, result)
        proposals = list(loop.on_cycle_complete(cycle_entry) or [])
        proposal_count += len(proposals)
        if on_cycle_complete is not None:
            on_cycle_complete(cycle_number, result, proposals)
        cycles.append(
            {
                "cycle_number": cycle_number,
                "cycle_entry": cycle_entry,
                "result": result,
                "proposals": proposals,
            }
        )

    return {
        "goal": goal,
        "max_cycles": max_cycles,
        "proposal_count": proposal_count,
        "cycles": cycles,
    }
