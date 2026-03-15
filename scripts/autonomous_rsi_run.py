#!/usr/bin/env python3
"""
Run the canonical RSI verification harness for 10 cycles.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aura_cli.cli_main import create_runtime, default_agents
from core.recursive_improvement import RecursiveImprovementService
from core.rsi_integration_verification import (
    DEFAULT_RSI_VERIFICATION_GOAL,
    build_evolution_loop,
    ensure_rsi_environment,
    load_optional_dotenv,
    run_rsi_verification,
)

def main():
    load_optional_dotenv()
    ensure_rsi_environment()

    max_cycles = 10
    print(f">>> Starting {max_cycles} cycles of autonomous RSI verification...")

    runtime = create_runtime(project_root)
    loop = build_evolution_loop(
        runtime,
        project_root,
        improvement_service=RecursiveImprovementService(),
        default_agents_factory=default_agents,
    )

    report = run_rsi_verification(
        loop,
        goal=DEFAULT_RSI_VERIFICATION_GOAL,
        max_cycles=max_cycles,
        on_cycle_complete=_print_cycle_result,
    )

    print(
        "\n>>> RSI verification finished. "
        f"Cycles: {len(report['cycles'])} | Proposals: {report['proposal_count']}"
    )


def _print_cycle_result(cycle_number: int, result: dict, proposals: list[dict]) -> None:
    validation = result.get("validation", {}) if isinstance(result, dict) else {}
    decision = validation.get("decision", "UNKNOWN")
    applied = bool(result.get("mutation_applied")) if isinstance(result, dict) else False
    status = "applied" if applied else "rejected"
    print(
        f"--- Cycle {cycle_number} complete: {status} "
        f"(decision={decision}, proposals={len(proposals)})"
    )

if __name__ == "__main__":
    main()
