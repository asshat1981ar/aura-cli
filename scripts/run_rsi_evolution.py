#!/usr/bin/env python3
"""
Run an RSI evolution audit across repeated orchestrator cycles.

This script drives ``LoopOrchestrator.run_cycle()`` for a configurable number
of cycles, records the resulting history, and emits an audit report using the
RSI verification helpers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aura_cli.cli_main import create_runtime
from core.logging_utils import log_json
from core.rsi_integration_verification import summarize_rsi_audit

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RSI evolution audit.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=50,
        help="Number of orchestrator cycles to execute.",
    )
    parser.add_argument(
        "--goal",
        default="evolve and improve the AURA system via recursive self-improvement",
        help="Goal to run for each audit cycle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the audit without applying orchestrator code changes.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path to write the final audit payload as JSON.",
    )
    parser.add_argument(
        "--sleep-on-error",
        type=float,
        default=5.0,
        help="Seconds to wait before continuing after a cycle failure.",
    )
    return parser


def _get_evolution_loop(orchestrator) -> Any:
    # First check for the explicit attribute (PRD-003 style)
    if hasattr(orchestrator, "evolution_loop"):
        return orchestrator.evolution_loop
    # Fallback for older versions
    for loop in getattr(orchestrator, "_improvement_loops", []):
        if type(loop).__name__ == "EvolutionLoop":
            return loop
    return None


def _call_count(obj: Any) -> int:
    value = getattr(obj, "call_count", None)
    return int(value) if isinstance(value, int) else 0


def _detect_trigger_cause(entry: Dict[str, Any], cycle_index: int, trigger_every: int) -> str | None:
    goal = str(entry.get("goal", "")).lower()
    phase_outputs = entry.get("phase_outputs", {}) if isinstance(entry, dict) else {}
    skill_context = phase_outputs.get("skill_context", {}) if isinstance(phase_outputs, dict) else {}
    context_blob = json.dumps(skill_context, sort_keys=True).lower()

    if "refactor hotspot" in goal or "structural_hotspot" in context_blob:
        return "hotspot"
    if trigger_every > 0 and cycle_index % trigger_every == 0:
        return "scheduled"
    return None


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cycles <= 0:
        parser.error("--cycles must be > 0")

    if load_dotenv:
        load_dotenv()

    os.environ.setdefault("AURA_SKIP_CHDIR", "1")
    os.environ.setdefault("DISABLE_SAFE_GETCWD", "1")

    try:
        runtime = create_runtime(project_root)
        orchestrator = runtime["orchestrator"]
    except Exception as exc:
        print(json.dumps({"status": "init_error", "error": str(exc)}, indent=2))
        return 1

    evolution_loop = _get_evolution_loop(orchestrator)
    notes: List[str] = []
    trigger_events: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    cycle_errors: List[Dict[str, Any]] = []
    evolution_runs = 0
    scheduled_triggers = 0
    hotspot_triggers = 0
    current_cause: str | None = None
    current_cycle_index = 0

    original_run = getattr(evolution_loop, "run", None) if evolution_loop is not None else None
    original_on_cycle_complete = (
        getattr(evolution_loop, "on_cycle_complete", None) if evolution_loop is not None else None
    )
    proposal_logger = (
        getattr(getattr(evolution_loop, "improvement_service", None), "log_proposal", None)
        if evolution_loop is not None
        else None
    )
    proposal_count_before = _call_count(proposal_logger)

    if evolution_loop is None:
        notes.append("No EvolutionLoop was attached to the runtime orchestrator.")
    else:
        trigger_every = int(getattr(evolution_loop, "TRIGGER_EVERY_N", 0) or 0)

        def _wrapped_on_cycle_complete(entry: Dict[str, Any]) -> None:
            nonlocal current_cause
            current_cause = _detect_trigger_cause(entry, current_cycle_index, trigger_every)
            return original_on_cycle_complete(entry)

        def _wrapped_run(goal: str):
            nonlocal evolution_runs, scheduled_triggers, hotspot_triggers
            evolution_runs += 1
            if current_cause == "hotspot":
                hotspot_triggers += 1
            elif current_cause == "scheduled":
                scheduled_triggers += 1
            trigger_events.append(
                {
                    "cycle": current_cycle_index,
                    "cause": current_cause or "unknown",
                    "goal": goal,
                }
            )
            if callable(original_run):
                return original_run(goal)
            return None

        evolution_loop.on_cycle_complete = _wrapped_on_cycle_complete
        evolution_loop.run = _wrapped_run

    try:
        for cycle_index in range(1, args.cycles + 1):
            current_cycle_index = cycle_index
            started_at = time.time()
            print(f">>> Cycle {cycle_index}/{args.cycles} starting...", flush=True)

            try:
                result = orchestrator.run_cycle(args.goal, dry_run=args.dry_run)
                entries.append(result)
                verification = result.get("phase_outputs", {}).get("verification", {})
                status = verification.get("status", "unknown") if isinstance(verification, dict) else "unknown"
                elapsed = round(time.time() - started_at, 2)
                print(
                    f"--- Cycle {cycle_index} complete in {elapsed:.2f}s | verification={status}",
                    flush=True,
                )
            except KeyboardInterrupt:
                notes.append(f"Audit interrupted by user at cycle {cycle_index}.")
                break
            except Exception as exc:
                elapsed = round(time.time() - started_at, 2)
                error_row = {
                    "cycle": cycle_index,
                    "elapsed_sec": elapsed,
                    "error": str(exc),
                }
                cycle_errors.append(error_row)
                log_json("ERROR", "rsi_audit_cycle_failed", details=error_row)
                print(f"!!! Cycle {cycle_index} failed after {elapsed:.2f}s: {exc}", flush=True)
                current_cause = None
                if args.sleep_on_error > 0:
                    time.sleep(args.sleep_on_error)
    finally:
        if evolution_loop is not None:
            evolution_loop.run = original_run
            evolution_loop.on_cycle_complete = original_on_cycle_complete

    proposal_count_after = _call_count(proposal_logger)
    if cycle_errors:
        notes.append(f"{len(cycle_errors)} cycle(s) raised exceptions during the audit run.")

    report = summarize_rsi_audit(
        entries,
        target_cycles=args.cycles,
        evolution_runs=evolution_runs,
        scheduled_triggers=scheduled_triggers,
        hotspot_triggers=hotspot_triggers,
        proposal_count=max(0, proposal_count_after - proposal_count_before),
        notes=notes,
    )

    payload = {
        "status": "ok" if not cycle_errors else "completed_with_errors",
        "goal": args.goal,
        "cycles_requested": args.cycles,
        "dry_run": args.dry_run,
        "provider_status": runtime.get("provider_status"),
        "trigger_events": trigger_events,
        "cycle_errors": cycle_errors,
        "report": report.as_dict(),
    }

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2), flush=True)
    return 0 if not cycle_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
