#!/usr/bin/env python
"""
Run synthetic goals through the AURA loop and emit JSON results.
Usage:
  python scripts/benchmark_loop.py --goals "add docstring" "refactor utils" --cycles 3 --json out.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aura_cli.cli_main import create_runtime


def run_goal(orchestrator, goal: str, max_cycles: int, dry_run: bool):
    start = time.time()
    result = orchestrator.run_loop(goal, max_cycles=max_cycles, dry_run=dry_run)
    elapsed = time.time() - start
    stop = result.get("stop_reason")
    failures = []
    try:
        history = result.get("history", [])
        if history:
            verification = history[-1].get("phase_outputs", {}).get("verification", {})
            failures = verification.get("failures", [])
    except Exception:
        pass
    return {
        "goal": goal,
        "stop_reason": stop,
        "failures": failures,
        "elapsed_sec": round(elapsed, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", nargs="+", required=True, help="List of goals to run")
    ap.add_argument("--cycles", type=int, default=3, help="Max cycles per goal")
    ap.add_argument("--json", type=str, help="Path to write JSON report")
    ap.add_argument("--dry-run", action="store_true", help="Do not apply changes")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    runtime = create_runtime(project_root, overrides=None)
    orchestrator = runtime["orchestrator"]

    reports = []
    for g in args.goals:
        reports.append(run_goal(orchestrator, g, args.cycles, args.dry_run))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(reports, indent=2))
    else:
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
