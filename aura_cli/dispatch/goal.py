"""Dispatch handlers for goal-related commands (B4)."""
from __future__ import annotations

from aura_cli.dispatch._helpers import _print_json_payload, _run_json_printing_callable_with_warnings

from core.config_manager import config
from core.logging_utils import log_json
from core.task_handler import run_goals_loop
from aura_cli.commands import _handle_status


def _maybe_add_goal(ctx) -> None:
    if not getattr(ctx.args, "add_goal", None):
        return
    goal_queue = ctx.runtime["goal_queue"]
    goal_queue.add(ctx.args.add_goal)
    log_json("INFO", "goal_added_from_cli", goal=ctx.args.add_goal)
    if not getattr(ctx.args, "json", False):
        print(f"Added goal: {ctx.args.add_goal}")
        print(f"Queue length: {len(goal_queue.queue)}")


def handle_goal_once(ctx) -> int:
    from core.explain import format_decision_log
    from core.operator_runtime import build_beads_runtime_metadata

    args = ctx.args
    orchestrator = ctx.runtime["orchestrator"]
    result = orchestrator.run_loop(
        args.goal,
        max_cycles=args.max_cycles or config.get("policy_max_cycles", config.get("max_cycles", 5)),
        dry_run=args.dry_run,
    )
    history = result.get("history", [])
    if args.explain:
        print(format_decision_log(history))

    if getattr(args, "json", False):
        _print_json_payload(
            {
                "goal": args.goal,
                "stop_reason": result.get("stop_reason"),
                "cycles": len(history),
                "dry_run": args.dry_run,
                "beads_runtime": build_beads_runtime_metadata(orchestrator),
            },
            parsed=ctx.parsed,
            indent=2,
        )
    else:
        print(f"\n--- Goal Result Summary ---")
        print(f"Goal: {args.goal}")
        print(f"Stop Reason: {result.get('stop_reason')}")
        print(f"Cycles Completed: {len(history)}")
        if args.dry_run:
            print("Mode: Dry-run (read-only)")
        print("---------------------------\n")
    return 0


def handle_goal_run(ctx) -> int:
    args = ctx.args
    runtime = ctx.runtime
    run_goals_loop(
        args,
        runtime["goal_queue"],
        runtime["orchestrator"],
        runtime["debugger"],
        runtime["planner"],
        runtime["goal_archive"],
        ctx.project_root,
        decompose=args.decompose,
    )
    return 0


def handle_goal_add(ctx) -> int:
    _maybe_add_goal(ctx)
    return 0


def handle_goal_add_run(ctx) -> int:
    _maybe_add_goal(ctx)
    return handle_goal_run(ctx)


def handle_goal_status(ctx) -> int:
    runtime = ctx.runtime
    if ctx.args.json:
        _run_json_printing_callable_with_warnings(
            ctx,
            _handle_status,
            runtime["goal_queue"],
            runtime["goal_archive"],
            runtime["orchestrator"],
            as_json=True,
            project_root=ctx.project_root,
            memory_persistence_path=runtime.get("memory_persistence_path"),
        )
    else:
        _handle_status(
            runtime["goal_queue"],
            runtime["goal_archive"],
            runtime["orchestrator"],
            as_json=False,
            project_root=ctx.project_root,
            memory_persistence_path=runtime.get("memory_persistence_path"),
        )
    return 0


def handle_interactive(ctx) -> int:
    from aura_cli.cli_main import cli_interaction_loop

    runtime = ctx.runtime
    cli_interaction_loop(ctx.args, runtime)
    return 0
