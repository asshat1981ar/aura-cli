"""Dispatch handlers for operational commands (B4).

Covers: queue, skills, metrics, workflow, scaffold, evolve, logs, watch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aura_cli.dispatch._helpers import _print_json_payload

from core.config_manager import config
from memory.brain import Brain


def handle_queue_list(ctx) -> int:
    goal_queue = ctx.runtime["goal_queue"]
    if getattr(ctx.args, "json", False):
        _print_json_payload({"queue": list(goal_queue.queue), "count": len(goal_queue.queue)}, parsed=ctx.parsed, indent=2)
        return 0

    if not goal_queue.queue:
        print("Goal queue is empty.")
        return 0

    print(f"Goal Queue ({len(goal_queue.queue)} goals):")
    for i, goal in enumerate(goal_queue.queue, 1):
        print(f"  {i}. {goal}")
    return 0


def handle_queue_clear(ctx) -> int:
    goal_queue = ctx.runtime["goal_queue"]
    count = len(goal_queue.queue)
    goal_queue.queue = []
    goal_queue._save()
    if getattr(ctx.args, "json", False):
        _print_json_payload({"cleared_count": count}, parsed=ctx.parsed, indent=2)
    else:
        print(f"Cleared {count} goals from the queue.")
    return 0


def handle_skills_list(ctx) -> int:
    """List all registered skills by name."""
    try:
        from agents.skills.registry import all_skills
        skills = all_skills()
        skill_names = sorted(skills.keys())
    except Exception as exc:
        if getattr(ctx.args, "json", False):
            _print_json_payload({"error": str(exc), "skills": []}, parsed=ctx.parsed, indent=2)
        else:
            print(f"Error loading skills: {exc}", file=sys.stderr)
        return 1

    if getattr(ctx.args, "json", False):
        _print_json_payload({"skills": skill_names, "count": len(skill_names)}, parsed=ctx.parsed, indent=2)
        return 0

    print(f"Registered skills ({len(skill_names)}):")
    for name in skill_names:
        print(f"  {name}")
    return 0


def handle_metrics_show(ctx) -> int:
    memory_store = ctx.runtime["memory_store"]
    log_entries = memory_store.read_log(limit=100)

    recent_data = []
    successes = 0
    skipped = 0
    fails = 0
    total_time = 0.0
    sandbox_failures = 0
    verification_failures = 0
    total_proposals = 0
    total_auto_queued_goals = 0
    total_queue_blocks = 0

    # 1. Try structured summaries from decision log
    summaries = [e["cycle_summary"] for e in log_entries if "cycle_summary" in e]
    if summaries:
        sandbox_failures = sum(
            1 for s in summaries
            if isinstance(s, dict) and (s.get("phase_status") or {}).get("sandbox") == "fail"
        )
        verification_failures = sum(
            1 for s in summaries
            if isinstance(s, dict) and s.get("verification_status") == "fail"
        )
        total_proposals = sum(
            int((s.get("proposal_count", 0) or 0))
            for s in summaries
            if isinstance(s, dict)
        )
        total_auto_queued_goals = sum(
            len(s.get("auto_queued_goals", []))
            for s in summaries
            if isinstance(s, dict)
        )
        total_queue_blocks = sum(
            len(s.get("queue_block_reasons", []))
            for s in summaries
            if isinstance(s, dict)
        )
        for s in summaries[-10:]:
            outcome = s.get("outcome", "FAILED")
            duration = s.get("duration_s", 0.0)

            if outcome == "SUCCESS": successes += 1
            elif outcome == "SKIPPED": skipped += 1
            else: fails += 1

            total_time += duration
            recent_data.append({
                "cycle_id": s.get("cycle_id"),
                "status": outcome,
                "duration": duration,
                "goal": s.get("goal"),
                "proposal_count": int(s.get("proposal_count", 0) or 0),
                "self_dev_mode": s.get("self_dev_mode"),
                "auto_queued_goals": list(s.get("auto_queued_goals", [])),
            })

    # 2. Fallback to legacy outcome strings in brain
    if not recent_data:
        brain = ctx.runtime["brain"]
        outcomes = [e for e in brain.recall_recent(limit=100) if "outcome:" in e]
        for raw in outcomes[:10]:
            try:
                # Format: outcome:id -> json
                data = json.loads(raw.split("->", 1)[1])
                status = "SUCCESS" if data.get("success") else "FAILED"
                duration = data.get("completed_at", 0) - data.get("started_at", 0)

                if data.get("success"): successes += 1
                else: fails += 1

                total_time += duration
                recent_data.append({
                    "cycle_id": data.get("cycle_id"),
                    "status": status,
                    "duration": duration,
                    "goal": data.get("goal")
                })
            except Exception:
                continue

    count = len(recent_data)
    avg_time = total_time / count if count else 0
    win_rate = (successes / count * 100) if count else 0

    if getattr(ctx.args, "json", False):
        payload = {
            "recent_cycles": recent_data,
            "summary": {
                "win_rate": win_rate,
                "avg_duration": avg_time,
                "avg_cycle_time": avg_time,
                "count": count,
                "successes": successes,
                "skipped": skipped,
                "fails": fails,
                "sandbox_failures": sandbox_failures,
                "verification_failures": verification_failures,
                "proposal_count": total_proposals,
                "auto_queued_goals": total_auto_queued_goals,
                "queue_block_reasons": total_queue_blocks,
            }
        }
        for i, s in enumerate(summaries[-10:]):
            recent_data[i]["stop_reason"] = s.get("stop_reason")
        _print_json_payload(payload, parsed=ctx.parsed, indent=2)
        return 0

    if not recent_data:
        print("No metrics recorded yet.")
        return 0

    print("Recent Cycle Metrics (last 10 cycles):\n")
    print(f"{'Cycle ID':<10} | {'Status':<8} | {'Dur':<6} | {'Stop Reason':<15} | {'Goal'}")
    print("-" * 85)

    for i, item in enumerate(recent_data):
        cid = (item['cycle_id'] or "unknown")[:8]
        stop = (summaries[-len(recent_data):][i].get("stop_reason") or "N/A") if summaries else "N/A"
        print(f"{cid:<10} | {item['status']:<8} | {item['duration']:>5.1f}s | {stop:<15} | {item['goal'][:30]}...")

    print("-" * 85)
    print(f"Summary: {win_rate:.1f}% success rate | {successes} pass, {skipped} skip, {fails} fail")
    print(f"Avg duration: {avg_time:.1f}s")
    print(f"Sandbox failures: {sandbox_failures} | Verification failures: {verification_failures}")
    print(f"Ralph proposals: {total_proposals} | Auto-queued goals: {total_auto_queued_goals} | Queue blocks: {total_queue_blocks}")

    # 3. Add strategy win rates from Brain KV store
    brain = ctx.runtime["brain"]
    strategy_stats = []
    for gt in ["bug_fix", "feature", "refactor", "security", "docs", "default"]:
        for s in ["minimal", "normal", "deep"]:
            stats = brain.get(f"__strategy_stats__:{gt}:{s}")
            if stats:
                wins = stats.get("wins", 0)
                losses = stats.get("losses", 0)
                total = wins + losses
                rate = (wins / total * 100) if total > 0 else 0.0
                strategy_stats.append(f"  {gt}/{s}: {rate:.1f}% ({wins}/{total})")
    
    if strategy_stats:
        print("\nStrategy Performance (Win Rates):")
        for line in strategy_stats:
            print(line)

    return 0


def handle_workflow_run(ctx) -> int:
    from core.operator_runtime import build_beads_runtime_metadata

    args = ctx.args
    orchestrator = ctx.runtime["orchestrator"]
    result = orchestrator.run_loop(
        args.workflow_goal,
        max_cycles=args.workflow_max_cycles or args.max_cycles or config.get("policy_max_cycles", config.get("max_cycles", 5)),
        dry_run=args.dry_run,
    )
    _print_json_payload(
        {
            "goal": args.workflow_goal,
            "stop_reason": result.get("stop_reason"),
            "cycles": len(result.get("history", [])),
            "beads_runtime": build_beads_runtime_metadata(orchestrator),
        },
        parsed=ctx.parsed,
        indent=2,
    )
    return 0


def handle_scaffold(ctx) -> int:
    from agents.scaffolder import ScaffolderAgent

    args = ctx.args
    runtime = ctx.runtime

    scaffolder = ScaffolderAgent(runtime.get("brain", None) or Brain(), runtime["model_adapter"])
    result = scaffolder.scaffold_project(args.scaffold, args.scaffold_desc)

    if getattr(args, "json", False):
        _print_json_payload({"result": result, "project_name": args.scaffold}, parsed=ctx.parsed, indent=2)
    else:
        print(result)
    return 0


def handle_evolve(ctx) -> int:
    args = ctx.args
    runtime = ctx.runtime
    orchestrator = runtime.get("orchestrator")
    goal = args.goal or args.workflow_goal or "evolve and improve the AURA system"
    result = orchestrator.run_self_development(goal=goal, mode=getattr(args, "evolve_mode", None))
    _print_json_payload(result, parsed=ctx.parsed, indent=2, default=str)
    return 0


def handle_logs(ctx) -> int:
    from aura_cli.tui.log_streamer import LogStreamer

    streamer = LogStreamer(level_filter=getattr(ctx.args, "level", "info"))
    if getattr(ctx.args, "file", None):
        streamer.stream_file(Path(ctx.args.file), tail=getattr(ctx.args, "tail", None), follow=getattr(ctx.args, "follow", False))
    else:
        streamer.stream_stdin(tail=getattr(ctx.args, "tail", None))
    return 0


def handle_watch(ctx) -> int:
    from aura_cli.tui.app import AuraStudio

    studio = AuraStudio(runtime=ctx.runtime or {})
    orchestrator = ctx.runtime.get("orchestrator")
    if orchestrator:
        orchestrator.attach_ui_callback(studio)

    studio.run(autonomous=getattr(ctx.args, "autonomous", False))
    return 0
