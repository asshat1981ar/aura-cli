import json
from pathlib import Path

from core.capability_manager import build_capability_status_report, capability_doctor_check
from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive
from core.logging_utils import log_json
from core.operator_runtime import build_beads_runtime_metadata, build_operator_runtime_snapshot
from core.task_manager import TaskManager
from core.task_handler import run_goals_loop

def _truncate_status_text(value, *, limit: int = 60) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

def _summarize_failed_phases(summary: dict) -> str | None:
    phase_status = summary.get("phase_status")
    if not isinstance(phase_status, dict):
        return None
    failed = [phase for phase, status in phase_status.items() if status == "fail"]
    if not failed:
        return None
    return ", ".join(failed)

def _summarize_failures(summary: dict, *, limit: int = 3) -> str | None:
    failures = summary.get("failures")
    if not isinstance(failures, list) or not failures:
        return None

    rendered = [_truncate_status_text(item) for item in failures[:limit]]
    remaining = len(failures) - limit
    summary_text = "; ".join(rendered)
    if remaining > 0:
        summary_text += f" (+{remaining} more)"
    return summary_text

def _handle_help():
    try:
        from aura_cli.cli_options import render_help
        print(render_help().rstrip())
    except Exception:
        # Fallback to a minimal static help if parser help rendering fails.
        print("\n--- AURA CLI Commands ---")
        print("add <goal_description> - Add a new goal to the queue.")
        print("run                    - Run the AURA loop for goals in the queue.")
        print("status                 - Show the current status of tasks and goals.")
        print("doctor                 - Run system health checks.")
        print("clear                  - Clear the screen.")
        print("exit                   - Exit the AURA CLI.")
        print("help                   - Show this help message.")
        print("-------------------------\n")

def _handle_doctor(project_root: Path | None = None):
    log_json("INFO", "aura_doctor_requested")
    from aura_cli.doctor import run_doctor_v2

    repo_root = Path(project_root or Path.cwd())
    # run_doctor_v2 handles rich output printing internally if rich is available
    run_doctor_v2(project_root=repo_root, rich_output=True)

def _handle_clear():
    import os
    os.system('clear' if os.name == 'posix' else 'cls')

def _handle_add(goal_queue: GoalQueue, command: str):
    goal = command[4:].strip()
    if not goal:
        log_json("WARN", "add_command_no_goal_provided")
        return

    # Basic Sanitization & Security
    if len(goal) > 500:
        log_json("ERROR", "goal_title_too_long", details={"length": len(goal)})
        print("Error: Goal description is too long (max 500 chars).")
        return

    # Check for suspicious shell-like characters if needed, 
    # though GoalQueue usually just stores it as a string in JSON.
    # However, preventing characters that might be interpreted in some contexts is good.
    forbidden_chars = [";", "&&", "||", "`", "$("]
    if any(char in goal for char in forbidden_chars):
        log_json("SECURITY_WARN", "suspicious_goal_content", details={"goal": goal})
        print("Error: Goal description contains suspicious characters.")
        return

    goal_queue.add(goal)
    log_json("INFO", "goal_added", goal=goal)
    print(f"Added goal: {goal}")
    print(f"Queue length: {len(goal_queue.queue)}")

def _handle_run(args, goal_queue: GoalQueue, goal_archive: GoalArchive, orchestrator, debugger_instance, planner_instance, project_root: Path):
    if not goal_queue.has_goals():
        log_json("WARN", "run_command_no_goals_in_queue")
        return

    # Delegate to the shared task handler loop which manages retries and policy errors.
    run_goals_loop(
        args,
        goal_queue,
        orchestrator,
        debugger_instance,
        planner_instance,
        goal_archive,
        project_root,
        decompose=getattr(args, "decompose", False),
    )

def _handle_status(
    goal_queue: GoalQueue,
    goal_archive: GoalArchive,
    loop,
    *,
    as_json: bool = False,
    project_root: Path | None = None,
    memory_persistence_path: Path | str | None = None,
):
    log_json("INFO", "aura_status_requested")
    capability_status = build_capability_status_report(
        Path(project_root or Path.cwd()),
        goal_queue=goal_queue,
        last_status=getattr(loop, "last_capability_status", None),
    )
    active_cycle = getattr(loop, "active_cycle_summary", None)
    last_cycle = getattr(loop, "last_cycle_summary", None)
    beads_runtime = build_beads_runtime_metadata(loop)
    has_operator_runtime = any(
        value is not None for value in (active_cycle, last_cycle, beads_runtime)
    )

    if as_json:
        if has_operator_runtime:
            data = build_operator_runtime_snapshot(
                goal_queue,
                goal_archive,
                active_cycle=active_cycle,
                last_cycle=last_cycle,
                beads_runtime=beads_runtime,
            )
            queue_summary = data["queue"]
            data["queue_length"] = queue_summary["pending_count"]
            data["completed_count"] = queue_summary["completed_count"]
            data["completed"] = queue_summary["completed"]
            data["capabilities"] = capability_status
        else:
            data = {
                "queue_length": len(goal_queue.queue),
                "queue": list(goal_queue.queue),
                "completed_count": len(goal_archive.completed),
                "completed": [{"goal": g, "score": s} for g, s in goal_archive.completed],
                "capabilities": capability_status,
            }
        print(json.dumps(data))
        return

    print("\n--- AURA Status ---")
    print(f"Goals in queue: {len(goal_queue.queue)}")
    for i, goal in enumerate(goal_queue.queue):
        print(f"  {i+1}. {goal}")
    print(f"Completed goals: {len(goal_archive.completed)}")
    for goal, score in goal_archive.completed:
        print(f"  - '{goal}' (Score: {score:.2f})")

    print("\n--- Capability Bootstrap ---")
    print(f"Last analyzed goal: {capability_status['last_goal'] or 'None'}")
    matched = capability_status["matched_capability_ids"]
    print(f"Matched capability rules: {', '.join(matched) if matched else 'None'}")
    print(f"Pending self-development goals: {len(capability_status['pending_self_development_goals'])}")
    print(
        "MCP bootstrap pending: "
        + (", ".join(capability_status["pending_bootstrap_actions"]) if capability_status["pending_bootstrap_actions"] else "None")
    )
    print(
        "MCP bootstrap running: "
        + (", ".join(capability_status["running_bootstrap_actions"]) if capability_status["running_bootstrap_actions"] else "None")
    )

    print("\n--- BEADS Gate ---")
    if beads_runtime:
        print(f"Mode: {'required' if beads_runtime.get('required') else 'optional'}")
        print(f"Enabled: {'on' if beads_runtime.get('enabled') else 'off'}")
        if beads_runtime.get("scope"):
            print(f"Scope: {beads_runtime['scope']}")
        if beads_runtime.get("runtime_mode"):
            print(f"Runtime: {beads_runtime['runtime_mode']}")
    else:
        print("Mode: unavailable")

    def _render_cycle(label: str, summary: dict | None):
        if not summary:
            return
        print(f"\n--- {label} ---")
        print(f"Goal: {summary.get('goal') or 'None'}")
        print(f"State: {summary.get('state') or 'unknown'}")
        if summary.get("current_phase"):
            print(f"Current Phase: {summary['current_phase']}")
        if summary.get("outcome"):
            print(f"Outcome: {summary['outcome']}")
        if summary.get("stop_reason"):
            print(f"Stop Reason: {summary['stop_reason']}")
        verification_status = summary.get("verification_status")
        if verification_status and verification_status != "pass":
            print(f"Verification: {str(verification_status).upper()}")
        retry_count = summary.get("retry_count")
        if isinstance(retry_count, int) and retry_count > 0:
            print(f"Retries: {retry_count}")
        failed_phases = _summarize_failed_phases(summary)
        if failed_phases:
            print(f"Failed Phases: {failed_phases}")
        applied_files = summary.get("applied_files")
        failed_files = summary.get("failed_files")
        applied_count = len(applied_files) if isinstance(applied_files, list) else 0
        failed_count = len(failed_files) if isinstance(failed_files, list) else 0
        if applied_count or failed_count:
            print(f"Files Applied: {applied_count}")
            print(f"Files Failed: {failed_count}")
        failure_summary = _summarize_failures(summary)
        if failure_summary:
            print(f"Failures: {failure_summary}")
        queued_follow_up_goals = summary.get("queued_follow_up_goals")
        if isinstance(queued_follow_up_goals, list) and queued_follow_up_goals:
            print(f"Follow-up Goals Queued: {len(queued_follow_up_goals)}")
        if summary.get("failure_routing_decision"):
            print(f"Failure Route: {summary['failure_routing_decision']}")
        if summary.get("failure_routing_reason"):
            print(f"Failure Route Reason: {_truncate_status_text(summary['failure_routing_reason'], limit=80)}")
        beads_status = summary.get("beads_status")
        beads_decision_id = summary.get("beads_decision_id")
        if beads_status:
            if beads_decision_id:
                print(f"BEADS: {beads_status} ({beads_decision_id})")
            else:
                print(f"BEADS: {beads_status}")
        if summary.get("beads_summary"):
            print(f"BEADS Summary: {summary['beads_summary']}")

    _render_cycle("Active Cycle", active_cycle)
    _render_cycle("Last Cycle", last_cycle)

    # Task Hierarchy
    task_manager = TaskManager(persistence_path=memory_persistence_path)
    if task_manager.root_tasks:
        print("\n--- Task Hierarchy ---")
        for task in task_manager.root_tasks:
            print(task.display())

    # Add Loop status
    print("\n--- AURA Loop Status ---")
    if hasattr(loop, "current_score"):
        print(f"Current Loop Score: {loop.current_score:.2f}")
        print(f"Regression Count: {loop.regression_count}")
        print(f"Stable Convergence Count: {loop.stable_convergence_count}")
        print(f"Current Goal: {loop.current_goal if loop.current_goal else 'None'}")
    else:
        print("Loop status: orchestrator active (no legacy score fields).")
    print("------------------------\n")

def _handle_exit():
    log_json("INFO", "aura_cli_exit")
