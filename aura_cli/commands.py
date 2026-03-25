import json
from pathlib import Path

from core.capability_manager import build_capability_status_report, capability_doctor_check
from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive
from core.logging_utils import log_json
from core.task_handler import run_goals_loop  # Import run_goals_loop from core.task_handler
from core.task_manager import TaskManager


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

    run_doctor_v2(
        project_root=project_root,
        rich_output=True,
        capability_check=capability_doctor_check,
    )


def _handle_readiness():
    """M7-004: Validate async runtime and MCP registry health."""
    import anyio
    from core.mcp_health import check_all_mcp_health, get_health_summary
    from core.mcp_agent_registry import agent_registry
    from core.config_manager import config

    print("\n--- AURA Readiness Check (V2) ---")
    print(f"Async Orchestrator: {'ENABLED' if config.get('enable_new_orchestrator') else 'DISABLED'}")
    print(f"Typed Registry:     {'ENABLED' if config.get('enable_mcp_registry') else 'DISABLED'}")

    results = anyio.run(check_all_mcp_health)
    summary = get_health_summary(results)

    print(f"\nMCP Servers: {summary['healthy_count']}/{summary['total_servers']} healthy")
    for res in results:
        status_icon = "✅" if res["status"] == "healthy" else "❌"
        print(f"  {status_icon} {res['name']:<15} (Port: {res.get('port', 'N/A')})")

    agents = agent_registry.list_agents()
    print(f"\nRegistered Agents: {len(agents)}")
    local_count = sum(1 for a in agents if a.source == "local")
    mcp_count = sum(1 for a in agents if a.source == "mcp")
    print(f"  - Local: {local_count}")
    print(f"  - MCP:   {mcp_count}")

    if summary["all_healthy"]:
        print("\n✅ System is READY for autonomous operation.\n")
    else:
        print("\n⚠️  System is DEGRADED. Check unhealthy MCP servers.\n")


def _handle_clear():
    import os

    os.system("clear" if os.name == "posix" else "cls")


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
    run_goals_loop(args, goal_queue, orchestrator, debugger_instance, planner_instance, goal_archive, project_root, decompose=getattr(args, "decompose", False))


def _handle_status(
    goal_queue: GoalQueue,
    goal_archive: GoalArchive,
    orchestrator,
    *,
    as_json: bool = False,
    project_root: Path | None = None,
    memory_persistence_path: Path | str | None = None,
    memory_store=None,
):
    log_json("INFO", "aura_status_requested")

    from core.operator_runtime import build_beads_runtime_metadata, build_operator_runtime_snapshot

    active_cycle = getattr(orchestrator, "active_cycle_summary", None)
    last_cycle = getattr(orchestrator, "last_cycle_summary", None)
    beads_runtime = build_beads_runtime_metadata(orchestrator)

    snapshot = build_operator_runtime_snapshot(
        goal_queue=goal_queue,
        goal_archive=goal_archive,
        active_cycle=active_cycle,
        last_cycle=last_cycle,
        beads_runtime=beads_runtime,
        memory_store=memory_store,
    )

    if as_json:
        # Include legacy capabilities field for now to avoid breaking tools
        capability_status = build_capability_status_report(
            Path(project_root or Path.cwd()),
            goal_queue=goal_queue,
            last_status=getattr(orchestrator, "last_capability_status", None),
        )
        snapshot["capabilities"] = capability_status
        print(json.dumps(snapshot))
        return

    print("\n--- AURA Status ---")
    print(f"Goals in queue: {snapshot['queue']['pending_count']}")
    for item in snapshot["queue"]["pending"]:
        print(f"  {item['position']}. {item['goal']}")

    print(f"Completed goals: {snapshot['queue']['completed_count']}")
    for item in snapshot["queue"]["completed"]:
        score_str = f" (Score: {item['score']:.2f})" if item["score"] is not None else ""
        print(f"  - '{item['goal']}'{score_str}")

    if beads_runtime:
        gate_mode = "required" if beads_runtime.get("required") else "optional"
        print("\n--- BEADS Gate ---")
        print(f"Enabled: {'yes' if beads_runtime.get('enabled') else 'no'}")
        print(f"Mode: {gate_mode}")
        print(f"Scope: {beads_runtime.get('scope', 'goal_run')}")

    if active_cycle:
        print("\n--- Active Cycle ---")
        print(f"ID: {active_cycle['cycle_id']}")
        print(f"Goal: {active_cycle['goal']}")
        print(f"Phase: {active_cycle['current_phase']} ({active_cycle['state']})")

    run_tool_audit = snapshot.get("run_tool_audit")
    if isinstance(run_tool_audit, dict):
        print("\n--- Run Tool Audit ---")
        print(f"Recent commands tracked: {run_tool_audit.get('count', 0)}")
        print(f"Last command: {run_tool_audit.get('last_command') or 'n/a'}")
        print(
            "Recent outcomes: "
            f"{run_tool_audit.get('success_count', 0)} ok, "
            f"{run_tool_audit.get('error_count', 0)} error, "
            f"{run_tool_audit.get('timeout_count', 0)} timed out, "
            f"{run_tool_audit.get('truncated_count', 0)} truncated"
        )
        if active_cycle.get("beads_status"):
            beads_label = active_cycle["beads_status"]
            if active_cycle.get("beads_decision_id"):
                beads_label += f" ({active_cycle['beads_decision_id']})"
            print(f"BEADS: {beads_label}")
            if active_cycle.get("beads_summary"):
                print(f"BEADS Summary: {active_cycle['beads_summary']}")
    elif last_cycle:
        print("\n--- Last Cycle ---")
        print(f"ID: {last_cycle['cycle_id']}")
        print(f"Goal: {last_cycle['goal']}")
        print(f"Outcome: {last_cycle['outcome']}")
        print(f"Stop Reason: {last_cycle['stop_reason']}")
        if last_cycle.get("beads_status"):
            beads_label = last_cycle["beads_status"]
            if last_cycle.get("beads_decision_id"):
                beads_label += f" ({last_cycle['beads_decision_id']})"
            print(f"BEADS: {beads_label}")
            if last_cycle.get("beads_summary"):
                print(f"BEADS Summary: {last_cycle['beads_summary']}")

    # Task Hierarchy
    task_manager = TaskManager(persistence_path=memory_persistence_path)
    if task_manager.root_tasks:
        print("\n--- Task Hierarchy ---")
        for task in task_manager.root_tasks:
            print(task.display())

    # Add Loop status
    print("\n--- AURA Loop Status ---")
    if hasattr(orchestrator, "current_score"):
        print(f"Current Loop Score: {orchestrator.current_score:.2f}")
        print(f"Regression Count: {orchestrator.regression_count}")
        print(f"Stable Convergence Count: {orchestrator.stable_convergence_count}")
        print(f"Current Goal: {orchestrator.current_goal if orchestrator.current_goal else 'None'}")
    elif active_cycle:
        print(f"Orchestrator active. Current Goal: {active_cycle['goal']}")
    else:
        print("Loop status: idle.")
    print("------------------------\n")


def _handle_exit():
    log_json("INFO", "aura_cli_exit")
