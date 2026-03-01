import json
from pathlib import Path

from core.capability_manager import build_capability_status_report, capability_doctor_check
from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive
from core.logging_utils import log_json
from core.task_handler import run_goals_loop # Import run_goals_loop from core.task_handler
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
    run_doctor_v2(project_root=project_root, rich_output=True)

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
    run_goals_loop(args, goal_queue, orchestrator, debugger_instance, planner_instance, goal_archive, project_root, decompose=getattr(args, 'decompose', False))

def _handle_status(
    goal_queue: GoalQueue,
    goal_archive: GoalArchive,
    orchestrator,
    *,
    as_json: bool = False,
    project_root: Path | None = None,
    memory_persistence_path: Path | str | None = None,
):
    log_json("INFO", "aura_status_requested")
    
    from core.operator_runtime import build_operator_runtime_snapshot
    from core.capability_manager import build_capability_status_report

    active_cycle = getattr(orchestrator, "active_cycle_summary", None)
    last_cycle = getattr(orchestrator, "last_cycle_summary", None)
    
    snapshot = build_operator_runtime_snapshot(
        goal_queue=goal_queue,
        goal_archive=goal_archive,
        active_cycle=active_cycle,
        last_cycle=last_cycle,
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
    for item in snapshot['queue']['pending']:
        print(f"  {item['position']}. {item['goal']}")
    
    print(f"Completed goals: {snapshot['queue']['completed_count']}")
    for item in snapshot['queue']['completed']:
        score_str = f" (Score: {item['score']:.2f})" if item['score'] is not None else ""
        print(f"  - '{item['goal']}'{score_str}")

    if active_cycle:
        print("\n--- Active Cycle ---")
        print(f"ID: {active_cycle['cycle_id']}")
        print(f"Goal: {active_cycle['goal']}")
        print(f"Phase: {active_cycle['current_phase']} ({active_cycle['state']})")

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
