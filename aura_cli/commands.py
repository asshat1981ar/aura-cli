import json
from pathlib import Path

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

def _handle_doctor():
    log_json("INFO", "aura_doctor_requested")
    from aura_cli.doctor import check_python_version, check_env_vars, check_sqlite_write_access, check_git_status, check_pytest_and_run_tests
    from pathlib import Path
    
    repo_root = Path.cwd()
    results = [
        f"Python Version: {check_python_version()[0]} - {check_python_version()[1]}",
        f"Environment Variables: {check_env_vars()[0]} - {check_env_vars()[1]}",
        f"SQLite Write Access: {check_sqlite_write_access(repo_root)[0]} - {check_sqlite_write_access(repo_root)[1]}",
        f"Git Status: {check_git_status(repo_root)[0]} - {check_git_status(repo_root)[1]}",
        f"Pytest Tests: {check_pytest_and_run_tests(repo_root, False)[0]} - {check_pytest_and_run_tests(repo_root, False)[1]}"
    ]
    
    print("\n--- AURA Doctor Report ---")
    for res in results:
        print(f"- {res}")
    print("--------------------------\n")

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

def _handle_run(args, goal_queue: GoalQueue, goal_archive: GoalArchive, loop, debugger_instance, planner_instance, project_root: Path):
    if not goal_queue.has_goals():
        log_json("WARN", "run_command_no_goals_in_queue")
        return
    run_goals_loop(args, goal_queue, loop, debugger_instance, planner_instance, goal_archive, project_root, decompose=getattr(args, 'decompose', False))

def _handle_status(goal_queue: GoalQueue, goal_archive: GoalArchive, loop, as_json: bool = False):
    log_json("INFO", "aura_status_requested")
    if as_json:
        data = {
            "queue_length": len(goal_queue.queue),
            "queue": list(goal_queue.queue),
            "completed_count": len(goal_archive.completed),
            "completed": [{"goal": g, "score": s} for g, s in goal_archive.completed],
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

    # Task Hierarchy
    task_manager = TaskManager()
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
