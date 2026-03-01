import sys
from pathlib import Path

from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive
from core.logging_utils import log_json
from core.task_handler import run_goals_loop # Import run_goals_loop from core.task_handler
from core.task_manager import TaskManager

def _handle_help():
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
    from aura_doctor import check_python_version, check_env_vars, check_sqlite_write_access, check_git_status, check_pytest_and_run_tests
    from pathlib import Path
    
    repo_root = Path.cwd()
    results = [
        f"Python Version: {check_python_version()[0]} - {check_python_version()[1]}",
        f"Environment Variables: {check_env_vars()[0]} - {check_env_vars()[1]}",
        f"SQLite Write Access: {check_sqlite_write_access(repo_root)[0]} - {check_sqlite_write_access(repo_root)[1]}",
        f"Git Status: {check_git_status(repo_root)[0]} - {check_git_status(repo_root)[1]}",
        f"Pytest Tests: {check_pytest_and_run_tests(repo_root, True)[0]} - {check_pytest_and_run_tests(repo_root, True)[1]}"
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
    if goal:
        goal_queue.add(goal)
        log_json("INFO", "goal_added", goal=goal)
    else:
        log_json("WARN", "add_command_no_goal_provided")

def _handle_run(args, goal_queue: GoalQueue, goal_archive: GoalArchive, loop, debugger_instance, planner_instance, project_root: Path):
    if not goal_queue.has_goals():
        log_json("WARN", "run_command_no_goals_in_queue")
        return
    run_goals_loop(args, goal_queue, loop, debugger_instance, planner_instance, goal_archive, project_root, decompose=getattr(args, 'decompose', False))

def _handle_status(goal_queue: GoalQueue, goal_archive: GoalArchive, loop):
    log_json("INFO", "aura_status_requested")
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

    # Add HybridClosedLoop status
    print("\n--- AURA Loop Status ---")
    print(f"Current Loop Score: {loop.current_score:.2f}")
    print(f"Regression Count: {loop.regression_count}")
    print(f"Stable Convergence Count: {loop.stable_convergence_count}")
    print(f"Current Goal: {loop.current_goal if loop.current_goal else 'None'}")
    print("------------------------\n")

def _handle_exit():
    log_json("INFO", "aura_cli_exit")
