import os
import sys
import argparse
from pathlib import Path

try:
    import readline
except ImportError:
    readline = None

from core.goal_queue import GoalQueue
from core.hybrid_loop import HybridClosedLoop
from core.goal_archive import GoalArchive
from memory.brain import Brain
from core.model_adapter import ModelAdapter
from core.git_tools import GitTools, GitToolsError
from core.logging_utils import log_json
from agents.debugger import DebuggerAgent
from agents.planner import PlannerAgent

from core.task_handler import _check_project_writability, run_goals_loop
from cli.commands import _handle_add, _handle_run, _handle_status, _handle_exit, _handle_help

def cli_interaction_loop(args, goal_queue, goal_archive, loop, debugger_instance, planner_instance, project_root):
    while True:
        try:
            command_line = input("\nCommand (add/run/exit/status/help) > ").strip()
            command_parts = command_line.split(maxsplit=1)
            if not command_parts:
                continue
            cmd_name = command_parts[0]
            
            if cmd_name == "add":
                _handle_add(goal_queue, command_line)
            elif cmd_name == "run":
                _handle_run(args, goal_queue, goal_archive, loop, debugger_instance, planner_instance, project_root)
            elif cmd_name == "status":
                _handle_status(goal_queue, goal_archive, loop)
            elif cmd_name == "doctor":
                _handle_doctor()
            elif cmd_name == "clear":
                _handle_clear()
            elif cmd_name == "help":
                _handle_help()
            elif cmd_name == "exit":
                _handle_exit()
                break
            else:
                log_json("WARN", "invalid_cli_command", details={"command": cmd_name})
        except EOFError:
            log_json("INFO", "aura_cli_eof_received", details={"message": "End of input received, exiting."})
            break

def main(project_root_override=None):
    # Resolve project root
    if project_root_override:
        project_root = Path(project_root_override)
    else:
        project_root = Path(__file__).resolve().parent.parent
        
        # If we are in a test environment, prefer the current working directory
        if os.getenv("AURA_SKIP_CHDIR") == "1":
            project_root = Path.cwd()
        else:
            # Change directory to the project root for consistency in real-world usage
            os.chdir(project_root)

    # Ensure project root is on path for module imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Setup readline history
    if readline:
        history_file = project_root / "memory" / ".aura_history"
        try:
            if history_file.exists():
                readline.read_history_file(str(history_file))
            readline.set_history_length(1000)
        except Exception:
            pass # Silent fail if history loading fails

    parser = argparse.ArgumentParser(description="AURA CLI for autonomous development.")
    parser.add_argument("--dry-run", action="store_true", help="Run the AURA loop in dry-run mode.")
    parser.add_argument("--add-goal", type=str, help="Add a goal to the queue.")
    parser.add_argument("--run-goals", action="store_true", help="Run the AURA loop for goals currently in the queue.")
    parser.add_argument("--decompose", action="store_true", help="Decompose complex goals into hierarchical sub-tasks.")
    args = parser.parse_args()

    goal_queue = GoalQueue()
    goal_archive = GoalArchive()

    model_adapter = ModelAdapter()
    brain_instance = Brain()
    debugger_instance = DebuggerAgent(brain_instance, model_adapter)
    planner_instance = PlannerAgent(brain_instance, model_adapter)
    
    try:
        git_tools_instance = GitTools(repo_path=str(project_root))
    except GitToolsError as e:
        log_json("ERROR", "git_tools_init_failed", details={"error": str(e)})
        return

    loop = HybridClosedLoop(model_adapter, brain_instance, git_tools_instance)

    log_json("INFO", "aura_cli_online", details={"dry_run_mode": getattr(args, 'dry_run', False)})

    if not _check_project_writability(project_root):
        log_json("CRITICAL", "aura_cli_startup_aborted_not_writable")
        return

    if args.add_goal:
        goal_queue.add(args.add_goal)
        log_json("INFO", "goal_added_from_cli", goal=args.add_goal)

    try:
        if args.run_goals:
            run_goals_loop(args, goal_queue, loop, debugger_instance, planner_instance, goal_archive, project_root, decompose=args.decompose)
            return

        cli_interaction_loop(args, goal_queue, goal_archive, loop, debugger_instance, planner_instance, project_root)
    finally:
        if readline:
            try:
                readline.write_history_file(str(project_root / "memory" / ".aura_history"))
            except Exception:
                pass

if __name__ == "__main__":
    main()
