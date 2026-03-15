import difflib
import json
import re
import time
from pathlib import Path

from core.file_tools import (
    MISMATCH_OVERWRITE_BLOCK_EVENT,
    MismatchOverwriteBlockedError,
    mismatch_overwrite_block_log_details,
)
from core.logging_utils import log_json
from core.task_manager import TaskManager, Task
from core.path_resolver import (
    check_project_writability,
    validate_change_target_path,
    allow_new_test_file_target,
    find_candidate_existing_files,
)

# Constants moved to core/path_resolver.py, no longer needed here


def _goal_cycle_limit(args) -> int:
    """Return the per-goal cycle limit, honoring CLI overrides when present."""
    raw_limit = getattr(args, "max_cycles", None)
    if raw_limit is None:
        return 10
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return 10
    return max(1, limit)


def _compose_loop_goal(task_title: str, grounding_hint: str | None) -> str:
    if not grounding_hint:
        return task_title
    return f"{task_title}\n\nGROUNDING_HINT:\n{grounding_hint}"


def _invalid_path_grounding_hint(file_path: str, reason: str, candidate_files: list[str]) -> str:
    base = (
        "Previous IMPLEMENT proposed an invalid file target "
        f"('{file_path}', reason: {reason}). "
        "Use an existing repository file path for the next IMPLEMENT and keep edits targeted."
    )
    if candidate_files:
        primary = candidate_files[0]
        return (
            f"{base}\nClosest existing match: {primary}\n"
            "Do not invent a new top-level directory when an exact filename match already exists.\n"
            "Candidate existing files (choose one if relevant):\n- " + "\n- ".join(candidate_files)
        )
    return base


def _mismatch_overwrite_blocked_grounding_hint(file_path: str) -> str:
    return (
        "Previous IMPLEMENT targeted an existing file but the provided old_code did not match the current file "
        f"('{file_path}'). The queue safety policy blocked automatic full-file overwrite fallback. "
        "For the next IMPLEMENT, either provide an exact current old_code snippet from the file, or if a full-file "
        "replacement is intentional, set overwrite_file to true, set old_code to an empty string, and provide the "
        "complete replacement content."
    )


def run_goals_loop(args, goal_queue, orchestrator, debugger_instance, planner_instance, goal_archive, project_root, decompose=False):
    """
    Processes all goals in the queue using the hierarchical TaskManager and LoopOrchestrator.
    """
    task_manager = TaskManager()
    cycle_limit = _goal_cycle_limit(args)
    dry_run = getattr(args, "dry_run", False)

    while goal_queue.has_goals():
        goal = goal_queue.next()
        log_json("INFO", "processing_goal", goal=goal)

        if decompose and planner_instance:
            root_task = task_manager.decompose_goal(goal, planner_instance)
            tasks_to_process = root_task.subtasks
        else:
            root_task = Task(id=f"goal_{int(time.time())}", title=goal)
            task_manager.add_task(root_task)
            tasks_to_process = [root_task]

        for task in tasks_to_process:
            if task != root_task:
                log_json("INFO", "executing_subtask", details={"task_id": task.id, "title": task.title})

            task.status = "in_progress"
            task_manager.save()
            
            current_goal_text = task.title
            
            # Allow one retry for policy blocks
            for attempt_idx in range(2):
                try:
                    result = orchestrator.run_loop(current_goal_text, max_cycles=cycle_limit, dry_run=dry_run)
                    
                    # Check for policy blocking in history
                    history = result.get("history", [])
                    mismatch_block_error = None
                    if history:
                        last_cycle = history[-1]
                        apply_result = last_cycle.get("phase_outputs", {}).get("apply_result", {})
                        failed = apply_result.get("failed", [])
                        for failure in failed:
                            err_msg = str(failure.get("error", ""))
                            if "mismatch overwrite fallback is disabled" in err_msg:
                                mismatch_block_error = err_msg
                                break
                    
                    if mismatch_block_error:
                        if attempt_idx == 0:
                            log_json("WARN", MISMATCH_OVERWRITE_BLOCK_EVENT, details={"error": mismatch_block_error, "policy": "explicit_overwrite_file_required"})
                            
                            # Construct grounding hint
                            import re
                            # Extract file path from error message
                            matches = re.findall(r"'([^']+)'", mismatch_block_error)
                            file_path_hint = matches[1] if len(matches) >= 2 else "unknown_file"
                            
                            hint = _mismatch_overwrite_blocked_grounding_hint(file_path_hint)
                            current_goal_text = _compose_loop_goal(task.title, hint)
                            
                            log_json("INFO", "grounding_retry_scheduled", details={"original_goal": task.title})
                            continue
                        else:
                            # Second failure, abort
                            task.status = "failed"
                            task_manager.save()
                            log_json("ERROR", "goal_execution_failed_after_retry", details={"error": mismatch_block_error, "goal": task.title})
                            break

                    stop_reason = result.get("stop_reason")
                    
                    # Check outcome based on stop reason
                    if stop_reason == "MAX_CYCLES":
                        task.status = "failed"
                        log_json("WARN", "cycle_limit_reached", goal=task.title, details={"cycle_limit": cycle_limit})
                    elif stop_reason == "INVALID_OUTPUT":
                        if attempt_idx == 0:
                            candidates = find_candidate_existing_files(project_root, "", task.title)
                            
                            hint = _invalid_path_grounding_hint("", "file_not_found", candidates)
                            current_goal_text = _compose_loop_goal(task.title, hint)
                            log_json("WARN", "invalid_implement_target_path", details={
                                "reason": "file_not_found",
                                "candidate_files": candidates,
                                "retry_with_grounding_hint": True
                            })
                            log_json("INFO", "grounding_retry_scheduled", details={"original_goal": task.title})
                            continue

                        task.status = "failed"
                        log_json("WARN", "goal_failed_invalid_output", goal=task.title)
                    else:
                        task.status = "completed"
                        log_json("INFO", "goal_completed", goal=task.title, details={"stop_reason": stop_reason})
                    
                    # Store history/result?
                    task.result = result
                    task_manager.save()
                    break # Success or handled failure, exit retry loop
                    
                except Exception as e:
                    task.status = "failed"
                    task_manager.save()
                    log_json("ERROR", "goal_execution_failed", details={"error": str(e), "goal": task.title})
                    break

        if decompose and planner_instance:
            if all(st.status == "completed" for st in root_task.subtasks):
                root_task.status = "completed"
                log_json("INFO", "goal_completed", goal=goal)
            else:
                root_task.status = "failed"
                log_json("WARN", "goal_failed", goal=goal)
            task_manager.save()

        # Prefer orchestrator-provided score; otherwise derive a simple success metric
        final_score = getattr(orchestrator, "current_score", None)
        if final_score is None:
            # Treat non-terminal or policy stop reasons as success; failures (max cycles/invalid output) score 0
            final_score = 1.0 if stop_reason not in {"MAX_CYCLES", "INVALID_OUTPUT"} else 0.0
        goal_archive.record(goal, final_score)
