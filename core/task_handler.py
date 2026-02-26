import json
import time
from pathlib import Path
from core.logging_utils import log_json
from core.file_tools import replace_code, OldCodeNotFoundError
from task_manager import TaskManager, Task

def _check_project_writability(project_root: Path) -> bool:
    """
    Checks if the project directory is writable by attempting to create a temporary file.
    """
    try:
        test_file = project_root / ".aura_write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception as e:
        log_json("ERROR", "project_not_writable", details={"error": str(e), "path": str(project_root)})
        return False

def run_goals_loop(args, goal_queue, loop, debugger_instance, planner_instance, goal_archive, project_root, decompose=False):
    """
    Processes all goals in the queue using the hierarchical TaskManager.
    """
    task_manager = TaskManager()
    
    while goal_queue.has_goals():
        goal = goal_queue.next()
        log_json("INFO", "processing_goal", goal=goal)
        
        if decompose and planner_instance:
            # Decompose goal into subtasks
            root_task = task_manager.decompose_goal(goal, planner_instance)
            tasks_to_process = root_task.subtasks
        else:
            # Direct execution
            root_task = Task(id=f"goal_{int(time.time())}", title=goal)
            task_manager.add_task(root_task)
            tasks_to_process = [root_task]
        
        # Process each task
        for task in tasks_to_process:
            if task != root_task:
                log_json("INFO", "executing_subtask", details={"task_id": task.id, "title": task.title})
            
            task.status = "in_progress"
            task_manager.save()
            
            converged = False
            loop.previous_score = 0
            loop.regression_count = 0
            loop.stable_convergence_count = 0
            
            cycle_count = 0
            changes_applied_successfully = True
            while not converged and changes_applied_successfully:
                cycle_count += 1
                log_json("INFO", "cycle_start", goal=task.title, details={"cycle_count": cycle_count})
                
                loop_result_json_str = loop.run(task.title, dry_run=getattr(args, 'dry_run', False))
                
                try:
                    result_json = json.loads(loop_result_json_str)
                except json.JSONDecodeError:
                    log_json("ERROR", "invalid_json_from_loop", goal=task.title)
                    break
                
                # Handle implementation if present
                implement_data = result_json.get("IMPLEMENT")
                if implement_data:
                    changes_to_apply = []
                    if isinstance(implement_data, dict):
                        if all(k in implement_data for k in ["file_path", "old_code", "new_code"]):
                            changes_to_apply.append(implement_data)
                        elif "changes" in implement_data and isinstance(implement_data["changes"], list):
                            changes_to_apply.extend(implement_data["changes"])
                    
                    for change in changes_to_apply:
                        file_path = change.get("file_path")
                        old_code = change.get("old_code")
                        new_code = change.get("new_code")
                        overwrite_file = change.get("overwrite_file", False)
                        
                        if all([file_path is not None, old_code is not None, new_code is not None]):
                            if not getattr(args, 'dry_run', False):
                                try:
                                    full_target_path = str(project_root / file_path)
                                    log_json("INFO", "applying_code_change", goal=task.title, details={"file": file_path})
                                    replace_code(full_target_path, old_code, new_code, overwrite_file=overwrite_file)
                                except OldCodeNotFoundError as e:
                                    log_json("ERROR", "old_code_not_found", goal=task.title, details={"error": str(e), "file": file_path})
                                    changes_applied_successfully = False
                                    break
                                except Exception as e:
                                    log_json("ERROR", "apply_change_failed", goal=task.title, details={"error": str(e), "file": file_path})
                                    changes_applied_successfully = False
                                    break
                            else:
                                log_json("INFO", "replace_code_skipped", goal=task.title, details={"reason": "dry_run", "file": file_path})

                if "FINAL_STATUS" in result_json:
                    converged = True
                    task.status = "completed"
                    task.result = result_json["FINAL_STATUS"]
                    task_manager.save()
                    log_json("INFO", "goal_completed" if task == root_task else "subtask_completed", goal=task.title, details={"status": result_json["FINAL_STATUS"]})
                
                if cycle_count > 10: # Safety break
                    log_json("WARN", "cycle_limit_reached", goal=task.title)
                    break
            
            if not converged and not changes_applied_successfully:
                task.status = "failed"
                task_manager.save()
                log_json("WARN", "goal_terminated_without_convergence" if task == root_task else "subtask_failed", goal=task.title)
                break
        
        # Update root goal status if it was decomposed
        if decompose and planner_instance:
            if all(st.status == "completed" for st in root_task.subtasks):
                root_task.status = "completed"
                log_json("INFO", "goal_completed", goal=goal)
            else:
                root_task.status = "failed"
                log_json("WARN", "goal_failed", goal=goal)
            task_manager.save()

        # Record result in goal_archive
        final_score = loop.current_score if hasattr(loop, 'current_score') else 0.0
        goal_archive.record(goal, final_score)
