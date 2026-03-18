import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any

from core.logging_utils import log_json
from core.model_adapter import ModelAdapter
from core.git_tools import GitTools
from agents.codex_agent import CodexAgent
from agents.critic import CriticAgent


class CodexManager:
    """
    Orchestrates multiple CodexAgents to perform parallel software engineering
    tasks and merges their results.
    """

    def __init__(self, model_adapter: ModelAdapter, git_tools: GitTools, project_root: str, brain: Any = None):
        self.model = model_adapter
        self.git = git_tools
        self.project_root = project_root
        self.brain = brain
        self.critic = CriticAgent(brain, model_adapter) if brain else None

    def run(self, input_data: Dict) -> Dict:
        """
        Agent-compatible run method.
        """
        goal = input_data.get("task", "")
        task_bundle = input_data.get("task_bundle", {})
        return self.decompose_and_run_parallel(goal, task_bundle)

    def decompose_and_run_parallel(self, goal: str, task_bundle: Dict) -> Dict:
        """
        Decomposes the task bundle into parallel sub-tasks and executes them.
        """
        log_json("INFO", "codex_manager_starting_parallel_execution", details={"goal": goal})

        # 1. Identify sub-tasks that can run in parallel
        # For now, we'll use the 'tasks' list in the task_bundle if it exists, 
        # or ask the LLM to decompose the goal into parallel-ready sub-tasks.
        sub_tasks = self._get_parallel_tasks(goal, task_bundle)
        if not sub_tasks:
            log_json("WARN", "codex_manager_no_parallel_tasks_found")
            return {"changes": []}

        # 2. Dispatch to CodexAgents in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sub_tasks)) as executor:
            future_to_task = {
                executor.submit(self._run_single_agent_task, i, task): task 
                for i, task in enumerate(sub_tasks)
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    results.append(future.result())
                except Exception as exc:
                    log_json("ERROR", "codex_manager_subtask_failed", details={"error": str(exc)})

        # 3. Merge results from all agents
        merged_changes = self._merge_results(results)
        
        # 4. Review the merged changes
        if self.critic and merged_changes:
            return {"changes": self._review_and_fix(goal, merged_changes)}
            
        return {"changes": merged_changes}

    def _get_parallel_tasks(self, goal: str, task_bundle: Dict) -> List[str]:
        """
        Asks the LLM to decompose the goal into 2-4 parallel-executable sub-tasks.
        """
        prompt = f"""
You are a manager AI orchestrating parallel Codex sub-agents.
Goal: {goal}
Current task bundle info: {task_bundle.get('tasks', [])}

Break this goal into 2 to 4 strictly parallel-executable sub-tasks.
Each sub-task should be able to run in its own git worktree without immediate 
functional dependencies on other sub-tasks' changes (they will be merged later).

Return ONLY a JSON array of strings (sub-task descriptions).
"""
        try:
            from core.file_tools import _aura_safe_loads
            raw = self.model.respond(prompt)
            parsed = _aura_safe_loads(raw, "codex_manager_decomposer")
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            log_json("WARN", "codex_manager_decomposition_failed", details={"error": str(e)})
        
        # Fallback: if we have multiple tasks in the bundle, use them
        if "tasks" in task_bundle and isinstance(task_bundle["tasks"], list):
            return [t.get("intent", goal) for t in task_bundle["tasks"]]
        
        return [goal] # Single task fallback

    def _run_single_agent_task(self, index: int, task_description: str) -> Dict:
        """Runs a single sub-task using a CodexAgent."""
        agent = CodexAgent(self.model, self.git, self.project_root)
        try:
            return agent.run_task(f"sub-{index}", task_description)
        finally:
            # We don't cleanup immediately so we can inspect changed files in worktrees if needed.
            # But in a real production loop, we might want to.
            pass

    def _merge_results(self, results: List[Dict]) -> List[Dict]:
        """
        Collects all changed files from sub-agent worktrees and combines them.
        """
        merged_changes = []
        for res in results:
            if res.get("status") == "success":
                worktree_path = Path(res["worktree_path"])
                for file_rel_path in res.get("changed_files", []):
                    full_path = worktree_path / file_rel_path
                    if full_path.exists() and full_path.is_file():
                        content = full_path.read_text(encoding="utf-8", errors="ignore")
                        merged_changes.append({
                            "file_path": file_rel_path,
                            "old_code": "", # Overwrite policy
                            "new_code": content,
                            "overwrite_file": True
                        })
        return merged_changes

    def _review_and_fix(self, goal: str, changes: List[Dict]) -> List[Dict]:
        """
        Uses CriticAgent to review the merged changes and attempts fixes if needed.
        """
        final_changes = []
        for change in changes:
            file_path = change["file_path"]
            code = change["new_code"]
            
            feedback = self.critic.critique_code(goal, code, f"Reviewing changes for {file_path}")
            # Heuristic: if critic is happy, keep it. 
            # In a more advanced implementation, the critic would return structured pass/fail.
            if any(ok in feedback.lower() for ok in ["pass", "no issues", "looks good", "correct"]):
                final_changes.append(change)
            else:
                log_json("WARN", "codex_manager_review_feedback", details={"file": file_path, "feedback": feedback})
                fixed_code = self._attempt_fix(file_path, code, feedback)
                final_changes.append({
                    "file_path": file_path,
                    "old_code": "",
                    "new_code": fixed_code,
                    "overwrite_file": True
                })
        return final_changes

    def _attempt_fix(self, file_path: str, code: str, feedback: str) -> str:
        """
        Asks Codex to fix the code based on the feedback.
        """
        prompt = f"""
Fix the following Python code for {file_path}.

Current Code:
```python
{code}
```

Review Feedback:
{feedback}

Return ONLY the corrected full code. No conversational text.
"""
        try:
            fixed = self.model.call_codex(prompt)
            # Basic cleanup in case it returned markdown
            if fixed.startswith("```python"):
                fixed = fixed.split("\n", 1)[1].rsplit("\n", 1)[0]
            elif fixed.startswith("```"):
                fixed = fixed.split("\n", 1)[1].rsplit("\n", 1)[0]
            return fixed.strip()
        except Exception as e:
            log_json("ERROR", "codex_manager_fix_attempt_failed", details={"error": str(e)})
            return code
