import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from core.git_tools import GitTools
from core.logging_utils import log_json
from core.model_adapter import ModelAdapter


class CodexAgent:
    """
    A specialized worker agent that uses OpenAI Codex to perform software engineering
    tasks in isolated git worktrees.
    """

    def __init__(self, model_adapter: ModelAdapter, git_tools: GitTools, project_root: str):
        self.model = model_adapter
        self.git = git_tools
        self.project_root = Path(project_root).resolve()
        self.active_worktree: Optional[Path] = None
        self.active_branch: Optional[str] = None

    def _generate_branch_name(self, task_id: str) -> str:
        """Generates a unique branch name for the task."""
        import uuid
        return f"codex-task-{task_id}-{uuid.uuid4().hex[:8]}"

    def run_task(self, task_id: str, task_description: str) -> Dict:
        """
        Runs a software engineering task in an isolated git worktree.
        """
        branch_name = self._generate_branch_name(task_id)
        # Create a temp directory for the worktree
        temp_base = Path(tempfile.gettempdir()) / "aura_worktrees"
        temp_dir = temp_base / branch_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            log_json("INFO", "codex_agent_starting_task", details={"task_id": task_id, "branch": branch_name, "path": str(temp_dir)})
            
            # 1. Create worktree
            # Create a new branch directly in the worktree command instead of checking it out in main repo first.
            self.git.repo.git.worktree('add', '-b', branch_name, str(temp_dir), 'HEAD')
            self.active_worktree = temp_dir
            self.active_branch = branch_name

            # 2. Prepare the prompt for Codex
            # We should include project context, AGENTS.md guidelines, and the specific task.
            agents_md_path = self.project_root / "AGENTS.md"
            agents_guidelines = ""
            if agents_md_path.exists():
                agents_guidelines = agents_md_path.read_text()

            codex_prompt = f"""
Guidelines:
{agents_guidelines}

Task:
{task_description}

You are working in an isolated git worktree at {temp_dir}.
Perform the requested changes. If you need to create new files, do so.
Respond with a summary of your actions.
"""

            # 3. Call Codex
            # Note: ModelAdapter.call_codex uses 'codex exec' which might run shell commands or edit files.
            # We need to make sure 'codex' is aware of the current working directory being the worktree.
            # We'll temporarily change CWD.
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                codex_output = self.model.call_codex(codex_prompt)
            finally:
                os.chdir(old_cwd)

            # 4. Capture changes
            # We'll use git status/diff in the worktree to see what changed.
            changed_files = self._get_changed_files(temp_dir)
            
            log_json("INFO", "codex_agent_task_complete", details={"task_id": task_id, "changed_files": changed_files})

            return {
                "status": "success",
                "task_id": task_id,
                "branch": branch_name,
                "worktree_path": str(temp_dir),
                "codex_output": codex_output,
                "changed_files": changed_files
            }

        except Exception as e:
            log_json("ERROR", "codex_agent_task_failed", details={"task_id": task_id, "error": str(e)})
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }

    def _get_changed_files(self, worktree_path: Path) -> List[str]:
        """Returns a list of files that were modified or created in the worktree."""
        # We'll use a local Repo object for the worktree
        from git import Repo
        wt_repo = Repo(worktree_path)
        changed = []
        # Modified files
        changed.extend([item.a_path for item in wt_repo.index.diff(None)])
        # Staged files
        changed.extend([item.a_path for item in wt_repo.index.diff("HEAD")])
        # Untracked files
        changed.extend(wt_repo.untracked_files)
        return list(set(changed))

    def cleanup(self):
        """Removes the active worktree and deletes the temporary branch."""
        if self.active_worktree and self.active_worktree.exists():
            try:
                self.git.remove_worktree(str(self.active_worktree), force=True)
                shutil.rmtree(self.active_worktree, ignore_errors=True)
                log_json("INFO", "codex_agent_cleanup_worktree", details={"path": str(self.active_worktree)})
            except Exception as e:
                log_json("ERROR", "codex_agent_cleanup_failed", details={"error": str(e)})

        if self.active_branch:
            try:
                self.git.repo.git.branch('-D', self.active_branch)
                log_json("INFO", "codex_agent_cleanup_branch", details={"branch": self.active_branch})
            except Exception as e:
                log_json("ERROR", "codex_agent_cleanup_branch_failed", details={"error": str(e)})
        
        self.active_worktree = None
        self.active_branch = None
