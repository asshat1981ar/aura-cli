import unittest
import os
import shutil
import tempfile
from pathlib import Path
from core.git_tools import GitTools
from agents.codex_agent import CodexAgent
from core.model_adapter import ModelAdapter


class TestCodexSubagents(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for the test repo
        self.test_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.test_dir / "repo"
        self.repo_dir.mkdir()
        
        # Initialize a git repo
        from git import Repo
        self.repo = Repo.init(self.repo_dir)
        # Create an initial commit
        readme = self.repo_dir / "README.md"
        readme.write_text("Initial commit")
        self.repo.index.add(["README.md"])
        self.repo.index.commit("Initial commit")
        
        self.git_tools = GitTools(str(self.repo_dir))

    def tearDown(self):
        # Cleanup
        shutil.rmtree(self.test_dir, ignore_errors=True)
        # Also cleanup temp aura_worktrees if any were created
        shutil.rmtree(Path(tempfile.gettempdir()) / "aura_worktrees", ignore_errors=True)

    def test_git_tools_worktree(self):
        """Test GitTools worktree operations."""
        wt_path = self.test_dir / "wt-1"
        branch_name = "test-branch"
        
        # 1. Create worktree
        # Directly use worktree add -b to create and checkout in one step.
        self.git_tools.repo.git.worktree('add', '-b', branch_name, str(wt_path), 'HEAD')
        
        self.assertTrue(wt_path.exists())
        self.assertTrue((wt_path / ".git").exists())
        
        # 2. List worktrees
        worktrees = self.git_tools.list_worktrees()
        paths = [wt['path'] for wt in worktrees]
        # On some systems paths might be resolved/absolute
        self.assertTrue(any(str(wt_path) in p for p in paths))
        
        # 3. Remove worktree
        self.git_tools.remove_worktree(str(wt_path), force=True)
        # Note: remove_worktree doesn't necessarily delete the directory if it has untracked files
        # but the worktree should be gone from git list.
        worktrees_after = self.git_tools.list_worktrees()
        paths_after = [wt['path'] for wt in worktrees_after]
        self.assertFalse(any(str(wt_path) in p for p in paths_after))

    def test_codex_agent_isolation(self):
        """Test CodexAgent runs in isolation."""
        # Mock ModelAdapter
        class MockModel:
            def call_codex(self, prompt):
                # Simulate Codex making a change
                (Path(os.getcwd()) / "new_file.py").write_text("print('hello')")
                return "Created new_file.py"
        
        model = MockModel()
        agent = CodexAgent(model, self.git_tools, str(self.repo_dir))
        
        result = agent.run_task("test-1", "Create new_file.py")
        
        self.assertEqual(result["status"], "success")
        self.assertIn("new_file.py", result["changed_files"])
        
        # Verify the file exists in the worktree but NOT in the main repo
        wt_path = Path(result["worktree_path"])
        self.assertTrue((wt_path / "new_file.py").exists())
        self.assertFalse((self.repo_dir / "new_file.py").exists())
        
        agent.cleanup()

    def test_codex_manager_parallel(self):
        """Test CodexManager parallel execution and merging."""
        from core.codex_manager import CodexManager
        
        class MockModel:
            def respond(self, prompt):
                # Simulate decomposition into 2 tasks
                if "Break this goal into 2 to 4" in prompt:
                    return '["Task A", "Task B"]'
                return "OK"
            
            def call_codex(self, prompt):
                # Simulate Codex making changes based on task
                cwd = Path(os.getcwd())
                if "Task A" in prompt:
                    (cwd / "file_a.py").write_text("content a")
                elif "Task B" in prompt:
                    (cwd / "file_b.py").write_text("content b")
                return "Done"

        model = MockModel()
        manager = CodexManager(model, self.git_tools, str(self.repo_dir))
        
        # We don't have a real brain/critic here, so review step will be skipped
        result = manager.decompose_and_run_parallel("Parallel Goal", {})
        
        self.assertEqual(len(result["changes"]), 2)
        paths = [c["file_path"] for c in result["changes"]]
        self.assertIn("file_a.py", paths)
        self.assertIn("file_b.py", paths)
        
        contents = {c["file_path"]: c["new_code"] for c in result["changes"]}
        self.assertEqual(contents["file_a.py"], "content a")
        self.assertEqual(contents["file_b.py"], "content b")

if __name__ == "__main__":
    unittest.main()
