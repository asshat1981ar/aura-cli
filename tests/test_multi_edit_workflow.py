import unittest
import json
import sys
import io
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import tempfile

# Ensure the project root is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set environment variable before any AURA imports to ensure isolation
os.environ["AURA_SKIP_CHDIR"] = "1"

from core.hybrid_loop import HybridClosedLoop
from core.git_tools import GitTools
from memory.brain import Brain
from core.model_adapter import ModelAdapter
from core.file_tools import replace_code, OldCodeNotFoundError, FileToolsError # Import for direct mocking

class TestMultiEditWorkflow(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = Path(os.getcwd()).absolute() / f"test_workspace_{os.getpid()}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.original_cwd = os.getcwd()
        os.environ["AURA_SKIP_CHDIR"] = "1"
        os.chdir(self.test_dir) # Change CWD to the temporary directory

        # Create dummy test files
        (self.test_dir / "file1.txt").write_text("initial content 1")
        (self.test_dir / "file2.txt").write_text("initial content 2")
        (self.test_dir / "file3.txt").write_text("initial content 3")

        # Mock dependencies for HybridClosedLoop (used by main_cli implicitly)
        self.mock_model = MagicMock(spec=ModelAdapter)
        self.mock_brain = MagicMock(spec=Brain)
        self.mock_git = MagicMock(spec=GitTools)

        # Mock sys.stdout for capturing log_json output
        self.original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Mock sys.stdin for CLI input
        self.original_stdin = sys.stdin
        sys.stdin = io.StringIO()

        # Mock sys.argv for argparse in main.py
        self.original_argv = sys.argv

    def tearDown(self):
        # Restore original CWD
        os.chdir(self.original_cwd)
        if "AURA_SKIP_CHDIR" in os.environ:
            del os.environ["AURA_SKIP_CHDIR"]
        # Clean up the temporary directory and its contents
        import shutil
        for item in self.test_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        self.test_dir.rmdir()

        # Restore stdout, stdin, argv
        sys.stdout = self.original_stdout
        sys.stdin = self.original_stdin
        sys.argv = self.original_argv

    def _get_captured_logs(self):
        output = sys.stdout.getvalue()
        entries = []
        for line in output.strip().split('\n'):
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass # Ignore non-JSON lines
        return entries
    
    def _simulate_cli_commands(self, commands: list):
        sys.stdin.seek(0) # Reset stdin buffer
        sys.stdin.truncate(0)
        sys.stdin.write("\n".join(commands) + "\n")
        sys.stdin.seek(0) # Rewind to beginning for main_cli to read

    @patch('core.task_handler.replace_code')
    @patch('cli.cli_main.GitTools') # Patch GitTools from cli.cli_main
    def test_old_implement_format_works(self, mock_main_GitTools, mock_replace_code):
        # mock_replace_code.return_value = None # Assume success

        # Mock methods of the GitTools instance that main.py initializes
        mock_git_instance = MagicMock()
        mock_git_instance.stash.return_value = None
        mock_git_instance.commit_all.return_value = None
        mock_git_instance.rollback_last_commit.return_value = None
        mock_git_instance.stash_pop.return_value = None
        mock_main_GitTools.return_value = mock_git_instance # Ensure GitTools returns our mock

        model_response = {
            "DEFINE": "Test old format.",
            "PLAN": "Plan.",
            "IMPLEMENT": {
                "file_path": "file1.txt",
                "old_code": "initial content 1",
                "new_code": "updated content 1"
            },
            "TEST": "Test.",
            "CRITIQUE": {"performance_score": 9, "stability_score": 9, "security_score": 9, "elegance_score": 9, "weaknesses": []},
            "IMPROVE": "Improve.",
            "VERSION": "Old format test",
            "SUMMARY": "Old format summary."
        }
        self.mock_model.respond.return_value = json.dumps(model_response)

        # Mock the HybridClosedLoop to return our specific JSON immediately
        # Patch the actual loop.run from main's perspective
        with patch('cli.cli_main.HybridClosedLoop.run') as mock_loop_run:
            mock_loop_run.return_value = json.dumps({
                **model_response,
                "FINAL_STATUS": "Optimization converged at 9.0 with Robust Confirmation."
            })
            
            # Simulate CLI commands
            self._simulate_cli_commands(["add Test old format", "run", "exit"])
            sys.argv = ['main.py'] # Reset argv for main_cli call
            from main import main as main_cli
            main_cli(project_root_override=self.test_dir)
        
        # Verify replace_code was called once with correct arguments
        mock_replace_code.assert_called_once_with(
            str(self.test_dir / "file1.txt"), "initial content 1", "updated content 1", overwrite_file=False
        )
        
        logs = self._get_captured_logs()
        self.assertTrue(any(e.get("event") == "applying_code_change" and e.get("details", {}).get("file") == "file1.txt" for e in logs))
        self.assertTrue(any(e.get("event") == "goal_completed" for e in logs))

    @patch('core.task_handler.replace_code')
    @patch('cli.cli_main.GitTools') # Patch GitTools from cli.cli_main
    def test_new_implement_format_works(self, mock_main_GitTools, mock_replace_code):
        mock_replace_code.return_value = None # Assume success

        # Mock methods of the GitTools instance that main.py initializes
        mock_git_instance = MagicMock()
        mock_git_instance.stash.return_value = None
        mock_git_instance.commit_all.return_value = None
        mock_git_instance.rollback_last_commit.return_value = None
        mock_git_instance.stash_pop.return_value = None
        mock_main_GitTools.return_value = mock_git_instance # Ensure GitTools returns our mock

        model_response = {
            "DEFINE": "Test new format.",
            "PLAN": "Plan.",
            "IMPLEMENT": {
                "changes": [
                    {"file_path": "file1.txt", "old_code": "initial content 1", "new_code": "new content for 1"},
                    {"file_path": "file2.txt", "old_code": "initial content 2", "new_code": "new content for 2", "overwrite_file": True}
                ]
            },
            "TEST": "Test.",
            "CRITIQUE": {"performance_score": 9, "stability_score": 9, "security_score": 9, "elegance_score": 9, "weaknesses": []},
            "IMPROVE": "Improve.",
            "VERSION": "New format test",
            "SUMMARY": "New format summary."
        }
        self.mock_model.respond.return_value = json.dumps(model_response)

        with patch('cli.cli_main.HybridClosedLoop.run') as mock_loop_run:
            mock_loop_run.return_value = json.dumps({
                **model_response,
                "FINAL_STATUS": "Optimization converged at 9.0 with Robust Confirmation."
            })
            
            self._simulate_cli_commands(["add Test new format", "run", "exit"])
            sys.argv = ['main.py']
            from main import main as main_cli
            main_cli(project_root_override=self.test_dir)
        
        # Verify replace_code was called twice with correct arguments
        calls = mock_replace_code.call_args_list
        self.assertEqual(len(calls), 2)
        
        self.assertEqual(calls[0].args, (str(self.test_dir / "file1.txt"), "initial content 1", "new content for 1"))
        self.assertEqual(calls[0].kwargs, {"overwrite_file": False})

        self.assertEqual(calls[1].args, (str(self.test_dir / "file2.txt"), "initial content 2", "new content for 2"))
        self.assertEqual(calls[1].kwargs, {"overwrite_file": True})

        logs = self._get_captured_logs()
        self.assertTrue(any(e.get("event") == "applying_code_change" and e.get("details", {}).get("file") == "file1.txt" for e in logs))
        self.assertTrue(any(e.get("event") == "applying_code_change" and e.get("details", {}).get("file") == "file2.txt" for e in logs))
        self.assertTrue(any(e.get("event") == "goal_completed" for e in logs))

    @patch('core.task_handler.replace_code')
    @patch('cli.cli_main.GitTools') # Patch GitTools from cli.cli_main
    def test_multi_change_failure_aborts_and_logs_regression(self, mock_main_GitTools, mock_replace_code):
        # Mock methods of the GitTools instance that main.py initializes
        mock_git_instance = MagicMock()
        mock_git_instance.stash.return_value = None
        mock_git_instance.commit_all.return_value = None
        mock_git_instance.rollback_last_commit.return_value = None
        mock_git_instance.stash_pop.return_value = None
        mock_main_GitTools.return_value = mock_git_instance # Ensure GitTools returns our mock
        # First call succeeds, second call fails
        mock_replace_code.side_effect = [
            None, # First call succeeds
            OldCodeNotFoundError("Simulated old code not found") # Second call fails
        ]

        model_response = {
            "DEFINE": "Test multi-change failure.",
            "PLAN": "Plan.",
            "IMPLEMENT": {
                "changes": [
                    {"file_path": "file1.txt", "old_code": "initial content 1", "new_code": "first change ok"},
                    {"file_path": "file2.txt", "old_code": "non-existent", "new_code": "second change fails"},
                    {"file_path": "file3.txt", "old_code": "initial content 3", "new_code": "third change never applied"}
                ]
            },
            "TEST": "Test.",
            "CRITIQUE": {"performance_score": 5, "stability_score": 5, "security_score": 5, "elegance_score": 5, "weaknesses": ["Simulated failure"]},
            "IMPROVE": "Improve.",
            "VERSION": "Multi-change failure test",
            "SUMMARY": "Multi-change failure summary."
        }
        self.mock_model.respond.return_value = json.dumps(model_response)

        with patch('cli.cli_main.HybridClosedLoop.run') as mock_loop_run:
            # We want to check the internal state of loop.regression_count or similar for failure
            # HybridClosedLoop.run will return the status, which main.py passes through.
            # We need to ensure that the loop in main.py correctly handles the failure to apply changes.
            # HybridClosedLoop.run does not explicitly handle replace_code failures, main.py does.
            # So, we expect main.py to mark a regression or prevent further execution based on the replace_code exception.
            
            # The HybridClosedLoop returns a JSON, and its error handling for model response takes precedence.
            # If main.py catches replace_code error, it sets changes_applied_successfully = False.
            # This should then lead to the goal being terminated without convergence.
            mock_loop_run.return_value = json.dumps({
                **model_response,
                "STATUS": "Continuing evolution (Score: 5.0, Stable Convergence Count: 0)" # Simulate initial non-converged status
            })
            
            self._simulate_cli_commands(["add Test multi-change failure", "run", "exit"])
            sys.argv = ['main.py']
            from main import main as main_cli
            main_cli(project_root_override=self.test_dir)
        
        # Verify replace_code was called only for the first two changes (first succeeds, second fails)
        calls = mock_replace_code.call_args_list
        self.assertEqual(len(calls), 2)
        
        self.assertEqual(calls[0].args, (str(self.test_dir / "file1.txt"), "initial content 1", "first change ok"))
        self.assertEqual(calls[1].args, (str(self.test_dir / "file2.txt"), "non-existent", "second change fails"))

        logs = self._get_captured_logs()
        # Verify that error for old_code_not_found was logged
        self.assertTrue(any(e.get("event") == "old_code_not_found" and e.get("level") == "ERROR" for e in logs))
        # Verify that the goal terminated without convergence
        self.assertTrue(any(e.get("event") == "goal_terminated_without_convergence" and e.get("level") == "WARN" for e in logs))
        # Verify that the third change was NOT attempted
        self.assertFalse(any(e.get("details", {}).get("file") == "file3.txt" for e in logs))

if __name__ == '__main__':
    unittest.main()