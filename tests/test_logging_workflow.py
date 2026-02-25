import unittest
import json
import sys
import io
from unittest.mock import MagicMock, patch
from pathlib import Path
import os

# Ensure the project root is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hybrid_loop import HybridClosedLoop
from core.git_tools import GitTools, GitToolsError
from git.exc import GitCommandError # Import GitCommandError
from memory.brain import Brain
from core.model_adapter import ModelAdapter
from core.logging_utils import log_json # Import for direct testing if needed

class TestLoggingWorkflow(unittest.TestCase):

    def setUp(self):
        # Mock dependencies for HybridClosedLoop
        self.mock_model = MagicMock(spec=ModelAdapter)
        self.mock_brain = MagicMock(spec=Brain)
        self.mock_git_tools_instance = MagicMock(spec=GitTools) # Mock a GitTools instance
        
        # Instantiate HybridClosedLoop with mocks
        self.loop = HybridClosedLoop(self.mock_model, self.mock_brain, self.mock_git_tools_instance)

        # Define a sample model response for mocking
        self.sample_model_response = {
            "DEFINE": "Define phase output.",
            "PLAN": "Plan phase output.",
            "IMPLEMENT": {
                "file_path": "test_file.py",
                "old_code": "def old_func(): pass",
                "new_code": "def new_func(): pass"
            },
            "TEST": "Test phase output.",
            "CRITIQUE": {
                "performance_score": 10,
                "stability_score": 10,
                "security_score": 10,
                "elegance_score": 10,
                "weaknesses": ["Minor weakness identified."]
            },
            "IMPROVE": "Improve phase output.",
            "VERSION": "Logging test commit message.",
            "SUMMARY": "Summary of logging test iteration."
        }
        self.mock_model.respond.return_value = json.dumps(self.sample_model_response)

        # Mock initial state for snapshot
        self.mock_brain.recall_all.return_value = []
        self.mock_brain.recall_weaknesses.return_value = []

        # Capture stdout
        self.original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def tearDown(self):
        # Restore stdout
        sys.stdout = self.original_stdout

    def get_log_entries(self):
        output = sys.stdout.getvalue()
        entries = []
        for line in output.strip().split('\n'):
            if line: # Ensure not to process empty lines
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Ignore non-JSON lines (e.g., from external subprocesses like pytest itself)
                    pass
        return entries

    def test_hybrid_loop_logging_in_normal_run(self):
        goal = "Test logging normal run"
        
        # Simulate convergence over 3 iterations for HybridClosedLoop
        self.mock_model.respond.side_effect = [
            json.dumps(self.sample_model_response),
            json.dumps(self.sample_model_response),
            json.dumps(self.sample_model_response)
        ]
        
        # Mock git methods to simulate successful operations
        self.mock_git_tools_instance.stash.return_value = None
        self.mock_git_tools_instance.commit_all.return_value = None
        self.mock_git_tools_instance.stash_pop.return_value = None

        # Run in normal mode for 3 cycles to trigger convergence
        for _ in range(3):
            self.loop.run(goal, dry_run=False)
        
        log_entries = self.get_log_entries()

        # Check for model response parsed log
        self.assertTrue(any(e.get("event") == "model_response_parsed" and e.get("level") == "INFO" for e in log_entries))
        
        # Check for git events (stash, commit, stash_pop)
        self.assertTrue(any(e.get("event") == "git_stashed" and e.get("level") == "INFO" for e in log_entries))
        self.assertTrue(any(e.get("event") == "git_committed" and e.get("level") == "INFO" for e in log_entries))
        self.assertTrue(any(e.get("event") == "git_stash_popped" and e.get("level") == "INFO" for e in log_entries))

        # Check for brain.remembered log
        # remember is called twice per run: for goal and for structured_response.
        # So it should be called 6 times for 3 iterations.
        self.assertEqual(self.mock_brain.remember.call_count, 6)

        # Check for convergence log
        self.assertTrue(any(e.get("event") == "optimization_converged" and e.get("level") == "INFO" for e in log_entries))
        
        # Check for regression (should not happen in this test)
        self.assertFalse(any(e.get("event") == "regression_detected" for e in log_entries))

    def test_hybrid_loop_logging_in_dry_run(self):
        goal = "Test logging dry run"
        
        self.mock_model.respond.side_effect = [
            json.dumps(self.sample_model_response),
            json.dumps(self.sample_model_response),
            json.dumps(self.sample_model_response)
        ]

        # Run in dry-run mode for 3 cycles to trigger convergence
        for _ in range(3):
            self.loop.run(goal, dry_run=True)
        
        log_entries = self.get_log_entries()

        # Check for model response parsed log
        self.assertTrue(any(e.get("event") == "model_response_parsed" and e.get("level") == "INFO" for e in log_entries))
        
        # Check for skipped git events
        self.assertTrue(any(e.get("event") == "git_stash_skipped" and e.get("level") == "INFO" for e in log_entries))
        self.assertTrue(any(e.get("event") == "git_commit_skipped" and e.get("level") == "INFO" for e in log_entries))
        self.assertTrue(any(e.get("event") == "git_stash_pop_skipped" and e.get("level") == "INFO" for e in log_entries)) # For both after commit and no improvement

        # Check for brain.remembered log
        # remember is called twice per run: for goal and for structured_response.
        # So it should be called 6 times for 3 iterations.
        self.assertEqual(self.mock_brain.remember.call_count, 6)

        # Check for convergence log
        self.assertTrue(any(e.get("event") == "optimization_converged" and e.get("level") == "INFO" for e in log_entries))

    @patch('core.git_tools.Repo') # Patch core.git_tools.Repo
    def test_git_tools_logging_commit(self, MockRepo):
        # Clear logs from previous tests that might have run on same stdout
        sys.stdout = io.StringIO()

        mock_repo_instance = MockRepo.return_value
        mock_repo_instance.is_dirty.return_value = True
        mock_repo_instance.untracked_files = []
        
        # Mock the git object and its methods that GitTools calls
        mock_repo_instance.git.add.return_value = None
        mock_repo_instance.index.commit.return_value = None

        # Instantiate actual GitTools, which will now use the mocked Repo
        git_tools_instance = GitTools(repo_path=".") 
        
        git_tools_instance.commit_all("Test Commit Message")
        log_entries = self.get_log_entries()
        
        self.assertTrue(any(e.get("event") == "git_committed" and e.get("level") == "INFO" for e in log_entries))
        self.assertTrue(any(e.get("details", {}).get("message") == "Test Commit Message" for e in log_entries))

    @patch('core.git_tools.Repo') # Patch core.git_tools.Repo
    def test_git_tools_logging_stash_pop_rollback_error(self, MockRepo):
        # Clear logs from previous tests that might have run on same stdout
        sys.stdout = io.StringIO()
        
        mock_repo_instance = MockRepo.return_value
        # Mock behavior needed for GitTools.stash/stash_pop/rollback
        mock_repo_instance.is_dirty.return_value = True # For stash
        
        # Mock the git object and its methods that GitTools calls
        mock_git_cmd = MagicMock()
        mock_repo_instance.git = mock_git_cmd # Assign mock_git_cmd to mock_repo_instance.git

        # For stash error: make self.repo.git.stash raise GitCommandError
        mock_git_cmd.stash.side_effect = GitCommandError("git stash", 1, "Simulated stash error output")

        mock_repo_instance.head.commit.parents = [MagicMock()] # Simulate parent commit for rollback
        
        # For rollback error: make self.repo.git.reset raise GitCommandError
        mock_git_cmd.reset.side_effect = GitCommandError("git reset", 1, "Simulated rollback error output")
        
        # Instantiate actual GitTools, which will now use the mocked Repo
        git_tools_instance = GitTools(repo_path=".") 
        
        # Test stash error logging
        with self.assertRaises(GitToolsError): # Expect GitToolsError
            git_tools_instance.stash("Error stash")
        log_entries = self.get_log_entries()
        self.assertTrue(any(e.get("event") == "git_stash_failed" and e.get("level") == "ERROR" for e in log_entries))

        # Clear logs again
        sys.stdout = io.StringIO()
        
        # Test rollback error logging
        with self.assertRaises(GitToolsError): # Expect GitToolsError
            git_tools_instance.rollback_last_commit("Error rollback")
        log_entries = self.get_log_entries()
        self.assertTrue(any(e.get("event") == "git_rollback_failed" and e.get("level") == "ERROR" for e in log_entries))