import unittest
import json
import sys
import io
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ensure the project root is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.git_tools import GitTools, GitToolsError, GitCommitError
from git.exc import GitCommandError # Import GitCommandError
from core.logging_utils import log_json # Imported for parsing verification

class TestGitToolsEnhancements(unittest.TestCase):

    def setUp(self):
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
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass # Ignore non-JSON lines
        return entries

    @patch('core.git_tools.Repo')
    def test_commit_all_no_changes_is_no_op(self, MockRepo):
        mock_repo_instance = MockRepo.return_value
        mock_repo_instance.is_dirty.return_value = False # Simulate no changes
        mock_repo_instance.untracked_files = [] # Simulate no untracked files

        git_tools_instance = GitTools(repo_path=".")
        git_tools_instance.commit_all("Test message for no changes")

        # Assert that git add/commit methods were NOT called
        mock_repo_instance.git.add.assert_not_called()
        mock_repo_instance.index.commit.assert_not_called()

        # Assert that the appropriate log message was emitted
        log_entries = self.get_log_entries()
        self.assertTrue(any(e.get("event") == "git_no_changes_to_commit" and e.get("level") == "INFO" for e in log_entries))
        self.assertFalse(any(e.get("event") == "git_committed" for e in log_entries))

    @patch('core.git_tools.Repo')
    def test_commit_all_with_changes(self, MockRepo):
        mock_repo_instance = MockRepo.return_value
        mock_repo_instance.is_dirty.return_value = True # Simulate changes present
        mock_repo_instance.untracked_files = ["new_file.txt"] # Simulate untracked files

        # Mock the git object and its methods
        mock_git_cmd = MagicMock()
        mock_repo_instance.git = mock_git_cmd
        mock_repo_instance.index.commit.return_value = None # Commit succeeds

        commit_message = "Test message with changes"
        git_tools_instance = GitTools(repo_path=".")
        git_tools_instance.commit_all(commit_message)

        # Assert that git add/commit methods WERE called
        mock_repo_instance.git.add.assert_any_call(["new_file.txt"]) # For untracked files
        mock_repo_instance.git.add.assert_any_call(A=True) # For staging all changes
        mock_repo_instance.index.commit.assert_called_once_with(commit_message)

        # Assert that the appropriate log message was emitted
        log_entries = self.get_log_entries()
        self.assertTrue(any(e.get("event") == "git_add_untracked" and e.get("level") == "INFO" for e in log_entries))
        self.assertTrue(any(e.get("event") == "git_committed" and e.get("level") == "INFO" and e.get("details", {}).get("message") == commit_message for e in log_entries))
        self.assertFalse(any(e.get("event") == "git_no_changes_to_commit" for e in log_entries))

    @patch('core.git_tools.Repo')
    def test_commit_all_git_command_error(self, MockRepo):
        mock_repo_instance = MockRepo.return_value
        mock_repo_instance.is_dirty.return_value = True
        mock_repo_instance.untracked_files = []

        # Make commit fail
        mock_repo_instance.index.commit.side_effect = GitCommandError("Simulated commit command", 1, "Simulated commit failure output")

        git_tools_instance = GitTools(repo_path=".")
        with self.assertRaises(GitCommitError):
            git_tools_instance.commit_all("Failing commit")
        
        # Assert log for commit failure
        log_entries = self.get_log_entries()
        self.assertTrue(any(e.get("event") == "git_commit_failed" and e.get("level") == "ERROR" for e in log_entries))

if __name__ == '__main__':
    unittest.main()
