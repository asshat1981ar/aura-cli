"""Unit tests for VerifierAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agents.verifier import VerifierAgent


class TestVerifierAgent(unittest.TestCase):
    """Test suite for VerifierAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = VerifierAgent()

    def test_init(self):
        """Test agent initialization with default timeout."""
        agent = VerifierAgent()
        self.assertEqual(agent.timeout, 120)

    def test_init_custom_timeout(self):
        """Test agent initialization with custom timeout."""
        agent = VerifierAgent(timeout=60)
        self.assertEqual(agent.timeout, 60)

    def test_capabilities(self):
        """Test agent capabilities."""
        self.assertIn("testing", VerifierAgent.capabilities)
        self.assertIn("verification", VerifierAgent.capabilities)
        self.assertIn("lint", VerifierAgent.capabilities)
        self.assertIn("quality", VerifierAgent.capabilities)
        self.assertIn("test_runner", VerifierAgent.capabilities)

    def test_name(self):
        """Test agent name."""
        self.assertEqual(VerifierAgent.name, "verify")

    # Tests for _tokenize_path

    def test_tokenize_path_basic(self):
        """Test basic path tokenization."""
        result = self.agent._tokenize_path("test_file.py")
        self.assertIn("test", result)
        self.assertIn("file", result)
        self.assertIn("py", result)

    def test_tokenize_path_empty(self):
        """Test tokenization of empty string."""
        result = self.agent._tokenize_path("")
        self.assertEqual(result, [])

    def test_tokenize_path_none(self):
        """Test tokenization of None."""
        result = self.agent._tokenize_path(None)
        self.assertEqual(result, [])

    def test_tokenize_path_single_chars_filtered(self):
        """Test that single character tokens are filtered."""
        result = self.agent._tokenize_path("a_b_c_test")
        self.assertIn("test", result)
        # Single chars should be filtered
        self.assertNotIn("a", result)

    # Tests for _available_test_files

    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.rglob")
    @patch("pathlib.Path.is_file")
    def test_available_test_files_success(self, mock_is_file, mock_rglob, mock_is_dir):
        """Test finding available test files."""
        mock_is_dir.return_value = True
        mock_is_file.return_value = True
        
        # Use absolute paths that can be made relative to /project
        mock_paths = [
            Path("/project/tests/test_one.py"),
            Path("/project/tests/test_two.py"),
        ]
        mock_rglob.return_value = mock_paths

        result = self.agent._available_test_files(Path("/project"))
        
        self.assertEqual(len(result), 2)
        self.assertIn("tests/test_one.py", result)
        self.assertIn("tests/test_two.py", result)

    @patch("pathlib.Path.is_dir")
    def test_available_test_files_no_tests_dir(self, mock_is_dir):
        """Test when tests directory doesn't exist."""
        mock_is_dir.return_value = False

        result = self.agent._available_test_files(Path("/project"))
        
        self.assertEqual(result, [])

    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.rglob")
    @patch("pathlib.Path.is_file")
    def test_available_test_files_filters_non_files(self, mock_is_file, mock_rglob, mock_is_dir):
        """Test that non-files are filtered out."""
        mock_is_dir.return_value = True
        mock_is_file.side_effect = [True, False]  # First is file, second is not
        
        # Use absolute paths that can be made relative to /project
        mock_paths = [
            Path("/project/tests/test_one.py"),
            Path("/project/tests/not_a_file.py"),
        ]
        mock_rglob.return_value = mock_paths

        result = self.agent._available_test_files(Path("/project"))
        
        self.assertEqual(len(result), 1)
        self.assertIn("tests/test_one.py", result)

    # Tests for _related_test_files

    def test_related_test_files_exact_match(self):
        """Test finding related test with exact match."""
        project_root = Path("/project")
        available_tests = ["tests/test_module.py", "tests/test_other.py"]
        
        result = self.agent._related_test_files(
            project_root, "module.py", available_tests
        )
        
        self.assertIn("tests/test_module.py", result)

    def test_related_test_files_preferred_names(self):
        """Test scoring based on preferred names."""
        project_root = Path("/project")
        available_tests = ["tests/test_module.py"]
        
        result = self.agent._related_test_files(
            project_root, "module.py", available_tests
        )
        
        self.assertEqual(result[0], "tests/test_module.py")

    def test_related_test_files_wrapper_variant(self):
        """Test matching wrapper variant test files."""
        project_root = Path("/project")
        available_tests = ["tests/test_module_wrapper.py"]
        
        result = self.agent._related_test_files(
            project_root, "module.py", available_tests
        )
        
        self.assertEqual(result[0], "tests/test_module_wrapper.py")

    def test_related_test_files_direct_match_for_tests_path(self):
        """Test direct match when file is already in tests/."""
        project_root = Path("/project")
        available_tests = ["tests/test_module.py"]
        
        with patch("pathlib.Path.is_file") as mock_is_file:
            mock_is_file.return_value = True
            result = self.agent._related_test_files(
                project_root, "tests/test_module.py", available_tests
            )
        
        self.assertEqual(result[0], "tests/test_module.py")

    def test_related_test_files_token_scoring(self):
        """Test token-based scoring."""
        project_root = Path("/project")
        available_tests = [
            "tests/test_user_auth.py",
            "tests/test_database.py",
            "tests/test_utils.py"
        ]
        
        result = self.agent._related_test_files(
            project_root, "auth/user_login.py", available_tests
        )
        
        # Should prioritize user_auth due to token overlap
        self.assertEqual(result[0], "tests/test_user_auth.py")

    def test_related_test_files_limit(self):
        """Test that result is limited."""
        project_root = Path("/project")
        available_tests = [
            "tests/test_one.py",
            "tests/test_two.py",
            "tests/test_three.py",
            "tests/test_four.py",
            "tests/test_five.py"
        ]
        
        result = self.agent._related_test_files(
            project_root, "one.py", available_tests, limit=3
        )
        
        self.assertEqual(len(result), 3)

    def test_related_test_files_empty_available(self):
        """Test with empty available tests list."""
        project_root = Path("/project")
        result = self.agent._related_test_files(
            project_root, "module.py", []
        )
        
        self.assertEqual(result, [])

    # Tests for _changed_files_from_git

    @patch("subprocess.run")
    def test_changed_files_from_git_success(self, mock_run):
        """Test successfully getting changed files from git."""
        mock_result = Mock()
        mock_result.stdout = "file1.py\nfile2.py\n"
        mock_run.return_value = mock_result

        result = self.agent._changed_files_from_git(Path("/project"))
        
        self.assertEqual(result, ["file1.py", "file2.py"])
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("agents.verifier.log_json")
    def test_changed_files_from_git_failure(self, mock_log, mock_run):
        """Test handling git command failure."""
        mock_run.side_effect = Exception("git error")

        result = self.agent._changed_files_from_git(Path("/project"))
        
        self.assertEqual(result, [])
        mock_log.assert_called_once()

    @patch("subprocess.run")
    def test_changed_files_from_git_empty_output(self, mock_run):
        """Test handling empty git output."""
        mock_result = Mock()
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        result = self.agent._changed_files_from_git(Path("/project"))
        
        self.assertEqual(result, [])

    # Tests for _changed_test_files

    @patch.object(VerifierAgent, "_changed_files_from_git")
    @patch.object(VerifierAgent, "_available_test_files")
    def test_changed_test_files_with_change_set(self, mock_available, mock_git):
        """Test getting test files from explicit change_set."""
        mock_available.return_value = ["tests/test_module.py"]
        
        change_set = {
            "changes": [
                {"file_path": "module.py"}
            ]
        }

        result = self.agent._changed_test_files(
            Path("/project"), change_set=change_set
        )
        
        self.assertIn("tests/test_module.py", result)
        mock_git.assert_not_called()  # Should not call git when change_set provided

    @patch.object(VerifierAgent, "_changed_files_from_git")
    @patch.object(VerifierAgent, "_available_test_files")
    def test_changed_test_files_fallback_to_git(self, mock_available, mock_git):
        """Test fallback to git when no change_set."""
        mock_git.return_value = ["module.py"]
        mock_available.return_value = ["tests/test_module.py"]

        result = self.agent._changed_test_files(Path("/project"))
        
        self.assertIn("tests/test_module.py", result)
        mock_git.assert_called_once()

    @patch.object(VerifierAgent, "_changed_files_from_git")
    @patch.object(VerifierAgent, "_available_test_files")
    def test_changed_test_files_empty(self, mock_available, mock_git):
        """Test when no changed files found."""
        mock_git.return_value = []

        result = self.agent._changed_test_files(Path("/project"))
        
        self.assertEqual(result, [])

    @patch.object(VerifierAgent, "_changed_files_from_git")
    @patch.object(VerifierAgent, "_available_test_files")
    def test_changed_test_files_no_available_tests(self, mock_available, mock_git):
        """Test when no test files available."""
        mock_git.return_value = ["module.py"]
        mock_available.return_value = []

        result = self.agent._changed_test_files(Path("/project"))
        
        self.assertEqual(result, [])

    @patch.object(VerifierAgent, "_changed_files_from_git")
    @patch.object(VerifierAgent, "_available_test_files")
    def test_changed_test_files_invalid_change_entry(self, mock_available, mock_git):
        """Test handling invalid change entry in change_set."""
        mock_available.return_value = []
        
        change_set = {
            "changes": [
                "not_a_dict",  # Invalid entry
                {"file_path": "module.py"}
            ]
        }

        result = self.agent._changed_test_files(
            Path("/project"), change_set=change_set
        )
        
        # Should handle gracefully without error
        self.assertIsInstance(result, list)

    # Tests for _normalize_test_command

    def test_normalize_test_command_list(self):
        """Test normalizing list command."""
        result = self.agent._normalize_test_command(["pytest -x tests/"])
        self.assertEqual(result, ["pytest", "-x", "tests/"])

    def test_normalize_test_command_string(self):
        """Test normalizing string command."""
        result = self.agent._normalize_test_command("pytest -x tests/")
        self.assertEqual(result, ["pytest", "-x", "tests/"])

    def test_normalize_test_command_empty_list(self):
        """Test normalizing empty list."""
        result = self.agent._normalize_test_command([])
        self.assertEqual(result, [])

    def test_normalize_test_command_none(self):
        """Test normalizing None."""
        result = self.agent._normalize_test_command(None)
        self.assertEqual(result, [])

    def test_normalize_test_command_empty_string(self):
        """Test normalizing empty string."""
        result = self.agent._normalize_test_command("")
        self.assertEqual(result, [])

    # Tests for _is_repo_wide_pytest_command

    def test_is_repo_wide_pytest_command_simple(self):
        """Test detecting simple pytest command."""
        result = self.agent._is_repo_wide_pytest_command(["pytest"])
        self.assertTrue(result)

    def test_is_repo_wide_pytest_command_quiet(self):
        """Test detecting pytest -q command."""
        result = self.agent._is_repo_wide_pytest_command(["pytest", "-q"])
        self.assertTrue(result)

    def test_is_repo_wide_pytest_command_python_module(self):
        """Test detecting python -m pytest command."""
        result = self.agent._is_repo_wide_pytest_command(["python", "-m", "pytest", "-q"])
        self.assertTrue(result)

    def test_is_repo_wide_pytest_command_python3_module(self):
        """Test detecting python3 -m pytest command."""
        result = self.agent._is_repo_wide_pytest_command(["python3", "-m", "pytest", "-q"])
        self.assertTrue(result)

    def test_is_repo_wide_pytest_command_specific_file(self):
        """Test that specific file commands are not repo-wide."""
        result = self.agent._is_repo_wide_pytest_command(["pytest", "test_file.py"])
        self.assertFalse(result)

    def test_is_repo_wide_pytest_command_empty(self):
        """Test empty command."""
        result = self.agent._is_repo_wide_pytest_command([])
        self.assertFalse(result)

    # Tests for run

    def test_run_dry_run(self):
        """Test dry run mode."""
        input_data = {"dry_run": True}
        
        result = self.agent.run(input_data)
        
        self.assertEqual(result["status"], "skip")
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["logs"], "dry_run")

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    def test_run_success(self, mock_sanitize, mock_run):
        """Test successful test run."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test session starts...\n1 passed"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        input_data = {"project_root": "/project"}
        result = self.agent.run(input_data)
        
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["failures"], [])
        self.assertIn("1 passed", result["logs"])
        mock_sanitize.assert_called_once()

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    def test_run_failure(self, mock_sanitize, mock_run):
        """Test failed test run."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "test session starts...\n1 failed"
        mock_result.stderr = "error output"
        mock_run.return_value = mock_result

        input_data = {"project_root": "/project"}
        result = self.agent.run(input_data)
        
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failures"], ["pytest_failed"])
        self.assertIn("1 failed", result["logs"])

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    def test_run_timeout(self, mock_sanitize, mock_run):
        """Test timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["pytest"], timeout=120)

        input_data = {"project_root": "/project"}
        result = self.agent.run(input_data)
        
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failures"], ["pytest_timeout"])
        self.assertIn("timeout", result["logs"])

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    def test_run_with_explicit_tests(self, mock_sanitize, mock_run):
        """Test running with explicit test command."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        input_data = {
            "project_root": "/project",
            "tests": "pytest tests/test_specific.py"
        }
        result = self.agent.run(input_data)
        
        self.assertEqual(result["status"], "pass")
        # Verify the command was used
        call_args = mock_run.call_args
        self.assertIn("tests/test_specific.py", call_args[0][0])

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    @patch.object(VerifierAgent, "_changed_test_files")
    def test_run_incremental_tests(self, mock_changed, mock_sanitize, mock_run):
        """Test running incremental tests based on changes."""
        mock_changed.return_value = ["tests/test_changed.py"]
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        input_data = {
            "project_root": "/project",
            "change_set": {"changes": [{"file_path": "module.py"}]}
        }
        result = self.agent.run(input_data)
        
        self.assertEqual(result["status"], "pass")
        call_args = mock_run.call_args
        self.assertIn("tests/test_changed.py", call_args[0][0])

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    @patch.object(VerifierAgent, "_changed_test_files")
    def test_run_repo_wide_fallback(self, mock_changed, mock_sanitize, mock_run):
        """Test fallback to repo-wide tests when no incremental tests found."""
        mock_changed.return_value = []
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        input_data = {
            "project_root": "/project",
            "tests": "pytest"  # This is a repo-wide command
        }
        result = self.agent.run(input_data)
        
        self.assertEqual(result["status"], "pass")
        call_args = mock_run.call_args
        # Should use explicit pytest command
        self.assertEqual(call_args[0][0], ["pytest"])

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    def test_run_env_var_set(self, mock_sanitize, mock_run):
        """Test that AURA_SKIP_CHDIR is set in environment."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        input_data = {"project_root": "/project"}
        self.agent.run(input_data)
        
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["env"]["AURA_SKIP_CHDIR"], "1")

    @patch("subprocess.run")
    @patch("agents.verifier.sanitize_command")
    def test_run_cwd_set(self, mock_sanitize, mock_run):
        """Test that working directory is set correctly."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        input_data = {"project_root": "/project"}
        self.agent.run(input_data)
        
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["cwd"], "/project")


class TestVerifierAgentIntegration(unittest.TestCase):
    """Integration-style tests for VerifierAgent."""

    @patch("subprocess.run")
    @patch.object(VerifierAgent, "_changed_files_from_git")
    @patch.object(VerifierAgent, "_available_test_files")
    def test_full_verification_workflow(self, mock_available, mock_git, mock_run):
        """Test complete verification workflow."""
        # Setup mocks
        mock_git.return_value = ["module.py"]
        mock_available.return_value = ["tests/test_module.py"]
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "============================= test session starts ==============================\nplatform linux -- Python 3.11.0\ncollected 5 items\n\ntests/test_module.py .....                                               [100%]\n\n============================== 5 passed in 0.12s ==============================="
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        agent = VerifierAgent(timeout=60)
        input_data = {
            "project_root": "/project",
            "change_set": {"changes": [{"file_path": "module.py"}]}
        }
        
        result = agent.run(input_data)
        
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["failures"], [])
        self.assertIn("5 passed", result["logs"])

    def test_tokenize_and_score_integration(self):
        """Test tokenization and scoring integration."""
        agent = VerifierAgent()
        
        # Test path tokenization
        tokens = agent._tokenize_path("src/user_authentication/login.py")
        self.assertIn("user", tokens)
        self.assertIn("authentication", tokens)
        self.assertIn("login", tokens)
        
        # Test related file finding with token overlap
        with patch("pathlib.Path.is_file") as mock_is_file:
            mock_is_file.return_value = True
            result = agent._related_test_files(
                Path("/project"),
                "src/user_authentication/login.py",
                [
                    "tests/test_user_auth.py",
                    "tests/test_database.py",
                    "tests/test_login.py"
                ]
            )
        
        # Should find related tests based on token matching
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()
