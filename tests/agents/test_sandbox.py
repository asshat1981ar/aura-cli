"""Unit tests for SandboxAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import subprocess
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from agents.sandbox import SandboxAgent, SandboxResult


class TestSandboxResult(unittest.TestCase):
    """Test suite for SandboxResult dataclass."""

    def test_init(self):
        """Test SandboxResult initialization."""
        result = SandboxResult(success=True, exit_code=0, stdout="output", stderr="error")

        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "output")
        self.assertEqual(result.stderr, "error")
        self.assertFalse(result.timed_out)
        self.assertIsNone(result.execution_path)
        self.assertEqual(result.metadata, {})

    def test_passed_property_success(self):
        """Test passed property with success."""
        result = SandboxResult(success=True, exit_code=0, stdout="", stderr="")
        self.assertTrue(result.passed)

    def test_passed_property_failure(self):
        """Test passed property with failure."""
        result = SandboxResult(success=False, exit_code=1, stdout="", stderr="")
        self.assertFalse(result.passed)

    def test_passed_property_timeout(self):
        """Test passed property with timeout."""
        result = SandboxResult(success=False, exit_code=-1, stdout="", stderr="timeout", timed_out=True)
        self.assertFalse(result.passed)

    def test_summary_pass(self):
        """Test summary for passed result."""
        result = SandboxResult(success=True, exit_code=0, stdout="ok", stderr="")
        summary = result.summary()

        self.assertIn("[PASS]", summary)
        self.assertIn("exit=0", summary)

    def test_summary_fail(self):
        """Test summary for failed result."""
        result = SandboxResult(success=False, exit_code=1, stdout="", stderr="error")
        summary = result.summary()

        self.assertIn("[FAIL]", summary)
        self.assertIn("exit=1", summary)

    def test_summary_timeout(self):
        """Test summary for timeout result."""
        result = SandboxResult(success=False, exit_code=-1, stdout="", stderr="timeout", timed_out=True)
        summary = result.summary()

        self.assertIn("[TIMEOUT]", summary)

    def test_str_representation(self):
        """Test string representation."""
        result = SandboxResult(success=True, exit_code=0, stdout="output", stderr="")
        s = str(result)

        self.assertIn("SandboxResult", s)
        self.assertIn("passed=True", s)
        self.assertIn("stdout:", s)


class TestSandboxAgent(unittest.TestCase):
    """Test suite for SandboxAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_brain = Mock()
        self.agent = SandboxAgent(self.mock_brain, timeout=30)

    def test_init_default(self):
        """Test default initialization."""
        agent = SandboxAgent(self.mock_brain)

        self.assertEqual(agent.brain, self.mock_brain)
        self.assertEqual(agent.timeout, 30)
        self.assertIsNotNone(agent.python_exec)

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        agent = SandboxAgent(self.mock_brain, timeout=60)

        self.assertEqual(agent.timeout, 60)

    def test_init_custom_python_exec(self):
        """Test initialization with custom python executable."""
        agent = SandboxAgent(self.mock_brain, python_exec="/usr/bin/python3")

        self.assertEqual(agent.python_exec, "/usr/bin/python3")

    def test_find_python(self):
        """Test finding Python executable."""
        import sys

        result = SandboxAgent._find_python()

        self.assertEqual(result, sys.executable)

    @patch("agents.sandbox.SandboxAgent._record")
    def test_run_code_success(self, mock_record):
        """Test running code successfully."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("output", "")

        with patch("subprocess.Popen", return_value=mock_process):
            with patch("tempfile.TemporaryDirectory") as mock_tempdir:
                mock_tempdir.return_value.__enter__ = Mock(return_value="/tmp/test")
                mock_tempdir.return_value.__exit__ = Mock(return_value=False)

                # Mock Path operations
                with patch("pathlib.Path.write_text"):
                    result = self.agent.run_code("print('hello')")

        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "output")

    @patch("agents.sandbox.SandboxAgent._record")
    def test_run_code_with_extra_files(self, mock_record):
        """Test running code with extra files."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("output", "")

        with patch("subprocess.Popen", return_value=mock_process):
            with patch("tempfile.TemporaryDirectory") as mock_tempdir:
                mock_tempdir.return_value.__enter__ = Mock(return_value="/tmp/test")
                mock_tempdir.return_value.__exit__ = Mock(return_value=False)

                with patch("pathlib.Path.write_text"):
                    extra_files = {"helper.py": "def helper(): return 42"}
                    result = self.agent.run_code("import helper", extra_files=extra_files)

        self.assertTrue(result.success)

    @patch("subprocess.Popen")
    def test_run_file(self, mock_popen):
        """Test running existing file."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("output", "")
        mock_popen.return_value = mock_process

        result = self.agent.run_file("/path/to/script.py")

        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)

    @patch("agents.sandbox.SandboxAgent._record")
    def test_run_tests_success(self, mock_record):
        """Test running tests successfully."""
        mock_run_pytest = Mock(return_value=SandboxResult(success=True, exit_code=0, stdout="1 passed", stderr="", execution_path="/tmp/test"))
        self.agent._run_pytest = mock_run_pytest

        with patch("tempfile.TemporaryDirectory") as mock_tempdir:
            mock_tempdir.return_value.__enter__ = Mock(return_value="/tmp/test")
            mock_tempdir.return_value.__exit__ = Mock(return_value=False)

            with patch("pathlib.Path.write_text"):
                code = "def add(a, b): return a + b"
                tests = "def test_add(): assert add(2, 3) == 5"
                result = self.agent.run_tests(code, tests)

        self.assertTrue(result.success)

    @patch("agents.sandbox.SandboxAgent._record")
    def test_run_tests_injects_import(self, mock_record):
        """Test that run_tests injects source import when missing."""
        mock_run_pytest = Mock(return_value=SandboxResult(success=True, exit_code=0, stdout="", stderr=""))
        self.agent._run_pytest = mock_run_pytest

        with patch("tempfile.TemporaryDirectory") as mock_tempdir:
            mock_tempdir.return_value.__enter__ = Mock(return_value="/tmp/test")
            mock_tempdir.return_value.__exit__ = Mock(return_value=False)

            with patch("pathlib.Path.write_text") as mock_write:
                code = "def func(): pass"
                tests = "def test_func(): pass"  # No import statement
                self.agent.run_tests(code, tests)

                # Check that import was injected in the test file content
                written_content = mock_write.call_args_list[-1][0][0]
                self.assertIn("sys.path.insert", written_content)

    @patch("agents.sandbox.SandboxAgent._record")
    def test_run_tests_preserves_existing_import(self, mock_record):
        """Test that run_tests preserves existing import."""
        mock_run_pytest = Mock(return_value=SandboxResult(success=True, exit_code=0, stdout="", stderr=""))
        self.agent._run_pytest = mock_run_pytest

        with patch("tempfile.TemporaryDirectory") as mock_tempdir:
            mock_tempdir.return_value.__enter__ = Mock(return_value="/tmp/test")
            mock_tempdir.return_value.__exit__ = Mock(return_value=False)

            with patch("pathlib.Path.write_text") as mock_write:
                code = "def func(): pass"
                tests = "from source import func\ndef test_func(): pass"
                self.agent.run_tests(code, tests)

                # Check that import was NOT injected (already present)
                written_content = mock_write.call_args_list[-1][0][0]
                self.assertNotIn("sys.path.insert", written_content)

    @patch("subprocess.Popen")
    def test_run_success(self, mock_popen):
        """Test internal _run method success."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("stdout", "stderr")
        mock_popen.return_value = mock_process

        result = self.agent._run("/path/to/script.py", "/cwd")

        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "stdout")
        self.assertEqual(result.stderr, "stderr")

    @patch("subprocess.Popen")
    def test_run_timeout(self, mock_popen):
        """Test internal _run method timeout."""
        from subprocess import TimeoutExpired

        mock_process = Mock()
        # First communicate raises TimeoutExpired, second (after kill) returns normally
        mock_process.communicate.side_effect = [TimeoutExpired(cmd="test", timeout=30), ("", "")]
        mock_process.returncode = -1
        mock_popen.return_value = mock_process

        result = self.agent._run("/path/to/script.py", "/cwd")

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        mock_process.kill.assert_called_once()

    @patch("subprocess.Popen")
    def test_run_exception(self, mock_popen):
        """Test internal _run method exception."""
        mock_popen.side_effect = Exception("Process error")

        result = self.agent._run("/path/to/script.py", "/cwd")

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("internal error", result.stderr)

    @patch("subprocess.Popen")
    def test_run_pytest_success(self, mock_popen):
        """Test _run_pytest success."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("1 passed", "")
        mock_popen.return_value = mock_process

        result = self.agent._run_pytest("/tmp/test")

        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "1 passed")

    @patch("subprocess.Popen")
    def test_run_pytest_not_found(self, mock_popen):
        """Test _run_pytest falls back to unittest."""
        mock_popen.side_effect = [FileNotFoundError(), Mock()]

        # Second mock is for unittest fallback
        mock_unittest = Mock()
        mock_unittest.returncode = 0
        mock_unittest.communicate.return_value = ("unittest output", "")

        def side_effect(*args, **kwargs):
            if "pytest" in args[0]:
                raise FileNotFoundError()
            return mock_unittest

        mock_popen.side_effect = side_effect

        result = self.agent._run_pytest("/tmp/test")

        # Should have fallen back to unittest
        self.assertTrue(result.success)

    @patch("subprocess.Popen")
    def test_run_pytest_timeout(self, mock_popen):
        """Test _run_pytest timeout."""
        from subprocess import TimeoutExpired

        mock_process = Mock()
        # First communicate raises TimeoutExpired, second (after kill) returns normally
        mock_process.communicate.side_effect = [TimeoutExpired(cmd="test", timeout=30), ("", "")]
        mock_process.returncode = -1
        mock_popen.return_value = mock_process

        result = self.agent._run_pytest("/tmp/test")

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)

    @patch("subprocess.Popen")
    def test_run_unittest_success(self, mock_popen):
        """Test _run_unittest success."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("unittest success", "")
        mock_popen.return_value = mock_process

        result = self.agent._run_unittest("/tmp/test")

        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "unittest success")

    @patch("subprocess.Popen")
    def test_run_unittest_exception(self, mock_popen):
        """Test _run_unittest exception."""
        mock_popen.side_effect = Exception("Unittest error")

        result = self.agent._run_unittest("/tmp/test")

        self.assertFalse(result.success)
        self.assertIn("unittest runner error", result.stderr)

    def test_parse_pytest_summary_passed(self):
        """Test parsing pytest summary with passed count."""
        output = "test_file.py::test_func PASSED [ 50%]\n1 passed in 0.01s"

        result = self.agent._parse_pytest_summary(output)

        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["errors"], 0)

    def test_parse_pytest_summary_failed(self):
        """Test parsing pytest summary with failed count."""
        output = "1 failed, 2 passed in 0.01s"

        result = self.agent._parse_pytest_summary(output)

        # Regex finds both passed and failed counts
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["failed"], 1)

    def test_parse_pytest_summary_errors(self):
        """Test parsing pytest summary with errors count."""
        output = "1 error in 0.01s"

        result = self.agent._parse_pytest_summary(output)

        self.assertEqual(result["errors"], 1)

    def test_parse_pytest_summary_multiple(self):
        """Test parsing pytest summary with multiple counts."""
        output = "5 passed, 2 failed, 1 error in 0.01s"

        result = self.agent._parse_pytest_summary(output)

        self.assertEqual(result["passed"], 5)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["errors"], 1)

    def test_record(self):
        """Test recording to brain."""
        result = SandboxResult(success=True, exit_code=0, stdout="output", stderr="")

        self.agent._record(result, "test_label", "code snippet")

        self.mock_brain.remember.assert_called_once()
        call_args = self.mock_brain.remember.call_args[0][0]
        self.assertIn("SandboxAgent", call_args)
        self.assertIn("test_label", call_args)


class TestSandboxAgentIntegration(unittest.TestCase):
    """Integration-style tests for SandboxAgent."""

    def test_full_sandbox_workflow(self):
        """Test complete sandbox workflow with mocked internals."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("Hello, World!", "")

        mock_brain = Mock()
        agent = SandboxAgent(mock_brain, timeout=30)

        with patch("subprocess.Popen", return_value=mock_process):
            with patch("tempfile.TemporaryDirectory") as mock_tempdir:
                mock_tempdir.return_value.__enter__ = Mock(return_value="/tmp/aura_test")
                mock_tempdir.return_value.__exit__ = Mock(return_value=False)

                with patch("pathlib.Path.write_text"):
                    code = "print('Hello, World!')"
                    result = agent.run_code(code)

        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "Hello, World!")
        mock_brain.remember.assert_called()


if __name__ == "__main__":
    unittest.main()
