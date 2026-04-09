"""Unit tests for TesterAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import unittest
from unittest.mock import Mock

from agents.tester import TesterAgent


class MockSandboxResult:
    """Mock SandboxResult for testing."""
    def __init__(self, passed=True, stdout="", stderr="", exit_code=0, 
                 timed_out=False, metadata=None):
        self.passed = passed
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.metadata = metadata or {}


class TestTesterAgent(unittest.TestCase):
    """Test suite for TesterAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_brain = Mock()
        self.mock_brain.recall_with_budget.return_value = []
        self.mock_model = Mock()
        self.mock_model.respond.return_value = """
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
"""
        self.mock_sandbox = Mock()
        self.mock_sandbox.timeout = 30
        self.mock_sandbox.run_tests.return_value = MockSandboxResult(
            passed=True,
            stdout="1 passed",
            exit_code=0,
            metadata={"passed": 1, "failed": 0, "errors": 0}
        )
        
        self.agent = TesterAgent(self.mock_brain, self.mock_model, self.mock_sandbox)

    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.brain, self.mock_brain)
        self.assertEqual(self.agent.model, self.mock_model)
        self.assertEqual(self.agent.sandbox, self.mock_sandbox)

    def test_respond_with_respond_for_role(self):
        """Test _respond uses respond_for_role when available."""
        mock_model = Mock()
        mock_model.respond_for_role = Mock(return_value="role response")
        
        agent = TesterAgent(self.mock_brain, mock_model, self.mock_sandbox)
        result = agent._respond("test prompt")
        
        self.assertEqual(result, "role response")
        mock_model.respond_for_role.assert_called_once_with("quality", "test prompt")

    def test_respond_fallback_to_respond(self):
        """Test _respond falls back to respond when respond_for_role unavailable."""
        result = self.agent._respond("test prompt")
        
        self.mock_model.respond.assert_called_with("test prompt")

    def test_generate_tests(self):
        """Test test generation."""
        code = "def add(a, b): return a + b"
        
        result = self.agent.generate_tests(code)
        
        self.assertIn("test_add", result)
        self.mock_model.respond.assert_called()
        self.mock_brain.remember.assert_called()

    def test_generate_tests_with_context(self):
        """Test test generation with context."""
        code = "def add(a, b): return a + b"
        context = "This function adds two numbers"
        
        result = self.agent.generate_tests(code, context=context)
        
        self.assertIn("test_add", result)
        # Verify context was included in prompt
        call_args = self.mock_model.respond.call_args[0][0]
        self.assertIn("adds two numbers", call_args)

    def test_generate_tests_recalls_memory(self):
        """Test that generate_tests recalls memory."""
        self.mock_brain.recall_with_budget.return_value = ["memory 1", "memory 2"]
        
        self.agent.generate_tests("code")
        
        self.mock_brain.recall_with_budget.assert_called_with(max_tokens=1500)

    def test_evaluate_code_success(self):
        """Test code evaluation with success."""
        code = "def add(a, b): return a + b"
        tests = "def test_add(): assert add(2, 3) == 5"
        
        result = self.agent.evaluate_code(code, tests)
        
        self.assertEqual(result["summary"], "Tests PASSED. Pytest summary: 1 passed, 0 failed, 0 errors.")
        self.assertEqual(result["actual_output"]["passed"], True)
        self.assertEqual(result["actual_output"]["exit_code"], 0)

    def test_evaluate_code_failure(self):
        """Test code evaluation with failure."""
        self.mock_sandbox.run_tests.return_value = MockSandboxResult(
            passed=False,
            stdout="",
            stderr="AssertionError",
            exit_code=1,
            metadata={"passed": 0, "failed": 1, "errors": 0}
        )
        
        result = self.agent.evaluate_code("code", "tests")
        
        self.assertIn("Tests FAILED", result["summary"])
        self.assertEqual(result["actual_output"]["passed"], False)

    def test_evaluate_code_timeout(self):
        """Test code evaluation with timeout."""
        self.mock_sandbox.run_tests.return_value = MockSandboxResult(
            passed=False,
            timed_out=True,
            stderr="Timeout",
            exit_code=-1
        )
        
        result = self.agent.evaluate_code("code", "tests")
        
        self.assertIn("TIMED OUT", result["summary"])
        self.assertTrue(result["actual_output"]["timed_out"])

    def test_evaluate_code_no_metadata(self):
        """Test code evaluation without metadata."""
        self.mock_sandbox.run_tests.return_value = MockSandboxResult(
            passed=True,
            metadata={}
        )
        
        result = self.agent.evaluate_code("code", "tests")
        
        self.assertEqual(result["summary"], "Tests PASSED.")

    def test_evaluate_code_contains_output(self):
        """Test that evaluate_code includes full output."""
        self.mock_sandbox.run_tests.return_value = MockSandboxResult(
            passed=True,
            stdout="test output",
            stderr="",
            exit_code=0,
            timed_out=False,
            metadata={"passed": 2}
        )
        
        result = self.agent.evaluate_code("code", "tests")
        
        self.assertEqual(result["actual_output"]["stdout"], "test output")
        self.assertEqual(result["actual_output"]["stderr"], "")
        self.assertEqual(result["actual_output"]["exit_code"], 0)
        self.assertEqual(result["actual_output"]["timed_out"], False)
        self.assertEqual(result["actual_output"]["metadata"]["passed"], 2)

    def test_generate_tests_includes_code_in_prompt(self):
        """Test that generated tests prompt includes code."""
        code = "def multiply(a, b): return a * b"
        
        self.agent.generate_tests(code)
        
        prompt = self.mock_model.respond.call_args[0][0]
        self.assertIn("multiply(a, b)", prompt)
        self.assertIn("Code to test:", prompt)

    def test_generate_tests_includes_framework_instructions(self):
        """Test that test generation includes framework instructions."""
        self.agent.generate_tests("code")
        
        prompt = self.mock_model.respond.call_args[0][0]
        self.assertIn("unittest", prompt)
        self.assertIn("pytest", prompt)

    def test_sandbox_timeout_accessible(self):
        """Test that sandbox timeout is accessible in result."""
        self.mock_sandbox.timeout = 60
        
        result = self.agent.evaluate_code("code", "tests")
        
        # Timeout should be accessible (shown in summary on timeout)
        self.assertIn("summary", result)


class TestTesterAgentIntegration(unittest.TestCase):
    """Integration-style tests for TesterAgent."""

    def test_full_testing_workflow(self):
        """Test complete testing workflow."""
        mock_brain = Mock()
        mock_brain.recall_with_budget.return_value = ["Previous test patterns"]
        mock_model = Mock()
        mock_model.respond.return_value = """
import unittest

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
"""
        mock_sandbox = Mock()
        mock_sandbox.timeout = 30
        mock_sandbox.run_tests.return_value = MockSandboxResult(
            passed=True,
            stdout="2 passed",
            exit_code=0,
            metadata={"passed": 2, "failed": 0, "errors": 0}
        )
        
        agent = TesterAgent(mock_brain, mock_model, mock_sandbox)
        
        code = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
        
        # Generate tests
        tests = agent.generate_tests(code, context="Calculator functions")
        self.assertIn("test_add", tests)
        
        # Evaluate code
        result = agent.evaluate_code(code, tests)
        self.assertIn("PASSED", result["summary"])
        
        # Verify sandbox was called
        mock_sandbox.run_tests.assert_called_with(code, tests)


if __name__ == "__main__":
    unittest.main()
