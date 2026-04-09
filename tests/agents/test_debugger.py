"""Unit tests for DebuggerAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import json
import unittest
from unittest.mock import Mock, patch

from agents.debugger import DebuggerAgent


class TestDebuggerAgent(unittest.TestCase):
    """Test suite for DebuggerAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_brain = Mock()
        self.mock_model = Mock()
        self.mock_model.respond.return_value = json.dumps({
            "summary": "Module not found",
            "diagnosis": "The module 'requests' is not installed",
            "fix_strategy": "Run pip install requests",
            "severity": "HIGH"
        })
        
        self.agent = DebuggerAgent(self.mock_brain, self.mock_model)

    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.brain, self.mock_brain)
        self.assertEqual(self.agent.model, self.mock_model)

    def test_respond_with_respond_for_role(self):
        """Test _respond uses respond_for_role when available."""
        mock_model = Mock()
        mock_model.respond_for_role = Mock(return_value="role response")
        
        agent = DebuggerAgent(self.mock_brain, mock_model)
        result = agent._respond("test prompt")
        
        self.assertEqual(result, "role response")
        mock_model.respond_for_role.assert_called_once_with("analysis", "test prompt")

    def test_respond_fallback_to_respond(self):
        """Test _respond falls back to respond when respond_for_role unavailable."""
        result = self.agent._respond("test prompt")
        
        self.mock_model.respond.assert_called_with("test prompt")

    def test_diagnose_success(self):
        """Test successful diagnosis."""
        result = self.agent.diagnose(
            error_message="ModuleNotFoundError: No module named 'requests'",
            current_goal="Install dependencies"
        )
        
        self.assertEqual(result["summary"], "Module not found")
        self.assertEqual(result["diagnosis"], "The module 'requests' is not installed")
        self.assertEqual(result["fix_strategy"], "Run pip install requests")
        self.assertEqual(result["severity"], "HIGH")

    def test_diagnose_with_context(self):
        """Test diagnosis with context."""
        result = self.agent.diagnose(
            error_message="Error message",
            current_goal="Goal",
            context="Code context here",
            improve_plan="Previous plan",
            implement_details={"key": "value"}
        )
        
        # Should have called model.respond
        self.mock_model.respond.assert_called()

    def test_diagnose_with_empty_implement_details(self):
        """Test diagnosis with empty implement_details defaults."""
        result = self.agent.diagnose(
            error_message="Error",
            current_goal="Goal"
        )
        
        # Should succeed even without implement_details
        self.assertIn("summary", result)

    def test_diagnose_model_returns_invalid_json(self):
        """Test diagnosis when model returns invalid JSON."""
        self.mock_model.respond.return_value = "invalid json"
        
        result = self.agent.diagnose(
            error_message="Test error",
            current_goal="Test goal"
        )
        
        # Should return fallback error response
        self.assertEqual(result["summary"], "LLM diagnosis failed.")
        self.assertEqual(result["severity"], "CRITICAL")
        self.assertIn("Failed to get LLM diagnosis", result["diagnosis"])

    def test_diagnose_model_raises_exception(self):
        """Test diagnosis when model raises an exception."""
        self.mock_model.respond.side_effect = Exception("Model error")
        
        result = self.agent.diagnose(
            error_message="Test error",
            current_goal="Test goal"
        )
        
        # Should return fallback error response
        self.assertEqual(result["summary"], "LLM diagnosis failed.")
        self.assertEqual(result["severity"], "CRITICAL")

    def test_diagnose_all_severities(self):
        """Test diagnosis with different severity levels."""
        severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        
        for severity in severities:
            self.mock_model.respond.return_value = json.dumps({
                "summary": f"Test {severity}",
                "diagnosis": "Test diagnosis",
                "fix_strategy": "Fix it",
                "severity": severity
            })
            
            result = self.agent.diagnose("error", "goal")
            self.assertEqual(result["severity"], severity)

    def test_diagnose_with_complex_error(self):
        """Test diagnosis with complex multi-line error."""
        complex_error = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    result = function(arg)
  File "utils.py", line 20, in function
    return process(data)
AttributeError: 'NoneType' object has no attribute 'method'"""
        
        self.mock_model.respond.return_value = json.dumps({
            "summary": "Attribute error on NoneType",
            "diagnosis": "Variable is None when accessed",
            "fix_strategy": "Add null check before accessing",
            "severity": "MEDIUM"
        })
        
        result = self.agent.diagnose(
            error_message=complex_error,
            current_goal="Fix attribute error"
        )
        
        self.assertEqual(result["summary"], "Attribute error on NoneType")

    @patch("agents.debugger.parse_llm_to_model")
    def test_diagnose_with_parsing_exception(self, mock_parse):
        """Test diagnosis when parsing raises exception."""
        mock_parse.side_effect = ValueError("Parse error")
        
        result = self.agent.diagnose("error", "goal")
        
        self.assertEqual(result["summary"], "LLM diagnosis failed.")
        self.assertEqual(result["severity"], "CRITICAL")

    def test_diagnose_prompt_contains_all_info(self):
        """Test that diagnosis prompt contains all provided information."""
        self.agent.diagnose(
            error_message="Test error message",
            current_goal="Test current goal",
            context="Test context",
            improve_plan="Test improve plan",
            implement_details={"key": "value"}
        )
        
        # Get the prompt passed to respond
        call_args = self.mock_model.respond.call_args
        prompt = call_args[0][0]
        
        # Verify all information is in the prompt
        self.assertIn("Test error message", prompt)
        self.assertIn("Test current goal", prompt)
        self.assertIn("Test context", prompt)
        self.assertIn("Test improve plan", prompt)
        self.assertIn("key", prompt)

    def test_diagnose_implement_details_serialization(self):
        """Test that implement_details are properly serialized."""
        details = {
            "file_path": "/path/to/file.py",
            "line_number": 42,
            "function_name": "test_func"
        }
        
        self.agent.diagnose(
            error_message="Error",
            current_goal="Goal",
            implement_details=details
        )
        
        call_args = self.mock_model.respond.call_args
        prompt = call_args[0][0]
        
        self.assertIn("file_path", prompt)
        self.assertIn("line_number", prompt)
        self.assertIn("test_func", prompt)

    def test_respond_handles_missing_method_gracefully(self):
        """Test _respond handles model without respond_for_role."""
        class SimpleModel:
            def respond(self, prompt):
                return "simple response"
        
        agent = DebuggerAgent(self.mock_brain, SimpleModel())
        result = agent._respond("test")
        
        self.assertEqual(result, "simple response")

    def test_respond_handles_none_respond_for_role(self):
        """Test _respond when respond_for_role exists but is None."""
        mock_model = Mock()
        mock_model.respond_for_role = None
        mock_model.respond.return_value = "fallback"
        
        agent = DebuggerAgent(self.mock_brain, mock_model)
        result = agent._respond("test")
        
        self.assertEqual(result, "fallback")


class TestDebuggerAgentIntegration(unittest.TestCase):
    """Integration-style tests for DebuggerAgent."""

    def test_full_debugging_workflow(self):
        """Test complete debugging workflow."""
        mock_brain = Mock()
        mock_model = Mock()
        mock_model.respond.return_value = json.dumps({
            "summary": "Index out of range",
            "diagnosis": "List index exceeds bounds",
            "fix_strategy": "Check list length before indexing",
            "severity": "HIGH"
        })
        
        agent = DebuggerAgent(mock_brain, mock_model)
        
        result = agent.diagnose(
            error_message="IndexError: list index out of range",
            current_goal="Fix indexing bug",
            context="data = [1, 2, 3]; print(data[5])",
            improve_plan="Add bounds checking",
            implement_details={"file": "main.py", "line": 25}
        )
        
        # Verify complete response
        self.assertEqual(result["summary"], "Index out of range")
        self.assertEqual(result["severity"], "HIGH")
        self.assertIn("bounds", result["diagnosis"].lower())


if __name__ == "__main__":
    unittest.main()
