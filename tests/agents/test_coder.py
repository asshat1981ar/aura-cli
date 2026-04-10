"""Unit tests for CoderAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import json
import unittest
from unittest.mock import Mock, MagicMock, patch

from agents.coder import CoderAgent


class TestCoderAgent(unittest.TestCase):
    """Test suite for CoderAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_brain = Mock()
        self.mock_brain.recall_with_budget.return_value = []

        self.mock_model = Mock()
        self.mock_model.respond.return_value = json.dumps({"aura_target": "test.py", "code": "def hello(): return 'world'"})

        self.agent = CoderAgent(self.mock_brain, self.mock_model)

    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.brain, self.mock_brain)
        self.assertEqual(self.agent.model, self.mock_model)
        self.assertEqual(self.agent.MAX_ITERATIONS, 3)
        self.assertIsNone(self.agent.tester)

    def test_capabilities(self):
        """Test agent capabilities."""
        self.assertIn("code_generation", CoderAgent.capabilities)
        self.assertIn("coding", CoderAgent.capabilities)
        self.assertIn("implement", CoderAgent.capabilities)
        self.assertIn("refactor", CoderAgent.capabilities)

    def test_implement_success_no_tester(self):
        """Test successful code generation without tester."""
        result = self.agent.implement("Create a hello function")

        # Should return formatted code
        self.assertIn("def hello()", result)
        self.assertIn("AURA_TARGET", result)

        # Should have called model.respond
        self.mock_model.respond.assert_called()

    def test_implement_with_tester_success(self):
        """Test code generation with tester that approves."""
        mock_tester = Mock()
        mock_tester.generate_tests.return_value = "def test_hello(): pass"
        mock_tester.evaluate_code.return_value = {"summary": "likely pass"}

        agent = CoderAgent(self.mock_brain, self.mock_model, tester=mock_tester)
        result = agent.implement("Create a hello function")

        # Should return formatted code
        self.assertIn("def hello()", result)

        # Should have called tester
        mock_tester.generate_tests.assert_called()
        mock_tester.evaluate_code.assert_called()

    def test_implement_with_tester_retry(self):
        """Test code generation with tester feedback loop."""
        mock_tester = Mock()
        mock_tester.generate_tests.return_value = "def test_hello(): pass"
        # First iteration fails, second passes
        mock_tester.evaluate_code.side_effect = [{"summary": "needs work"}, {"summary": "likely pass"}]

        agent = CoderAgent(self.mock_brain, self.mock_model, tester=mock_tester)
        result = agent.implement("Create a hello function")

        # Should eventually return code
        self.assertIn("def hello()", result)

        # Should have called tester twice
        self.assertEqual(mock_tester.evaluate_code.call_count, 2)

    def test_implement_model_error(self):
        """Test handling of model error on first iteration."""
        self.mock_model.respond.return_value = "invalid json"

        result = self.agent.implement("Create a hello function")

        # Should return result (may contain error or fallback content)
        self.assertIsInstance(result, str)

    def test_respond_with_respond_for_role(self):
        """Test _respond uses respond_for_role when available."""
        mock_model = Mock()
        mock_model.respond_for_role = Mock(return_value="role response")

        agent = CoderAgent(self.mock_brain, mock_model)
        result = agent._respond("test prompt")

        self.assertEqual(result, "role response")
        mock_model.respond_for_role.assert_called_once_with("code_generation", "test prompt")

    def test_respond_fallback_to_respond(self):
        """Test _respond falls back to respond when respond_for_role unavailable."""
        result = self.agent._respond("test prompt")

        self.mock_model.respond.assert_called_with("test prompt")

    def test_format_final_code(self):
        """Test code formatting."""
        output = {"aura_target": "path/to/file.py", "code": "def test(): pass", "explanation": "Test function"}

        result = self.agent._format_final_code(output)

        self.assertIn("# AURA_TARGET: path/to/file.py", result)
        self.assertIn("def test(): pass", result)

    def test_remember_output(self):
        """Test memory storage of output."""
        result = {"aura_target": "test.py", "code": "def test(): pass", "explanation": "Test"}
        tests = "def test_test(): pass"

        self.agent._remember_output("Create test", result, tests)

        # Should have attempted to store in brain
        # (brain.remember or brain.store may be called)
        self.assertTrue(self.mock_brain.remember.called or self.mock_brain.store.called or self.mock_brain.add.called, "Brain memory method should have been called")

    def test_code_block_regex(self):
        """Test code block extraction regex."""
        # Test matching code blocks
        text = "```python\ndef hello():\n    pass\n```"
        match = self.agent.CODE_BLOCK_RE.search(text)
        self.assertIsNotNone(match)
        self.assertIn("def hello()", match.group(1))

    def test_aura_target_directive(self):
        """Test AURA_TARGET directive extraction."""
        text = "# AURA_TARGET: path/to/file.py\ncode here"
        lines = text.splitlines()

        target = None
        for line in lines:
            if line.startswith(self.agent.AURA_TARGET_DIRECTIVE):
                target = line[len(self.agent.AURA_TARGET_DIRECTIVE) :].strip()
                break

        self.assertEqual(target, "path/to/file.py")

    @patch("agents.coder.SCHEMAS_AVAILABLE", True)
    @patch("agents.coder.render_prompt")
    @patch("agents.coder.CoderOutput")
    def test_implement_structured_success(self, mock_coder_output, mock_render_prompt):
        """Test structured output path when schemas available."""
        mock_render_prompt.return_value = "rendered prompt"

        mock_output = Mock()
        mock_output.dict.return_value = {"key": "value"}
        mock_output.aura_target = "test.py"
        mock_output.code = "def test(): pass"
        mock_output.explanation = "Test"
        mock_output.dependencies = []
        mock_output.edge_cases_handled = []
        mock_output.confidence = 0.9
        mock_output.problem_analysis = "Analysis"
        mock_output.approach_selection = "Approach"
        mock_output.design_considerations = "Design"
        mock_output.testing_strategy = "Tests"

        mock_coder_output.return_value = mock_output

        result = self.agent._implement_structured("task", "memory", "", "", "")

        self.assertEqual(result["aura_target"], "test.py")
        self.assertEqual(result["code"], "def test(): pass")
        self.assertEqual(result["confidence"], 0.9)

    def test_implement_legacy_json_parsing(self):
        """Test legacy JSON parsing."""
        self.mock_model.respond.return_value = '```json\n{"aura_target": "test.py", "code": "def test(): pass"}\n```'

        result = self.agent._implement_legacy("task", "memory", "", "", "")

        self.assertEqual(result["aura_target"], "test.py")
        self.assertEqual(result["code"], "def test(): pass")

    def test_implement_legacy_code_block_extraction(self):
        """Test legacy code block extraction fallback."""
        self.mock_model.respond.return_value = """# AURA_TARGET: test.py
```python
def test():
    pass
```
"""

        result = self.agent._implement_legacy("task", "memory", "", "", "")

        self.assertEqual(result["aura_target"], "test.py")
        self.assertIn("def test():", result["code"])


class TestCoderAgentIntegration(unittest.TestCase):
    """Integration-style tests for CoderAgent."""

    def test_full_workflow_mocked(self):
        """Test full workflow with mocked dependencies."""
        mock_brain = Mock()
        mock_brain.recall_with_budget.return_value = ["previous context"]

        mock_model = Mock()
        mock_model.respond.return_value = json.dumps({"aura_target": "src/hello.py", "code": "def hello():\n    return 'Hello, World!'", "explanation": "Simple hello function"})

        agent = CoderAgent(mock_brain, mock_model)
        result = agent.implement("Create a hello function")

        # Verify complete workflow
        self.assertIn("AURA_TARGET: src/hello.py", result)
        self.assertIn("def hello():", result)

        # Verify brain was consulted
        mock_brain.recall_with_budget.assert_called()


if __name__ == "__main__":
    unittest.main()
