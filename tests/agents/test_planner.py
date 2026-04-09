"""Unit tests for PlannerAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import json
import unittest
from unittest.mock import Mock, patch

from agents.planner import PlannerAgent


class TestPlannerAgent(unittest.TestCase):
    """Test suite for PlannerAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_brain = Mock()
        self.mock_model = Mock()
        self.mock_model.respond.return_value = json.dumps([
            "Step 1: Analyze requirements",
            "Step 2: Design solution",
            "Step 3: Implement code"
        ])
        
        self.agent = PlannerAgent(self.mock_brain, self.mock_model)

    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.brain, self.mock_brain)
        self.assertEqual(self.agent.model, self.mock_model)

    def test_capabilities(self):
        """Test agent capabilities."""
        self.assertIn("planning", PlannerAgent.capabilities)
        self.assertIn("decomposition", PlannerAgent.capabilities)
        self.assertIn("design", PlannerAgent.capabilities)
        self.assertIn("tree_of_thought", PlannerAgent.capabilities)
        self.assertIn("strategy", PlannerAgent.capabilities)

    def test_run_with_steps(self):
        """Test run method returning steps."""
        input_data = {
            "goal": "Create a feature",
            "memory_snapshot": "",
            "similar_past_problems": "",
            "known_weaknesses": ""
        }
        
        result = self.agent.run(input_data)
        
        self.assertIn("steps", result)
        self.assertIsInstance(result["steps"], list)

    def test_run_returns_dict(self):
        """Test run method always returns dict."""
        input_data = {"goal": "Test"}
        
        result = self.agent.run(input_data)
        
        self.assertIsInstance(result, dict)

    def test_plan_success_legacy(self):
        """Test successful planning with legacy format."""
        result = self.agent.plan(
            "Create feature",
            "memory",
            "similar",
            "weaknesses"
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn("Step 1: Analyze requirements", result)

    def test_plan_with_backfill_context(self):
        """Test planning with backfill context."""
        backfill_context = [
            {"file": "test.py", "coverage_pct": 0.0},
            {"file": "utils.py", "coverage": 25.5}
        ]
        
        result = self.agent.plan(
            "Add tests",
            "memory",
            "similar",
            "weaknesses",
            backfill_context=backfill_context
        )
        
        # Should have called model
        self.mock_model.respond.assert_called()

    def test_respond_with_respond_for_role(self):
        """Test _respond uses respond_for_role when available."""
        mock_model = Mock()
        mock_model.respond_for_role = Mock(return_value="role response")
        
        agent = PlannerAgent(self.mock_brain, mock_model)
        result = agent._respond("test prompt")
        
        self.assertEqual(result, "role response")
        mock_model.respond_for_role.assert_called_once_with("planning", "test prompt")

    def test_respond_fallback(self):
        """Test _respond falls back to respond."""
        result = self.agent._respond("test prompt")
        
        self.mock_model.respond.assert_called_with("test prompt")

    def test_parse_legacy_response_success(self):
        """Test parsing valid legacy response."""
        response = json.dumps(["Step 1: Do this", "Step 2: Do that"])
        
        result = self.agent._parse_legacy_response(response, "goal")
        
        self.assertEqual(result, ["Step 1: Do this", "Step 2: Do that"])
        self.mock_brain.remember.assert_called()

    def test_parse_legacy_response_invalid(self):
        """Test parsing invalid legacy response."""
        response = json.dumps({"not": "a list"})
        
        result = self.agent._parse_legacy_response(response, "goal")
        
        self.assertIn("ERROR", result[0])

    def test_parse_legacy_response_parse_error(self):
        """Test handling parse error in legacy response."""
        response = "invalid json"
        
        result = self.agent._parse_legacy_response(response, "goal")
        
        self.assertIn("ERROR", result[0])

    def test_update_plan_success(self):
        """Test successful plan update."""
        original = ["Step 1: Old", "Step 2: Old"]
        feedback = "Add more detail to step 2"
        
        self.mock_model.respond.return_value = json.dumps([
            "Step 1: Old",
            "Step 2: Old with more detail"
        ])
        
        result = self.agent._update_plan(original, feedback)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_update_plan_with_dict_input(self):
        """Test plan update with dict input containing steps."""
        original = {"steps": ["Step 1: Old", "Step 2: Old"]}
        feedback = "Update"
        
        self.mock_model.respond.return_value = json.dumps(["Step 1: New"])
        
        result = self.agent._update_plan(original, feedback)
        
        self.assertEqual(result, ["Step 1: New"])

    def test_update_plan_invalid_response(self):
        """Test plan update with invalid response."""
        original = ["Step 1: Keep", "Step 2: Keep"]
        feedback = "Update"
        
        self.mock_model.respond.return_value = json.dumps({"invalid": "format"})
        
        result = self.agent._update_plan(original, feedback)
        
        # Should return original plan on invalid response
        self.assertEqual(result, original)

    def test_update_plan_parse_error(self):
        """Test plan update with parse error."""
        original = ["Step 1: Keep"]
        feedback = "Update"
        
        self.mock_model.respond.return_value = "invalid json"
        
        result = self.agent._update_plan(original, feedback)
        
        # Should return original plan on error
        self.assertEqual(result, original)

    @patch("agents.planner.SCHEMAS_AVAILABLE", True)
    @patch("agents.planner.render_prompt")
    def test_plan_structured_success(self, mock_render_prompt):
        """Test structured planning path."""
        mock_render_prompt.return_value = "rendered prompt"
        
        # Mock the model to return valid JSON for structured parsing
        self.mock_model.respond.return_value = json.dumps({
            "plan": [
                {"step_number": 1, "description": "Do something", "target_file": "file.py"}
            ],
            "confidence": 0.9,
            "estimated_complexity": "medium",
            "analysis": "Analysis",
            "gap_assessment": "Gap",
            "approach": "Approach",
            "risk_assessment": "Risk"
        })
        
        # Mock PlannerOutput class
        mock_step = Mock()
        mock_step.step_number = 1
        mock_step.description = "Do something"
        mock_step.target_file = "file.py"
        
        mock_output = Mock()
        mock_output.dict.return_value = {"key": "value"}
        mock_output.plan = [mock_step]
        mock_output.confidence = 0.9
        mock_output.estimated_complexity = "medium"
        mock_output.analysis = "Analysis"
        mock_output.gap_assessment = "Gap"
        mock_output.approach = "Approach"
        mock_output.risk_assessment = "Risk"
        
        with patch("agents.planner.PlannerOutput", return_value=mock_output):
            result = self.agent._plan_structured(
                "goal", "memory", "similar", "weakness", ""
            )
        
        self.assertIn("steps", result)
        self.assertIn("structured_output", result)
        self.assertIn("Step 1: Do something [file.py]", result["steps"])

    @patch("agents.planner.SCHEMAS_AVAILABLE", True)
    @patch("agents.planner.render_prompt")
    def test_plan_structured_fallback(self, mock_render_prompt):
        """Test structured planning fallback to legacy."""
        mock_render_prompt.return_value = "rendered prompt"
        
        self.mock_model.respond.return_value = "invalid json"
        
        result = self.agent._plan_structured(
            "goal", "memory", "similar", "weakness", ""
        )
        
        # Should fall back to legacy format (list of strings)
        self.assertIsInstance(result, list)


class TestPlannerAgentIntegration(unittest.TestCase):
    """Integration-style tests for PlannerAgent."""

    def test_full_planning_workflow(self):
        """Test complete planning workflow."""
        mock_brain = Mock()
        mock_model = Mock()
        mock_model.respond.return_value = json.dumps([
            "Step 1: Understand requirements",
            "Step 2: Design solution",
            "Step 3: Implement",
            "Step 4: Test"
        ])
        
        agent = PlannerAgent(mock_brain, mock_model)
        
        result = agent.plan(
            "Build a REST API",
            "Previous API experience",
            "Similar Flask project",
            "Database connection issues"
        )
        
        # Verify complete workflow
        self.assertEqual(len(result), 4)
        self.assertIn("Step 1: Understand requirements", result)
        
        # Verify brain was consulted
        mock_brain.remember.assert_called()


if __name__ == "__main__":
    unittest.main()
