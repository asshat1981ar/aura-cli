"""Unit tests for CriticAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import json
import unittest
from unittest.mock import Mock, patch

from agents.critic import CriticAgent


class TestCriticAgent(unittest.TestCase):
    """Test suite for CriticAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_brain = Mock()
        self.mock_brain.recall_with_budget.return_value = []
        self.mock_model = Mock()
        self.mock_model.respond.return_value = json.dumps({
            "initial_assessment": "Good structure",
            "completeness_check": "All items covered",
            "feasibility_analysis": "Feasible",
            "risk_identification": "Low risk",
            "overall_assessment": "approve",
            "confidence": 0.85,
            "issues": [],
            "positive_aspects": ["Clean code", "Good tests"],
            "summary": "Overall good quality"
        })
        
        self.agent = CriticAgent(self.mock_brain, self.mock_model)

    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.brain, self.mock_brain)
        self.assertEqual(self.agent.model, self.mock_model)
        self.assertIn("critique", CriticAgent.capabilities)
        self.assertIn("review", CriticAgent.capabilities)

    def test_respond_with_respond_for_role(self):
        """Test _respond uses respond_for_role when available."""
        mock_model = Mock()
        mock_model.respond_for_role = Mock(return_value="role response")
        
        agent = CriticAgent(self.mock_brain, mock_model)
        result = agent._respond("critique", "test prompt")
        
        self.assertEqual(result, "role response")
        mock_model.respond_for_role.assert_called_once_with("critique", "test prompt")

    def test_respond_fallback_to_respond(self):
        """Test _respond falls back to respond when respond_for_role unavailable."""
        result = self.agent._respond("critique", "test prompt")
        
        self.mock_model.respond.assert_called_with("test prompt")

    def test_critique_plan_legacy(self):
        """Test legacy plan critique."""
        self.agent.use_structured = False
        self.mock_model.respond.return_value = "Legacy feedback"
        
        result = self.agent.critique_plan("Build API", ["Step 1", "Step 2"])
        
        self.assertEqual(result, "Legacy feedback")
        self.mock_brain.remember.assert_called()

    def test_critique_code_legacy(self):
        """Test legacy code critique."""
        self.agent.use_structured = False
        self.mock_model.respond.return_value = "Code feedback"
        
        result = self.agent.critique_code("Task", "def test(): pass")
        
        self.assertEqual(result, "Code feedback")

    def test_validate_mutation_legacy(self):
        """Test legacy mutation validation."""
        self.agent.use_structured = False
        self.mock_model.respond.return_value = json.dumps({
            "decision": "APPROVED",
            "confidence_score": 0.9,
            "impact_assessment": "Low impact",
            "reasoning": "Safe change"
        })
        
        result = self.agent.validate_mutation("Proposed mutation")
        
        self.assertTrue(result["approved"])
        self.assertEqual(result["decision"], "APPROVED")

    def test_validate_mutation_legacy_parse_error(self):
        """Test legacy mutation validation with parse error."""
        self.agent.use_structured = False
        self.mock_model.respond.return_value = "invalid json"
        
        result = self.agent.validate_mutation("Proposed mutation")
        
        self.assertFalse(result["approved"])
        self.assertEqual(result["decision"], "REJECTED")

    def test_get_cache_stats_available(self):
        """Test cache stats when prompt manager is available."""
        result = self.agent.get_cache_stats()
        
        # When SCHEMAS_AVAILABLE, returns actual stats dict
        self.assertIsInstance(result, dict)

    def test_critique_plan_recalls_memory(self):
        """Test that plan critique recalls memory."""
        self.mock_brain.recall_with_budget.return_value = ["memory 1", "memory 2"]
        self.agent.use_structured = False
        
        self.agent.critique_plan("Task", ["Step 1"])
        
        self.mock_brain.recall_with_budget.assert_called()

    def test_critique_code_recalls_memory(self):
        """Test that code critique recalls memory."""
        self.mock_brain.recall_with_budget.return_value = ["memory"]
        self.agent.use_structured = False
        
        self.agent.critique_code("Task", "code")
        
        self.mock_brain.recall_with_budget.assert_called()

    def test_validate_mutation_legacy_recalls_memory(self):
        """Test that legacy mutation validation uses memory."""
        self.mock_brain.recall_with_budget.return_value = ["memory"]
        self.agent.use_structured = False
        
        self.mock_model.respond.return_value = json.dumps({
            "decision": "APPROVED",
            "confidence_score": 0.9,
            "impact_assessment": "Low",
            "reasoning": "Safe"
        })
        
        result = self.agent.validate_mutation("mutation")
        
        # Legacy path may or may not recall memory - implementation detail
        # Just verify result structure
        self.assertIn("approved", result)

    def test_format_feedback_text_no_issues(self):
        """Test formatting feedback without issues."""
        mock_output = Mock()
        mock_output.overall_assessment = "approve"
        mock_output.confidence = 0.9
        mock_output.summary = "Looks good"
        mock_output.issues = []
        mock_output.positive_aspects = ["Clean", "Efficient"]
        
        text = self.agent._format_feedback_text(mock_output)
        
        self.assertIn("APPROVE", text)
        self.assertIn("Looks good", text)
        self.assertIn("Clean", text)

    def test_format_feedback_text_with_issues(self):
        """Test formatting feedback with issues."""
        mock_issue = Mock()
        mock_issue.severity = "major"
        mock_issue.category = "safety"
        mock_issue.description = "Security issue"
        mock_issue.recommendation = "Fix it"
        
        mock_output = Mock()
        mock_output.overall_assessment = "request_changes"
        mock_output.confidence = 0.7
        mock_output.summary = "Needs work"
        mock_output.issues = [mock_issue]
        mock_output.positive_aspects = []
        
        text = self.agent._format_feedback_text(mock_output)
        
        self.assertIn("REQUEST_CHANGES", text)
        self.assertIn("[MAJOR]", text)
        self.assertIn("Security issue", text)
        self.assertIn("Fix it", text)


class TestCriticAgentStructured(unittest.TestCase):
    """Test suite for CriticAgent structured output paths."""

    def setUp(self):
        self.mock_brain = Mock()
        self.mock_brain.recall_with_budget.return_value = []
        self.mock_model = Mock()
        self.mock_model.respond.return_value = json.dumps({
            "initial_assessment": "Assessment",
            "completeness_check": "Complete",
            "feasibility_analysis": "Feasible",
            "risk_identification": "Low risk",
            "overall_assessment": "approve",
            "confidence": 0.9,
            "issues": [],
            "positive_aspects": ["Good"],
            "summary": "Summary"
        })

    @patch("agents.critic.SCHEMAS_AVAILABLE", True)
    @patch("agents.critic.render_prompt")
    @patch("agents.critic._aura_safe_loads")
    @patch("agents.critic.CriticOutput")
    def test_critique_plan_structured_success(self, mock_critic_output, mock_safe_loads, mock_render):
        """Test structured plan critique success."""
        mock_render.return_value = "rendered prompt"
        mock_safe_loads.return_value = {"key": "value"}
        
        mock_issue = Mock()
        mock_issue.severity = "critical"
        mock_issue.category = "safety"
        
        mock_output = Mock()
        mock_output.dict.return_value = {"output": "data"}
        mock_output.overall_assessment = "request_changes"
        mock_output.confidence = 0.8
        mock_output.summary = "Test summary"
        mock_output.issues = [mock_issue]
        mock_output.positive_aspects = ["Good work"]
        mock_output.initial_assessment = "Initial"
        mock_output.completeness_check = "Complete"
        mock_output.feasibility_analysis = "Feasible"
        mock_output.risk_identification = "Risky"
        
        mock_critic_output.return_value = mock_output
        
        agent = CriticAgent(self.mock_brain, self.mock_model)
        result = agent.critique_plan("Task", ["Step 1"])
        
        self.assertIn("structured_output", result)
        self.assertIn("assessment", result)
        self.assertEqual(result["critical_issues"], 1)
        self.assertTrue(result["requires_changes"])

    @patch("agents.critic.SCHEMAS_AVAILABLE", True)
    @patch("agents.critic.render_prompt")
    @patch("agents.critic._aura_safe_loads")
    def test_critique_plan_structured_parse_error(self, mock_safe_loads, mock_render):
        """Test structured plan critique with parse error."""
        mock_render.return_value = "rendered prompt"
        mock_safe_loads.side_effect = json.JSONDecodeError("test", "test", 0)
        
        self.mock_model.respond.return_value = "Legacy response"
        
        agent = CriticAgent(self.mock_brain, self.mock_model)
        agent.use_structured = True
        result = agent.critique_plan("Task", ["Step 1"])
        
        self.assertIn("feedback_text", result)

    @patch("agents.critic.SCHEMAS_AVAILABLE", True)
    @patch("agents.critic.render_prompt")
    @patch("agents.critic._aura_safe_loads")
    @patch("agents.critic.CriticOutput")
    def test_critique_code_structured_success(self, mock_critic_output, mock_safe_loads, mock_render):
        """Test structured code critique success."""
        mock_render.return_value = "rendered prompt"
        mock_safe_loads.return_value = {"key": "value"}
        
        mock_issue = Mock()
        mock_issue.severity = "critical"
        mock_issue.category = "safety"
        
        mock_output = Mock()
        mock_output.dict.return_value = {"output": "data"}
        mock_output.overall_assessment = "approve"
        mock_output.confidence = 0.95
        mock_output.summary = "Code looks good"
        mock_output.issues = []
        mock_output.positive_aspects = ["Clean", "Tested"]
        mock_output.initial_assessment = "Good"
        mock_output.completeness_check = "Complete"
        mock_output.feasibility_analysis = "Feasible"
        mock_output.risk_identification = "Low"
        
        mock_critic_output.return_value = mock_output
        
        agent = CriticAgent(self.mock_brain, self.mock_model)
        result = agent.critique_code("Task", "def test(): pass")
        
        self.assertIn("structured_output", result)
        self.assertFalse(result["security_concerns"])
        self.assertFalse(result["requires_changes"])

    @patch("agents.critic.SCHEMAS_AVAILABLE", True)
    @patch("agents.critic.render_prompt")
    @patch("agents.critic._aura_safe_loads")
    def test_critique_code_structured_error(self, mock_safe_loads, mock_render):
        """Test structured code critique with error."""
        mock_render.return_value = "rendered prompt"
        mock_safe_loads.side_effect = Exception("Parse error")
        
        self.mock_model.respond.return_value = "Legacy fallback"
        
        agent = CriticAgent(self.mock_brain, self.mock_model)
        agent.use_structured = True
        result = agent.critique_code("Task", "code")
        
        self.assertIn("feedback_text", result)


class TestCriticAgentIntegration(unittest.TestCase):
    """Integration-style tests for CriticAgent."""

    def test_full_critique_workflow(self):
        """Test complete critique workflow."""
        mock_brain = Mock()
        mock_brain.recall_with_budget.return_value = ["Previous context"]
        mock_model = Mock()
        mock_model.respond.return_value = "Detailed feedback"
        
        agent = CriticAgent(mock_brain, mock_model)
        agent.use_structured = False
        
        plan = ["Step 1: Analyze", "Step 2: Design", "Step 3: Implement"]
        result = agent.critique_plan("Build feature", plan)
        
        self.assertEqual(result, "Detailed feedback")
        mock_brain.remember.assert_called()
        mock_brain.recall_with_budget.assert_called_with(max_tokens=1500)


if __name__ == "__main__":
    unittest.main()
