# tests/test_agent_sdk_context_builder.py
"""Tests for Agent SDK context builder."""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestContextBuilder(unittest.TestCase):
    """Test context assembly from AURA subsystems."""

    def test_build_basic_context(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Add unit tests for auth module")
        self.assertIn("goal", ctx)
        self.assertEqual(ctx["goal"], "Add unit tests for auth module")
        self.assertIn("goal_type", ctx)
        self.assertIn("project_root", ctx)

    def test_goal_classification(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        self.assertEqual(builder.classify_goal("Fix null pointer in login"), "bug_fix")
        self.assertEqual(builder.classify_goal("Add OAuth2 support"), "feature")
        self.assertEqual(builder.classify_goal("Extract helper methods"), "refactor")
        self.assertEqual(builder.classify_goal("Scan for SQL injection"), "security")

    def test_skill_recommendations(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Fix authentication bug")
        self.assertIn("recommended_skills", ctx)
        self.assertIsInstance(ctx["recommended_skills"], list)
        self.assertGreater(len(ctx["recommended_skills"]), 0)

    def test_memory_hints_with_brain(self):
        from core.agent_sdk.context_builder import ContextBuilder

        mock_brain = MagicMock()
        # Brain.recall_with_budget(max_tokens) returns List[str]
        mock_brain.recall_with_budget.return_value = [
            "Auth uses JWT tokens with 30min expiry",
            "Token refresh implemented in core/auth.py",
        ]
        builder = ContextBuilder(
            project_root=Path("/tmp/test-project"),
            brain=mock_brain,
        )
        ctx = builder.build(goal="Fix auth token expiry")
        self.assertIn("memory_hints", ctx)
        self.assertGreater(len(ctx["memory_hints"]), 0)
        mock_brain.recall_with_budget.assert_called_once_with(max_tokens=2000)

    def test_memory_hints_without_brain(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Fix auth bug")
        self.assertIn("memory_hints", ctx)
        self.assertEqual(ctx["memory_hints"], [])

    def test_mcp_tool_summary(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Lint and format codebase")
        self.assertIn("available_mcp_categories", ctx)
        self.assertIsInstance(ctx["available_mcp_categories"], list)

    def test_build_system_prompt(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        prompt = builder.build_system_prompt(goal="Add tests", goal_type="feature")
        self.assertIn("AURA", prompt)
        self.assertIn("feature", prompt)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)


if __name__ == "__main__":
    unittest.main()
