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


class TestContextBuilderV2(unittest.TestCase):
    """Tests for enhanced context builder with failure_patterns, skill_weights, workflow_info, model_tier."""

    def test_failure_patterns_rendered_in_prompt(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        context = {
            "failure_patterns": ["timeout on large files", "import error in test env"],
        }
        prompt = builder.build_system_prompt(
            goal="Fix auth bug", goal_type="bug_fix", context=context
        )
        self.assertIn("Failure Patterns", prompt)
        self.assertIn("timeout on large files", prompt)
        self.assertIn("import error in test env", prompt)

    def test_skill_weights_rendered_sorted_descending(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        context = {
            "skill_weights": {"linter": 0.5, "type_checker": 0.9, "coverage": 0.7},
        }
        prompt = builder.build_system_prompt(
            goal="Refactor module", goal_type="refactor", context=context
        )
        self.assertIn("Skill Weights", prompt)
        # type_checker (0.9) should appear before linter (0.5) in sorted order
        # Use " (0.9)" and " (0.5)" markers to avoid substring collisions
        idx_type = prompt.index("type_checker (0.9)")
        idx_linter = prompt.index("linter (0.5)")
        self.assertLess(idx_type, idx_linter)

    def test_workflow_info_and_model_tier_rendered(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        context = {
            "workflow_info": "refactor-standard v1.2",
            "model_tier": "quality",
        }
        prompt = builder.build_system_prompt(
            goal="Refactor utils", goal_type="refactor", context=context
        )
        self.assertIn("Workflow", prompt)
        self.assertIn("refactor-standard v1.2", prompt)
        self.assertIn("Model Tier", prompt)
        self.assertIn("quality", prompt)


class TestContextBuilderSemantic(unittest.TestCase):
    """Test semantic querier integration in context builder."""

    def test_codebase_context_with_querier(self):
        from core.agent_sdk.context_builder import ContextBuilder
        mock_querier = MagicMock()
        mock_querier.architecture_overview.return_value = {
            "total_files": 50, "clusters": {"core": 20}, "summary": "50 files...",
            "top_coupled": [],
        }
        mock_querier.find_similar.return_value = [
            {"name": "auth", "path": "core/auth.py"}
        ]
        mock_querier.what_depends_on.return_value = [
            {"path": "core/server.py", "rel_type": "imports"}
        ]
        builder = ContextBuilder(
            project_root=Path("/tmp/test"),
            semantic_querier=mock_querier,
        )
        ctx = builder.build(goal="Fix auth bug")
        self.assertIn("codebase_overview", ctx)
        self.assertIn("relevant_symbols", ctx)

    def test_codebase_context_without_querier(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = builder.build(goal="Fix auth bug")
        self.assertNotIn("codebase_overview", ctx)

    def test_prompt_renders_codebase_understanding(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = {
            "recommended_skills": [],
            "codebase_overview": {
                "summary": "200 files in 4 clusters",
                "total_files": 200,
            },
            "relevant_symbols": [
                {"name": "auth_func", "path": "core/auth.py", "intent_summary": "Handles auth"}
            ],
        }
        prompt = builder.build_system_prompt(goal="Fix bug", goal_type="bug_fix", context=ctx)
        self.assertIn("Codebase Understanding", prompt)
        self.assertIn("200 files", prompt)


if __name__ == "__main__":
    unittest.main()
