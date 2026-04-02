# tests/test_agent_sdk_tool_registry.py
"""Tests for Agent SDK custom MCP tool registry."""
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


class TestToolRegistry(unittest.TestCase):
    """Test custom MCP tool creation wrapping AURA infrastructure."""

    def test_create_tools_returns_list(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_all_required_tools_present(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        tool_names = [t.name for t in tools]
        required = [
            "analyze_goal",
            "dispatch_skills",
            "create_plan",
            "critique_plan",
            "generate_code",
            "run_sandbox",
            "apply_changes",
            "verify_changes",
            "reflect_on_outcome",
            "search_memory",
            "store_memory",
            "manage_goals",
            "discover_mcp_tools",
            "invoke_mcp_tool",
            "run_workflow",
        ]
        for name in required:
            self.assertIn(name, tool_names, f"Missing required tool: {name}")

    def test_tools_have_descriptions(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        for t in tools:
            self.assertTrue(
                len(t.description) > 10,
                f"Tool {t.name} has insufficient description",
            )

    def test_tools_have_input_schemas(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        for t in tools:
            self.assertIsInstance(t.input_schema, dict, f"Tool {t.name} missing schema")
            self.assertIn("type", t.input_schema)


class TestAnalyzeGoalTool(unittest.TestCase):
    """Test the analyze_goal tool handler."""

    def test_analyze_goal_returns_context(self):
        from core.agent_sdk.tool_registry import _handle_analyze_goal

        result = _handle_analyze_goal(
            {"goal": "Fix login bug"},
            project_root=Path("/tmp/test"),
        )
        self.assertIn("goal_type", result)
        self.assertIn("recommended_skills", result)

    def test_analyze_goal_classifies_correctly(self):
        from core.agent_sdk.tool_registry import _handle_analyze_goal

        result = _handle_analyze_goal(
            {"goal": "Refactor the database layer"},
            project_root=Path("/tmp/test"),
        )
        self.assertEqual(result["goal_type"], "refactor")


class TestDispatchSkillsTool(unittest.TestCase):
    """Test the dispatch_skills tool handler."""

    def test_dispatch_with_no_skills_loaded(self):
        from core.agent_sdk.tool_registry import _handle_dispatch_skills

        result = _handle_dispatch_skills(
            {"goal_type": "bug_fix", "project_root": "/tmp/test"},
            project_root=Path("/tmp/test"),
        )
        self.assertIn("skills_run", result)
        self.assertIsInstance(result["skills_run"], list)


if __name__ == "__main__":
    unittest.main()
