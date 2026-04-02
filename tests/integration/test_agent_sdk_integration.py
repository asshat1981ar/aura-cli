# tests/integration/test_agent_sdk_integration.py
"""Integration tests for Agent SDK meta-controller.

These tests verify the full assembly pipeline without requiring
the claude-agent-sdk to be installed — they test everything up to
the actual SDK call.
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestFullAssembly(unittest.TestCase):
    """Test that all components wire together correctly."""

    def test_controller_assembles_all_components(self):
        """Controller should build options with tools, subagents, and hooks."""
        from core.agent_sdk.config import AgentSDKConfig
        from core.agent_sdk.controller import AuraController

        config = AgentSDKConfig(
            model="claude-sonnet-4-6",
            max_turns=10,
            enable_subagents=True,
            enable_hooks=True,
        )
        controller = AuraController(
            config=config,
            project_root=Path("/tmp/test"),
        )

        # Should build without errors
        prompt = controller._build_prompt("Fix the auth bug")
        self.assertIn("Fix the auth bug", prompt)
        self.assertIn("bug_fix", prompt)

        subagents = controller._build_subagent_defs()
        self.assertGreater(len(subagents), 0)

        mcp_server = controller._build_mcp_server()
        self.assertIsNotNone(mcp_server)

    def test_tool_registry_handles_all_tools(self):
        """All registered tools should be callable with minimal args."""
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))

        # analyze_goal should work without any external deps
        analyze = next(t for t in tools if t.name == "analyze_goal")
        result = analyze.handler({"goal": "Add OAuth support"})
        self.assertEqual(result["goal_type"], "feature")
        self.assertIn("recommended_skills", result)

    def test_context_builder_end_to_end(self):
        """Context builder should produce a complete system prompt."""
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = builder.build(goal="Refactor the database layer")
        prompt = builder.build_system_prompt(
            goal="Refactor the database layer",
            goal_type=ctx["goal_type"],
            context=ctx,
        )

        self.assertIn("Refactor the database layer", prompt)
        self.assertIn("refactor", prompt)
        self.assertIn("Recommended Skills", prompt)

    def test_metrics_survive_session(self):
        """Metrics should accumulate across tool calls."""
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("analyze_goal", 0.1, True)
        mc.record_tool_call("create_plan", 2.5, True)
        mc.record_tool_call("generate_code", 5.0, True)
        mc.record_tool_call("verify_changes", 3.0, False)

        summary = mc.get_summary()
        self.assertEqual(summary["total_calls"], 4)
        self.assertEqual(summary["total_successes"], 3)
        self.assertAlmostEqual(summary["success_rate"], 0.75)

        stats = mc.get_stats()
        self.assertEqual(stats["tool_calls"]["analyze_goal"]["count"], 1)
        self.assertEqual(stats["tool_calls"]["verify_changes"]["success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
