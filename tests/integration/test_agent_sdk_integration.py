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


class TestProductionLoopIntegration(unittest.TestCase):
    """Test the full production loop assembly."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_feedback_chain(self):
        """Goal → router → workflow → session → feedback."""
        from core.agent_sdk.config import AgentSDKConfig
        from core.agent_sdk.model_router import AdaptiveModelRouter
        from core.agent_sdk.workflow_templates import WorkflowExecutor, get_builtin_templates
        from core.agent_sdk.session_persistence import SessionStore
        from core.agent_sdk.feedback import SkillWeightUpdater, FeedbackCollector

        config = AgentSDKConfig(
            model_stats_path=Path(self.tmpdir) / "stats.json",
            session_db_path=Path(self.tmpdir) / "sessions.db",
            skill_weights_path=Path(self.tmpdir) / "weights.json",
        )

        router = AdaptiveModelRouter(stats_path=config.model_stats_path)
        session_store = SessionStore(db_path=config.session_db_path)
        skill_updater = SkillWeightUpdater(weights_path=config.skill_weights_path)
        feedback = FeedbackCollector(
            model_router=router, skill_updater=skill_updater,
            brain=None, session_store=session_store,
        )

        # 1. Router selects default model
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

        # 2. Create session
        pk = session_store.create_session("test-1", "Fix bug", "bug_fix", "bug_fix", model)
        self.assertGreater(pk, 0)

        # 3. Record a cycle event
        session_store.record_event(pk, "analyze_goal", "analyze_goal", model, 500, 200, True, None)

        # 4. Trigger feedback
        result = feedback.on_goal_complete(
            session_pk=pk, goal="Fix bug", goal_type="bug_fix",
            model=model, skills_used=["linter", "type_checker"],
            success=True, verification_result={"passed": True}, cost=0.05,
        )
        self.assertTrue(result["model_updated"])

        # 5. Router should now have stats
        stats = router.get_stats()
        self.assertIn("bug_fix", stats)

        # 6. Skill weights should be updated
        weights = skill_updater.get_weights()
        self.assertIn("linter", weights)

    def test_workflow_executor_with_mock_handlers(self):
        """Execute a workflow with mock tool handlers."""
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase,
        )

        success_handler = lambda args: {"result": "ok"}
        handlers = {
            "analyze_goal": success_handler,
            "create_plan": success_handler,
            "generate_code": success_handler,
            "verify_changes": success_handler,
        }
        wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="generate_code"),
                WorkflowPhase(tool_name="verify_changes"),
            ],
        )
        executor = WorkflowExecutor(templates={"test": wf}, tool_handlers=handlers)
        result = executor.execute(wf, goal="Test", context={})
        self.assertTrue(result.success)
        self.assertEqual(result.phases_completed, 4)


if __name__ == "__main__":
    unittest.main()
