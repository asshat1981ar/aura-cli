"""Tests for workflow templates and executor."""
import enum
import unittest
from unittest.mock import MagicMock


class TestWorkflowDataclasses(unittest.TestCase):
    """Test workflow phase and template dataclasses."""

    def test_workflow_phase_defaults(self):
        from core.agent_sdk.workflow_templates import WorkflowPhase
        phase = WorkflowPhase(tool_name="analyze_goal")
        self.assertTrue(phase.required)
        self.assertEqual(phase.retry_on_fail, 0)
        self.assertFalse(phase.escalate_on_fail)

    def test_workflow_template_has_phases(self):
        from core.agent_sdk.workflow_templates import WorkflowTemplate, WorkflowPhase
        wf = WorkflowTemplate(
            name="test",
            goal_types=["bug_fix"],
            phases=[WorkflowPhase(tool_name="analyze_goal")],
        )
        self.assertEqual(len(wf.phases), 1)
        self.assertEqual(wf.verification_mode, "post")

    def test_phase_result_fields(self):
        from core.agent_sdk.workflow_templates import PhaseResult
        pr = PhaseResult(tool_name="create_plan", success=True, output={"steps": []})
        self.assertTrue(pr.success)
        self.assertIsNone(pr.error)

    def test_workflow_result_fields(self):
        from core.agent_sdk.workflow_templates import WorkflowResult
        wr = WorkflowResult(success=True, phases_completed=3, phase_results=[])
        self.assertAlmostEqual(wr.total_cost_usd, 0.0)
        self.assertEqual(wr.model_escalations, 0)

    def test_failure_action_enum(self):
        from core.agent_sdk.workflow_templates import FailureAction
        self.assertEqual(FailureAction.RETRY_PHASE.value, "retry_phase")
        self.assertEqual(FailureAction.ABORT.value, "abort")


class TestBuiltinTemplates(unittest.TestCase):
    """Test the three built-in workflow templates."""

    def test_get_builtin_templates_returns_three(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        templates = get_builtin_templates()
        self.assertEqual(len(templates), 3)
        self.assertIn("bug_fix", templates)
        self.assertIn("feature", templates)
        self.assertIn("refactor", templates)

    def test_bug_fix_has_correct_phases(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        wf = get_builtin_templates()["bug_fix"]
        names = [p.tool_name for p in wf.phases]
        self.assertEqual(names[0], "analyze_goal")
        self.assertIn("generate_code", names)
        self.assertIn("verify_changes", names)
        self.assertIn("reflect_on_outcome", names)

    def test_feature_includes_critique(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        wf = get_builtin_templates()["feature"]
        names = [p.tool_name for p in wf.phases]
        self.assertIn("critique_plan", names)

    def test_refactor_has_pre_and_post_verification(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        wf = get_builtin_templates()["refactor"]
        self.assertEqual(wf.verification_mode, "pre_and_post")


class TestWorkflowExecutor(unittest.TestCase):
    """Test workflow execution engine."""

    def test_select_workflow_by_goal_type(self):
        from core.agent_sdk.workflow_templates import WorkflowExecutor, get_builtin_templates
        executor = WorkflowExecutor(templates=get_builtin_templates(), tool_handlers={})
        wf = executor.select_workflow("bug_fix")
        self.assertEqual(wf.name, "bug_fix")

    def test_select_workflow_falls_back_to_feature(self):
        from core.agent_sdk.workflow_templates import WorkflowExecutor, get_builtin_templates
        executor = WorkflowExecutor(templates=get_builtin_templates(), tool_handlers={})
        wf = executor.select_workflow("unknown_type")
        self.assertEqual(wf.name, "feature")

    def test_execute_runs_all_phases(self):
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase, get_builtin_templates,
        )
        mock_handler = MagicMock(return_value={"success": True})
        handlers = {"analyze_goal": mock_handler, "store_memory": mock_handler}
        simple_wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="store_memory"),
            ],
        )
        executor = WorkflowExecutor(templates={"test": simple_wf}, tool_handlers=handlers)
        result = executor.execute(simple_wf, goal="test goal", context={})
        self.assertTrue(result.success)
        self.assertEqual(result.phases_completed, 2)

    def test_execute_handles_phase_failure(self):
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase,
        )
        fail_handler = MagicMock(return_value={"error": "broken"})
        handlers = {"bad_tool": fail_handler}
        wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[WorkflowPhase(tool_name="bad_tool")],
            max_retries_total=0,
        )
        executor = WorkflowExecutor(templates={"test": wf}, tool_handlers=handlers)
        result = executor.execute(wf, goal="test", context={})
        self.assertFalse(result.success)

    def test_execute_retries_phase(self):
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase,
        )
        call_count = {"n": 0}
        def flaky_handler(args):
            call_count["n"] += 1
            if call_count["n"] < 2:
                return {"error": "transient"}
            return {"result": "ok"}

        handlers = {"flaky": flaky_handler}
        wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[WorkflowPhase(tool_name="flaky", retry_on_fail=2)],
        )
        executor = WorkflowExecutor(templates={"test": wf}, tool_handlers=handlers)
        result = executor.execute(wf, goal="test", context={})
        self.assertTrue(result.success)
        self.assertEqual(call_count["n"], 2)


if __name__ == "__main__":
    unittest.main()
