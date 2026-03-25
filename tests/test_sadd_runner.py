"""Tests for SubAgentRunner from core.sadd.sub_agent_runner."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.sadd.sub_agent_runner import SubAgentRunner
from core.sadd.types import WorkstreamResult, WorkstreamSpec
from core.sadd.workstream_graph import WorkstreamNode
from memory.brain import Brain


def _make_node(ws_id="ws_test", goal="Test goal", depends_on=None):
    spec = WorkstreamSpec(id=ws_id, title=ws_id, goal_text=goal, depends_on=depends_on or [])
    return WorkstreamNode(spec=spec)


def _mock_orchestrator(stop_reason="PASS"):
    orch = MagicMock()
    orch.run_loop.return_value = {
        "goal": "test",
        "stop_reason": stop_reason,
        "history": [
            {
                "phase_outputs": {
                    "verification": {"status": "pass" if stop_reason == "PASS" else "fail"},
                }
            }
        ],
    }
    return orch


class TestSubAgentRunner(unittest.TestCase):
    """Tests for SubAgentRunner."""

    def setUp(self):
        self.tmp_dir = Path(__file__).parent / "_tmp_brain_runner"
        self.tmp_dir.mkdir(exist_ok=True)
        self.brain = Brain(db_path=str(self.tmp_dir / "test_runner.db"))

    def tearDown(self):
        self.brain.db.close()
        import shutil

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_runner(self, node=None, stop_reason="PASS", deps=None):
        node = node or _make_node()
        orch = _mock_orchestrator(stop_reason)
        factory = MagicMock(return_value=orch)
        runner = SubAgentRunner(
            workstream=node,
            orchestrator_factory=factory,
            brain=self.brain,
            context_from_dependencies=deps or {},
        )
        return runner, orch, factory

    def test_run_successful(self):
        """Mock orchestrator returns PASS — verify result.status == 'completed'."""
        runner, orch, _ = self._make_runner(stop_reason="PASS")
        result = runner.run(max_cycles=5, dry_run=False)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.cycles_used, 1)
        self.assertEqual(result.ws_id, "ws_test")
        self.assertEqual(result.stop_reason, "PASS")

    def test_run_failed(self):
        """Mock orchestrator returns MAX_CYCLES — verify result.status == 'failed'."""
        runner, orch, _ = self._make_runner(stop_reason="MAX_CYCLES")
        result = runner.run(max_cycles=5, dry_run=False)

        self.assertEqual(result.status, "failed")

    def test_context_injection_passed(self):
        """Verify orchestrator.run_loop receives context_injection with sadd_dependencies."""
        dep_result = WorkstreamResult(ws_id="dep_a", status="completed")
        deps = {"dep_a": dep_result}
        runner, orch, _ = self._make_runner(deps=deps)
        runner.run()

        call_kwargs = orch.run_loop.call_args
        # context_injection should contain "sadd_dependencies"
        context_injection = call_kwargs.kwargs.get("context_injection") or call_kwargs[1].get("context_injection")
        self.assertIn("sadd_dependencies", context_injection)
        self.assertIn("dep_a", context_injection["sadd_dependencies"])

    def test_enriched_goal_includes_deps(self):
        """With dependency context, verify the goal starts with [SADD Context prefix."""
        dep_result = WorkstreamResult(ws_id="dep_a", status="completed")
        deps = {"dep_a": dep_result}
        node = _make_node(goal="Build feature X")
        runner, orch, _ = self._make_runner(node=node, deps=deps)
        runner.run()

        enriched_goal = orch.run_loop.call_args[0][0]
        self.assertTrue(enriched_goal.startswith("[SADD Context"))

    def test_dependency_context_written_to_brain(self):
        """With deps, verify brain.recall_tagged returns entries for session tag."""
        dep_result = WorkstreamResult(ws_id="dep_a", status="completed", changed_files=["a.py"])
        deps = {"dep_a": dep_result}
        node = _make_node(ws_id="ws_main")
        runner, _, _ = self._make_runner(node=node, deps=deps)
        runner.run()

        tag = "sadd:ws_main"
        entries = self.brain.recall_tagged(tag)
        self.assertGreater(len(entries), 0)
        self.assertTrue(any("dep_a" in e for e in entries))

    def test_run_handles_exception(self):
        """Mock orchestrator.run_loop raises RuntimeError — verify failed result."""
        node = _make_node()
        orch = MagicMock()
        orch.run_loop.side_effect = RuntimeError("orchestrator crashed")
        factory = MagicMock(return_value=orch)
        runner = SubAgentRunner(
            workstream=node,
            orchestrator_factory=factory,
            brain=self.brain,
            context_from_dependencies={},
        )
        result = runner.run()

        self.assertEqual(result.status, "failed")
        self.assertIn("orchestrator crashed", result.error)

    def test_run_no_dependencies(self):
        """With empty deps, goal is just the original goal_text without prefix."""
        node = _make_node(goal="Simple task")
        runner, orch, _ = self._make_runner(node=node, deps={})
        runner.run()

        enriched_goal = orch.run_loop.call_args[0][0]
        self.assertEqual(enriched_goal, "Simple task")


if __name__ == "__main__":
    unittest.main()
