"""Tests for SessionCoordinator from core.sadd.session_coordinator."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.sadd.types import (
    DesignSpec,
    SessionConfig,
    WorkstreamResult,
    WorkstreamSpec,
)
from core.sadd.session_coordinator import SessionCoordinator
from memory.brain import Brain


def _make_spec(workstreams):
    return DesignSpec(
        title="Test Design",
        summary="Test summary",
        workstreams=workstreams,
    )


def _completed_result(ws_id):
    return WorkstreamResult(
        ws_id=ws_id,
        status="completed",
        cycles_used=2,
        stop_reason="PASS",
        elapsed_s=0.1,
    )


def _failed_result(ws_id):
    return WorkstreamResult(
        ws_id=ws_id,
        status="failed",
        cycles_used=1,
        stop_reason="MAX_CYCLES",
        error="timed out",
        elapsed_s=0.1,
    )


class TestSessionCoordinator(unittest.TestCase):
    """Tests for SessionCoordinator."""

    def setUp(self):
        self.tmp_dir = Path(__file__).parent / "_tmp_brain_coord"
        self.tmp_dir.mkdir(exist_ok=True)
        self.brain = Brain(db_path=str(self.tmp_dir / "test_coord.db"))
        self.orchestrator_factory = MagicMock()

    def tearDown(self):
        self.brain.db.close()
        import shutil

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _patch_runner(self, side_effect_fn):
        """Return a patch context that replaces SubAgentRunner in session_coordinator.

        side_effect_fn(workstream_spec, dependency_context) -> WorkstreamResult
        is called whenever runner.run() is invoked.
        """

        def make_mock_runner(**kwargs):
            ws_node = kwargs.get("workstream")
            ws_spec = ws_node.spec if ws_node else kwargs.get("workstream_spec")
            dep_ctx = kwargs.get("context_from_dependencies", kwargs.get("dependency_context", {}))
            mock_runner = MagicMock()
            mock_runner.run.return_value = side_effect_fn(ws_spec, dep_ctx)
            # Store the init kwargs for later inspection
            mock_runner._init_kwargs = kwargs
            return mock_runner

        return patch(
            "core.sadd.session_coordinator.SubAgentRunner",
            side_effect=make_mock_runner,
        )

    def test_single_workstream_session(self):
        """Single workstream completes — report shows 1 total, 1 completed, 0 failed."""
        ws = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        spec = _make_spec([ws])

        def runner_fn(ws_spec, dep_ctx):
            return _completed_result(ws_spec.id)

        with self._patch_runner(runner_fn):
            coord = SessionCoordinator(
                spec,
                self.orchestrator_factory,
                self.brain,
                config=SessionConfig(retry_failed=False),
            )
            report = coord.run()

        self.assertEqual(report.total_workstreams, 1)
        self.assertEqual(report.completed, 1)
        self.assertEqual(report.failed, 0)

    def test_independent_workstreams_parallel(self):
        """Two independent workstreams both complete."""
        ws_a = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        ws_b = WorkstreamSpec(id="ws_b", title="B", goal_text="Do B")
        spec = _make_spec([ws_a, ws_b])

        def runner_fn(ws_spec, dep_ctx):
            return _completed_result(ws_spec.id)

        with self._patch_runner(runner_fn):
            coord = SessionCoordinator(
                spec,
                self.orchestrator_factory,
                self.brain,
                config=SessionConfig(retry_failed=False),
            )
            report = coord.run()

        self.assertEqual(report.total_workstreams, 2)
        self.assertEqual(report.completed, 2)
        self.assertEqual(report.failed, 0)

    def test_dependency_ordering(self):
        """A -> B dependency: B receives A's result in context_from_dependencies."""
        ws_a = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        ws_b = WorkstreamSpec(id="ws_b", title="B", goal_text="Do B", depends_on=["ws_a"])
        spec = _make_spec([ws_a, ws_b])

        captured_dep_contexts = {}

        def runner_fn(ws_spec, dep_ctx):
            captured_dep_contexts[ws_spec.id] = dict(dep_ctx)
            return _completed_result(ws_spec.id)

        with self._patch_runner(runner_fn):
            coord = SessionCoordinator(
                spec,
                self.orchestrator_factory,
                self.brain,
                config=SessionConfig(retry_failed=False, max_parallel=1),
            )
            report = coord.run()

        self.assertEqual(report.completed, 2)
        # ws_b should have received ws_a's result as dependency context
        self.assertIn("ws_a", captured_dep_contexts.get("ws_b", {}))
        # ws_a should have no dependencies
        self.assertEqual(len(captured_dep_contexts.get("ws_a", {})), 0)

    def test_failed_workstream_blocks_dependents(self):
        """A fails -> B (depends on A) should be blocked/skipped."""
        ws_a = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        ws_b = WorkstreamSpec(id="ws_b", title="B", goal_text="Do B", depends_on=["ws_a"])
        spec = _make_spec([ws_a, ws_b])

        def runner_fn(ws_spec, dep_ctx):
            if ws_spec.id == "ws_a":
                return _failed_result(ws_spec.id)
            return _completed_result(ws_spec.id)

        with self._patch_runner(runner_fn):
            coord = SessionCoordinator(
                spec,
                self.orchestrator_factory,
                self.brain,
                config=SessionConfig(retry_failed=False),
            )
            report = coord.run()

        self.assertEqual(report.failed, 1)
        self.assertEqual(report.skipped, 1)

    def test_dry_run_mode(self):
        """With config.dry_run=True, verify dry_run is passed through to runner."""
        ws = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        spec = _make_spec([ws])

        captured_dry_run = {}

        with patch("core.sadd.session_coordinator.SubAgentRunner") as MockRunner:
            mock_instance = MagicMock()
            mock_instance.run.return_value = _completed_result("ws_a")
            MockRunner.return_value = mock_instance

            coord = SessionCoordinator(
                spec,
                self.orchestrator_factory,
                self.brain,
                config=SessionConfig(dry_run=True, retry_failed=False),
            )
            report = coord.run()

            # Check that runner.run() was called with dry_run=True
            run_call = mock_instance.run.call_args
            self.assertTrue(run_call.kwargs.get("dry_run", False) or (len(run_call.args) > 1 and run_call.args[1] is True))

    def test_fail_fast_stops_session(self):
        """With fail_fast=True, first failure stops remaining work."""
        ws_a = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        ws_b = WorkstreamSpec(id="ws_b", title="B", goal_text="Do B")
        spec = _make_spec([ws_a, ws_b])

        call_count = {"n": 0}

        def runner_fn(ws_spec, dep_ctx):
            call_count["n"] += 1
            # First workstream always fails
            return _failed_result(ws_spec.id)

        with self._patch_runner(runner_fn):
            coord = SessionCoordinator(
                spec,
                self.orchestrator_factory,
                self.brain,
                config=SessionConfig(fail_fast=True, retry_failed=False, max_parallel=1),
            )
            report = coord.run()

        # At least one must have failed; fail_fast should limit execution
        self.assertGreaterEqual(report.failed, 1)

    def test_session_report_structure(self):
        """Verify report has correct fields: session_id, design_title, totals, outcomes, elapsed_s."""
        ws = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        spec = _make_spec([ws])

        def runner_fn(ws_spec, dep_ctx):
            return _completed_result(ws_spec.id)

        with self._patch_runner(runner_fn):
            coord = SessionCoordinator(
                spec,
                self.orchestrator_factory,
                self.brain,
                config=SessionConfig(retry_failed=False),
            )
            report = coord.run()

        self.assertIsInstance(report.session_id, str)
        self.assertEqual(report.design_title, "Test Design")
        self.assertEqual(report.total_workstreams, 1)
        self.assertIsInstance(report.outcomes, list)
        self.assertEqual(len(report.outcomes), 1)
        self.assertGreaterEqual(report.elapsed_s, 0.0)
        self.assertEqual(report.outcomes[0].id, "ws_a")
        self.assertEqual(report.outcomes[0].status, "completed")

    def test_status_returns_dict(self):
        """Call status() before and after run, verify it returns a dict with expected keys."""
        ws = WorkstreamSpec(id="ws_a", title="A", goal_text="Do A")
        spec = _make_spec([ws])

        coord = SessionCoordinator(
            spec,
            self.orchestrator_factory,
            self.brain,
            config=SessionConfig(retry_failed=False),
        )

        # Before run
        status_before = coord.status()
        self.assertIsInstance(status_before, dict)
        self.assertIn("session_id", status_before)
        self.assertEqual(status_before["state"], "not_started")

        # After run
        def runner_fn(ws_spec, dep_ctx):
            return _completed_result(ws_spec.id)

        with self._patch_runner(runner_fn):
            coord.run()

        status_after = coord.status()
        self.assertIsInstance(status_after, dict)
        self.assertIn("session_id", status_after)
        self.assertIn("total", status_after)
        self.assertIn("completed", status_after)


if __name__ == "__main__":
    unittest.main()
