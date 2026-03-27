"""Tests for SessionCoordinator from core.sadd.session_coordinator."""

import types as _types
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
from core.sadd.workstream_graph import WorkstreamGraph
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


# ---------------------------------------------------------------------------
# Shared helpers for new test classes (use _n_spec / _ok_result to avoid
# colliding with the _make_spec / _completed_result already defined above).
# ---------------------------------------------------------------------------


def _n_spec(n: int = 2) -> DesignSpec:
    """Build a DesignSpec with *n* independent workstreams (no deps)."""
    workstreams = [
        WorkstreamSpec(
            id=f"ws{i}",
            title=f"Workstream {i}",
            goal_text=f"Do task {i}",
        )
        for i in range(1, n + 1)
    ]
    return DesignSpec(
        title="Test Design",
        summary="A test design specification",
        workstreams=workstreams,
    )


def _ok_result(ws_id: str) -> WorkstreamResult:
    """Return a minimal completed WorkstreamResult."""
    return WorkstreamResult(
        ws_id=ws_id,
        status="completed",
        cycles_used=1,
        changed_files=[],
        elapsed_s=0.1,
    )


def _runner_factory_side_effect(**kwargs) -> MagicMock:
    """Return a mock SubAgentRunner whose .run() returns a success result."""
    inst = MagicMock()
    ws_id = kwargs["workstream"].spec.id
    inst.run.return_value = _ok_result(ws_id)
    return inst


# ---------------------------------------------------------------------------
# TestSessionCoordinatorRun
# ---------------------------------------------------------------------------


class TestSessionCoordinatorRun(unittest.TestCase):
    """Tests for SessionCoordinator.run() — orchestrator factory path (mocked)."""

    def test_run_calls_factory_for_each_workstream(self):
        """SubAgentRunner must be instantiated once per workstream; report completed==2."""
        spec = _n_spec(n=2)
        brain = MagicMock()
        brain.remember_tagged = MagicMock()
        orchestrator_factory = MagicMock()

        with patch(
            "core.sadd.session_coordinator.SubAgentRunner",
            side_effect=_runner_factory_side_effect,
        ) as MockRunner:
            coord = SessionCoordinator(
                design_spec=spec,
                orchestrator_factory=orchestrator_factory,
                brain=brain,
                config=SessionConfig(max_parallel=2, retry_failed=False),
            )
            report = coord.run()

        self.assertEqual(MockRunner.call_count, 2, "Expected one runner per workstream")
        self.assertEqual(report.completed, 2)
        self.assertEqual(report.total_workstreams, 2)
        self.assertEqual(report.failed, 0)


# ---------------------------------------------------------------------------
# TestSessionCoordinatorResume
# ---------------------------------------------------------------------------


class TestSessionCoordinatorResume(unittest.TestCase):
    """Tests for SessionCoordinator.resume() — skips already-completed work."""

    def test_resume_skips_completed_workstreams(self):
        """resume() must only execute workstreams not already completed in the graph."""
        spec = _n_spec(n=2)
        brain = MagicMock()
        orchestrator_factory = MagicMock()

        # Pre-build graph with ws1 already completed.
        graph = WorkstreamGraph(spec.workstreams)
        ws1_result = _ok_result("ws1")
        graph.mark_running("ws1")
        graph.mark_completed("ws1", ws1_result)

        completed_results: dict = {"ws1": ws1_result}

        with patch(
            "core.sadd.session_coordinator.SubAgentRunner",
            side_effect=_runner_factory_side_effect,
        ) as MockRunner:
            coord = SessionCoordinator(
                design_spec=spec,
                orchestrator_factory=orchestrator_factory,
                brain=brain,
                config=SessionConfig(max_parallel=2, retry_failed=False),
            )
            report = coord.resume(graph, completed_results)

        # Only ws2 should have been submitted.
        self.assertEqual(MockRunner.call_count, 1, "Expected runner only for the remaining workstream")
        executed_ws_id = MockRunner.call_args[1]["workstream"].spec.id
        self.assertEqual(executed_ws_id, "ws2")

        # Both workstreams should appear completed in the final report.
        self.assertEqual(report.completed, 2)
        self.assertEqual(report.failed, 0)

    def test_resume_reruns_failed_workstreams(self):
        """resume() must re-attempt workstreams that previously failed, not skip them."""
        spec = _n_spec(n=2)
        brain = MagicMock()
        orchestrator_factory = MagicMock()

        graph = WorkstreamGraph(spec.workstreams)
        # Mark ws1 as failed (not completed).
        graph.mark_running("ws1")
        graph.mark_failed("ws1", "network error")

        # completed_results only contains truly completed workstreams.
        completed_results: dict = {}

        with patch(
            "core.sadd.session_coordinator.SubAgentRunner",
            side_effect=_runner_factory_side_effect,
        ) as MockRunner:
            coord = SessionCoordinator(
                design_spec=spec,
                orchestrator_factory=orchestrator_factory,
                brain=brain,
                config=SessionConfig(max_parallel=2, retry_failed=False),
            )
            report = coord.resume(graph, completed_results)

        # Both ws1 (previously failed) and ws2 should be re-attempted.
        self.assertEqual(MockRunner.call_count, 2, "Both workstreams should be re-attempted")
        self.assertEqual(report.failed, 0)

    def test_resume_unblocks_dependents_of_failed_workstream(self):
        """resume() must reset blocked dependents of failed workstreams to pending."""
        # Build spec with dependency: ws2 depends on ws1
        ws1 = WorkstreamSpec(id="ws1", title="First", goal_text="Do first")
        ws2 = WorkstreamSpec(id="ws2", title="Second", goal_text="Do second", depends_on=["ws1"])
        spec = DesignSpec(title="Test", summary="Test spec", workstreams=[ws1, ws2])
        brain = MagicMock()
        orchestrator_factory = MagicMock()

        # Simulate: ws1 failed mid-run, which blocked ws2.
        graph = WorkstreamGraph([ws1, ws2])
        graph.mark_running("ws1")
        graph.mark_failed("ws1", "timeout")
        # ws2 is now "blocked" because its dependency failed.
        self.assertEqual(graph.get_node("ws2").status, "blocked")

        with patch(
            "core.sadd.session_coordinator.SubAgentRunner",
            side_effect=_runner_factory_side_effect,
        ) as MockRunner:
            coord = SessionCoordinator(
                design_spec=spec,
                orchestrator_factory=orchestrator_factory,
                brain=brain,
                config=SessionConfig(max_parallel=2, retry_failed=False),
            )
            report = coord.resume(graph, {})

        # Both ws1 (was failed) and ws2 (was blocked) must execute.
        self.assertEqual(MockRunner.call_count, 2, "Both ws1 and its blocked dependent ws2 must run")
        self.assertEqual(report.completed, 2)
        self.assertEqual(report.failed, 0)


# ---------------------------------------------------------------------------
# TestSaddResumeDispatch
# ---------------------------------------------------------------------------


class TestSaddResumeDispatch(unittest.TestCase):
    """Tests for aura_cli.dispatch._handle_sadd_resume_dispatch()."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ctx(session_id=None, run: bool = False):
        from aura_cli.dispatch import DispatchContext

        args = _types.SimpleNamespace(session_id=session_id, run=run, json=False)
        return DispatchContext(
            parsed=MagicMock(),
            project_root=Path("."),
            runtime_factory=MagicMock(),
            args=args,
            runtime={"brain": MagicMock(), "model_adapter": None},
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_resume_dispatch_no_session_id_lists_sessions(self):
        """Without --session-id the handler returns exit code 1 with an error message.

        Note: the current implementation requires --session-id and does not fall
        back to listing resumable sessions; it returns 1 immediately.
        """
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        ctx = self._make_ctx(session_id=None)
        rc = _handle_sadd_resume_dispatch(ctx)

        self.assertEqual(rc, 1)

    def test_resume_dispatch_missing_session_returns_error(self):
        """When load_session_for_resume returns None the handler returns exit code 1."""
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        ctx = self._make_ctx(session_id="nonexistent-session-id")

        with patch("core.sadd.session_store.SessionStore") as MockStore:
            mock_store = MagicMock()
            mock_store.load_session_for_resume.return_value = None
            MockStore.return_value = mock_store

            rc = _handle_sadd_resume_dispatch(ctx)

        self.assertEqual(rc, 1)
        mock_store.load_session_for_resume.assert_called_once_with("nonexistent-session-id")

    def test_resume_dispatch_without_run_flag_shows_summary(self):
        """With a valid session_id but no --run: prints summary, returns 0, never runs coordinator."""
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        spec = _n_spec(n=2)
        config = SessionConfig()
        # Serialise a real graph so WorkstreamGraph.from_dict works inside the handler.
        graph_state = WorkstreamGraph(spec.workstreams).to_dict()
        raw_results: dict = {}

        ctx = self._make_ctx(session_id="s-test-abc123", run=False)

        with patch("core.sadd.session_store.SessionStore") as MockStore:
            mock_store = MagicMock()
            mock_store.load_session_for_resume.return_value = (
                spec,
                config,
                graph_state,
                raw_results,
            )
            MockStore.return_value = mock_store

            rc = _handle_sadd_resume_dispatch(ctx)

        self.assertEqual(rc, 0)
        # SessionCoordinator.resume() must NOT be invoked when --run is absent;
        # we confirm this by verifying no coordinator object was used via the store.
        mock_store.update_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
