"""End-to-end integration test for SADD: parse spec -> build graph -> coordinator with mock orchestrator."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.sadd.design_spec_parser import DesignSpecParser
from core.sadd.session_coordinator import SessionCoordinator
from core.sadd.session_store import SessionStore
from core.sadd.types import SessionConfig, WorkstreamResult
from core.sadd.workstream_graph import WorkstreamGraph
from memory.brain import Brain


class TestSADDE2E(unittest.TestCase):
    def test_full_dry_run_pipeline(self):
        """Parse sample spec -> validate -> build graph -> verify waves."""
        spec_path = Path("tests/fixtures/sadd_sample_spec.md")
        parser = DesignSpecParser()
        spec = parser.parse_file(spec_path)

        self.assertEqual(spec.title, "AURA Server Test Suite")
        self.assertEqual(len(spec.workstreams), 3)
        self.assertGreaterEqual(spec.parse_confidence, 0.8)

        graph = WorkstreamGraph(spec.workstreams)
        waves = graph.execution_waves()
        self.assertGreaterEqual(len(waves), 2)

    def test_self_bootstrap_spec_parses(self):
        """Self-bootstrap spec has correct structure and dependencies."""
        spec_path = Path("docs/superpowers/specs/2026-03-25-sadd-self-bootstrap-spec.md")
        if not spec_path.exists():
            self.skipTest("Self-bootstrap spec not yet created")
        parser = DesignSpecParser()
        spec = parser.parse_file(spec_path)
        self.assertEqual(len(spec.workstreams), 3)
        self.assertGreaterEqual(spec.parse_confidence, 0.8)
        # First workstream has no deps
        self.assertEqual(spec.workstreams[0].depends_on, [])
        # Others have deps
        self.assertGreater(len(spec.workstreams[1].depends_on), 0)
        self.assertGreater(len(spec.workstreams[2].depends_on), 0)

    def test_coordinator_with_mock_orchestrator(self):
        """Full coordinator run with mocked orchestrator (no real LLM calls)."""
        spec_path = Path("tests/fixtures/sadd_sample_spec.md")
        parser = DesignSpecParser()
        spec = parser.parse_file(spec_path)

        with tempfile.TemporaryDirectory() as td:
            brain = Brain(os.path.join(td, "test.db"))
            store = SessionStore(Path(td) / "sessions.db")

            def mock_factory():
                orch = MagicMock()
                orch.run_loop.return_value = {
                    "goal": "test",
                    "stop_reason": "PASS",
                    "history": [{"phase_outputs": {"verification": {"status": "pass"}}}],
                }
                return orch

            config = SessionConfig(max_parallel=2, max_cycles_per_workstream=1, dry_run=False)
            coordinator = SessionCoordinator(
                design_spec=spec,
                orchestrator_factory=mock_factory,
                brain=brain,
                config=config,
                session_store=store,
            )
            report = coordinator.run()

            self.assertEqual(report.total_workstreams, 3)
            self.assertEqual(report.completed, 3)
            self.assertEqual(report.failed, 0)

            # Verify session was persisted
            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["status"], "completed")

    def test_coordinator_report_summary(self):
        """Report summary is human-readable."""
        spec_path = Path("tests/fixtures/sadd_minimal_spec.md")
        parser = DesignSpecParser()
        spec = parser.parse_file(spec_path)

        with tempfile.TemporaryDirectory() as td:
            brain = Brain(os.path.join(td, "test.db"))

            def mock_factory():
                orch = MagicMock()
                orch.run_loop.return_value = {
                    "goal": "test",
                    "stop_reason": "PASS",
                    "history": [{"phase_outputs": {"verification": {"status": "pass"}}}],
                }
                return orch

            config = SessionConfig(max_parallel=1, max_cycles_per_workstream=1)
            coordinator = SessionCoordinator(
                design_spec=spec,
                orchestrator_factory=mock_factory,
                brain=brain,
                config=config,
            )
            report = coordinator.run()
            summary = report.summary()
            self.assertIn("SADD Session", summary)
            self.assertIn("1 completed", summary)
