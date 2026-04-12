"""Integration tests for the n8n pipeline optimization layer.

Covers:
- WebhookGoalRequest metadata field (aura_cli/server.py)
- _enrich_act_context() quality-gate critique injection (core/orchestrator.py)
- _notify_n8n_feedback() P4 feedback dispatch (core/orchestrator.py)
- ReflectorAgent skill_context enrichment (agents/reflector.py)
- P5 observability fan-out (observability_webhook in config)

Run with:
    AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_pipeline_optimization.py -v
"""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

os.environ.setdefault("AURA_SKIP_CHDIR", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_orchestrator(**kwargs):
    """Return a minimal LoopOrchestrator instance with mocked heavy deps."""
    from core.orchestrator import LoopOrchestrator

    orch = LoopOrchestrator.__new__(LoopOrchestrator)
    # Minimal attribute set needed for tested methods
    orch.project_root = MagicMock()
    orch.project_root.__str__ = lambda _: "/tmp/test_project"
    orch.memory_controller = MagicMock()
    orch.brain = None
    orch.goal_queue = None
    orch.quality_trends = MagicMock()
    orch.quality_trends.record_from_cycle.return_value = []
    orch.adaptive_pipeline = None
    orch.context_graph = None
    orch._improvement_loops = []
    orch.propagation_engine = None
    orch.confidence_router = MagicMock()
    orch.active_cycle_summary = None
    orch.last_cycle_summary = None
    orch.current_goal = None
    orch._consecutive_fails = 0
    for k, v in kwargs.items():
        setattr(orch, k, v)
    return orch


class _CaptureServer:
    """Tiny HTTP server that captures POST bodies for test assertions."""

    def __init__(self):
        self.received: list = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, port: int = 0):
        captured = self.received

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    captured.append(json.loads(body))
                except Exception:
                    captured.append({"_raw": body.decode()})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *args):
                pass  # suppress output

        self._server = HTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_address[1]

    def stop(self):
        if self._server:
            self._server.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# WebhookGoalRequest
# ─────────────────────────────────────────────────────────────────────────────


class TestWebhookGoalRequest(unittest.TestCase):
    def test_metadata_field_accepted(self):
        from aura_cli.server import WebhookGoalRequest

        req = WebhookGoalRequest(
            goal="fix tests",
            metadata={"quality_gate_critique": "coupling too high", "pipeline_run_id": "run-001"},
        )
        self.assertEqual(req.metadata["quality_gate_critique"], "coupling too high")
        self.assertEqual(req.metadata["pipeline_run_id"], "run-001")

    def test_metadata_defaults_to_empty_dict(self):
        from aura_cli.server import WebhookGoalRequest

        req = WebhookGoalRequest(goal="fix tests")
        self.assertIsInstance(req.metadata, dict)
        self.assertEqual(req.metadata, {})


# ─────────────────────────────────────────────────────────────────────────────
# _enrich_act_context
# ─────────────────────────────────────────────────────────────────────────────


class TestEnrichActContext(unittest.TestCase):
    def _make_config(self, enabled: bool) -> dict:
        return {
            "n8n_connector": {
                "enabled": True,
                "quality_gate_enabled": enabled,
            }
        }

    def test_returns_unchanged_when_gate_disabled(self):
        orch = _make_orchestrator()
        bundle = {"critique": "original", "task": "do work"}
        with patch.object(type(orch), "_load_config_file", return_value=self._make_config(False)):
            result = orch._enrich_act_context(bundle)
        self.assertEqual(result, bundle)

    def test_prepends_critique_when_gate_enabled(self):
        orch = _make_orchestrator()
        orch._cycle_context = {"quality_gate_critique": "High coupling detected"}
        bundle = {"critique": "pre-existing", "task": "do work"}
        with patch.object(type(orch), "_load_config_file", return_value=self._make_config(True)):
            result = orch._enrich_act_context(bundle)
        self.assertIn("[Dev Suite Quality Gate Review]", result["critique"])
        self.assertIn("High coupling detected", result["critique"])
        self.assertIn("pre-existing", result["critique"])

    def test_no_critique_available_returns_unchanged(self):
        orch = _make_orchestrator()
        orch._cycle_context = {}
        bundle = {"task": "do work"}
        with patch.object(type(orch), "_load_config_file", return_value=self._make_config(True)):
            result = orch._enrich_act_context(bundle)
        self.assertEqual(result, bundle)

    def test_enrich_does_not_raise_on_exception(self):
        orch = _make_orchestrator()
        with patch.object(type(orch), "_load_config_file", side_effect=RuntimeError("boom")):
            result = orch._enrich_act_context({"task": "safe"})
        self.assertEqual(result["task"], "safe")


# ─────────────────────────────────────────────────────────────────────────────
# ReflectorAgent skill_context enrichment
# ─────────────────────────────────────────────────────────────────────────────


class TestReflectorSkillContext(unittest.TestCase):
    def setUp(self):
        from agents.reflector import ReflectorAgent

        self.agent = ReflectorAgent()

    def test_no_skill_context_produces_empty_summary(self):
        out = self.agent.run({"verification": {"status": "pass", "failures": []}})
        self.assertEqual(out["learnings"], [])
        self.assertEqual(out["skill_summary"], {})

    def test_security_critical_surfaces_as_learning(self):
        out = self.agent.run(
            {
                "verification": {"status": "pass", "failures": []},
                "skill_context": {"security_scanner": {"critical_count": 3, "findings": ["a", "b", "c"]}},
            }
        )
        self.assertTrue(any("security" in l for l in out["learnings"]))
        self.assertEqual(out["skill_summary"]["security_scanner"]["critical"], 3)

    def test_coupling_alert(self):
        out = self.agent.run(
            {
                "verification": {"status": "pass", "failures": []},
                "skill_context": {"architecture_validator": {"coupling_score": 1.8, "circular_deps": []}},
            }
        )
        self.assertTrue(any("coupling" in l for l in out["learnings"]))

    def test_coverage_below_target_alert(self):
        out = self.agent.run(
            {
                "verification": {"status": "pass", "failures": []},
                "skill_context": {"test_coverage_analyzer": {"coverage_pct": 55.0, "meets_target": False}},
            }
        )
        self.assertTrue(any("coverage" in l for l in out["learnings"]))

    def test_no_alert_when_all_ok(self):
        out = self.agent.run(
            {
                "verification": {"status": "pass", "failures": []},
                "skill_context": {
                    "security_scanner": {"critical_count": 0, "findings": []},
                    "architecture_validator": {"coupling_score": 0.7, "circular_deps": []},
                    "test_coverage_analyzer": {"coverage_pct": 85.0, "meets_target": True},
                },
            }
        )
        self.assertEqual(out["learnings"], [])

    def test_pipeline_run_id_propagated(self):
        out = self.agent.run(
            {
                "verification": {"status": "pass", "failures": []},
                "pipeline_run_id": "run-xyz",
            }
        )
        self.assertEqual(out["pipeline_run_id"], "run-xyz")

    def test_failure_analysis_still_works(self):
        out = self.agent.run(
            {
                "verification": {"status": "fail", "failures": ["NameError: foo not defined"]},
            }
        )
        self.assertTrue(any("context_gap" in l for l in out["learnings"]))


# ─────────────────────────────────────────────────────────────────────────────
# _notify_n8n_feedback + P5 fan-out
# ─────────────────────────────────────────────────────────────────────────────


class TestNotifyN8nFeedback(unittest.TestCase):
    def setUp(self):
        self.capture_p4 = _CaptureServer()
        self.capture_p5 = _CaptureServer()
        self.port_p4 = self.capture_p4.start()
        self.port_p5 = self.capture_p5.start()

    def tearDown(self):
        self.capture_p4.stop()
        self.capture_p5.stop()

    def _config(self, enabled=True, obs=True):
        cfg = {
            "n8n_connector": {
                "enabled": enabled,
                "feedback_loop_webhook": f"http://127.0.0.1:{self.port_p4}/feedback",
            }
        }
        if obs:
            cfg["n8n_connector"]["observability_webhook"] = f"http://127.0.0.1:{self.port_p5}/trace"
        return cfg

    def test_posts_to_feedback_webhook(self):
        orch = _make_orchestrator()
        phase_outputs = {
            "reflection": {"learnings": ["skill_alert: coupling"], "summary": "pass", "skill_summary": {}},
            "verification": {"status": "pass"},
        }
        with patch.object(type(orch), "_load_config_file", return_value=self._config()):
            orch._notify_n8n_feedback("refactor auth", "cycle_001", True, phase_outputs)

        import time

        time.sleep(0.2)
        self.assertEqual(len(self.capture_p4.received), 1)
        body = self.capture_p4.received[0]
        self.assertEqual(body["cycle_id"], "cycle_001")
        self.assertEqual(body["goal"], "refactor auth")
        self.assertTrue(body["passed"])
        self.assertIn("skill_alert: coupling", body["learnings"])

    def test_fan_out_to_observability(self):
        orch = _make_orchestrator()
        phase_outputs = {
            "reflection": {"learnings": [], "summary": "ok", "skill_summary": {}},
            "verification": {"status": "pass"},
        }
        with patch.object(type(orch), "_load_config_file", return_value=self._config(obs=True)):
            orch._notify_n8n_feedback("add tests", "cycle_002", True, phase_outputs)

        import time

        time.sleep(0.2)
        self.assertEqual(len(self.capture_p5.received), 1)
        self.assertEqual(self.capture_p5.received[0]["cycle_id"], "cycle_002")

    def test_skips_when_n8n_disabled(self):
        orch = _make_orchestrator()
        phase_outputs = {"reflection": {"learnings": [], "summary": "", "skill_summary": {}}}
        with patch.object(type(orch), "_load_config_file", return_value=self._config(enabled=False)):
            orch._notify_n8n_feedback("test", "cycle_003", False, phase_outputs)

        import time

        time.sleep(0.1)
        self.assertEqual(len(self.capture_p4.received), 0)

    def test_does_not_raise_on_connection_error(self):
        orch = _make_orchestrator()
        cfg = {"n8n_connector": {"enabled": True, "feedback_loop_webhook": "http://127.0.0.1:19999/gone"}}
        phase_outputs = {"reflection": {"learnings": [], "summary": "", "skill_summary": {}}}
        with patch.object(type(orch), "_load_config_file", return_value=cfg):
            # Must not raise
            orch._notify_n8n_feedback("test", "cycle_004", True, phase_outputs)

    def test_pipeline_run_id_falls_back_to_cycle_id(self):
        orch = _make_orchestrator()
        # No _cycle_context set
        phase_outputs = {
            "reflection": {"learnings": [], "summary": "", "skill_summary": {}},
            "verification": {"status": "skip"},
        }
        with patch.object(type(orch), "_load_config_file", return_value=self._config(obs=False)):
            orch._notify_n8n_feedback("goal", "cycle_fallback", True, phase_outputs)

        import time

        time.sleep(0.2)
        self.assertEqual(len(self.capture_p4.received), 1)
        self.assertEqual(self.capture_p4.received[0]["pipeline_run_id"], "cycle_fallback")


if __name__ == "__main__":
    unittest.main()
