"""Tests for core/sadd/n8n_pipeline_bridge.py — N8nPipelineBridge."""

import json
import unittest
from unittest.mock import MagicMock, patch

from core.sadd.n8n_pipeline_bridge import N8nPipelineBridge


def _make_bridge(**overrides):
    """Create an N8nPipelineBridge with test config."""
    cfg = {
        "n8n_connector": {
            "enabled": True,
            "base_url": "http://localhost:5678",
            "timeout_seconds": 5,
            "goal_route_webhook": "http://localhost:5678/webhook/goal-route",
            "quality_gate_webhook": "http://localhost:5678/webhook/quality-gate",
            "quality_gate_enabled": False,
            "pipeline_coordinator_webhook": "http://localhost:5678/webhook/pipeline",
            "route_complex_through_p3": False,
            "feedback_loop_webhook": "http://localhost:5678/webhook/feedback",
            "observability_webhook": "http://localhost:5678/webhook/trace",
            "trace_enabled": True,
            "session_manager_webhook": "http://localhost:5678/webhook/session",
            "workstream_monitor_webhook": "http://localhost:5678/webhook/workstream",
        }
    }
    cfg["n8n_connector"].update(overrides)
    return N8nPipelineBridge(cfg)


class TestInit(unittest.TestCase):

    def test_defaults_from_config(self):
        bridge = _make_bridge()
        self.assertTrue(bridge.enabled)
        self.assertEqual(bridge.base_url, "http://localhost:5678")
        self.assertEqual(bridge.timeout, 5)

    def test_disabled_by_default(self):
        bridge = N8nPipelineBridge({})
        self.assertFalse(bridge.enabled)

    def test_empty_n8n_connector(self):
        bridge = N8nPipelineBridge({"n8n_connector": {}})
        self.assertFalse(bridge.enabled)
        self.assertEqual(bridge.base_url, "http://localhost:5678")


class TestPost(unittest.TestCase):

    @patch("core.sadd.n8n_pipeline_bridge.urllib.request.urlopen")
    def test_successful_post(self, mock_urlopen):
        bridge = _make_bridge()
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = bridge._post("http://localhost:5678/webhook/test", {"key": "val"})
        self.assertEqual(result, {"ok": True})

    def test_post_disabled_returns_none(self):
        bridge = _make_bridge(enabled=False)
        result = bridge._post("http://localhost:5678/webhook/test", {})
        self.assertIsNone(result)

    def test_post_empty_url_returns_none(self):
        bridge = _make_bridge()
        result = bridge._post("", {"key": "val"})
        self.assertIsNone(result)

    @patch("core.sadd.n8n_pipeline_bridge.urllib.request.urlopen")
    def test_post_non_json_response(self, mock_urlopen):
        bridge = _make_bridge()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"plain text"
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = bridge._post("http://localhost:5678/webhook/test", {})
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["body"], "plain text")

    @patch("core.sadd.n8n_pipeline_bridge.urllib.request.urlopen")
    def test_post_exception_returns_none(self, mock_urlopen):
        bridge = _make_bridge()
        mock_urlopen.side_effect = Exception("connection refused")
        result = bridge._post("http://localhost:5678/webhook/test", {})
        self.assertIsNone(result)


class TestRouteGoal(unittest.TestCase):

    def test_successful_route(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value={"complexity": "complex", "lane": "dual", "route": "p3"})
        result = bridge.route_goal("ws-1", "Build new feature", ["python"])
        self.assertEqual(result["complexity"], "complex")
        self.assertEqual(result["lane"], "dual")

    def test_fallback_on_none(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value=None)
        result = bridge.route_goal("ws-1", "Simple task")
        self.assertEqual(result["complexity"], "simple")
        self.assertEqual(result["lane"], "fast")
        self.assertEqual(result["route"], "local")

    def test_fallback_on_missing_complexity(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value={"some": "data"})
        result = bridge.route_goal("ws-1", "task")
        self.assertEqual(result["complexity"], "simple")

    def test_route_goal_payload(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value=None)
        bridge.route_goal("ws-1", "goal text", ["tag1"])
        call_payload = bridge._post.call_args[0][1]
        self.assertEqual(call_payload["source"], "sadd")
        self.assertEqual(call_payload["ws_id"], "ws-1")
        self.assertEqual(call_payload["tags"], ["tag1"])


class TestQualityGate(unittest.TestCase):

    def test_disabled_returns_pass_skipped(self):
        bridge = _make_bridge(quality_gate_enabled=False)
        result = bridge.quality_gate("ws-1", {"steps": []}, "goal")
        self.assertEqual(result["verdict"], "pass")
        self.assertTrue(result["skipped"])

    def test_enabled_returns_verdict(self):
        bridge = _make_bridge(quality_gate_enabled=True)
        bridge._post = MagicMock(return_value={"verdict": "fail", "reason": "too risky"})
        result = bridge.quality_gate("ws-1", {"steps": []}, "goal")
        self.assertEqual(result["verdict"], "fail")

    def test_enabled_no_response_returns_pass(self):
        bridge = _make_bridge(quality_gate_enabled=True)
        bridge._post = MagicMock(return_value=None)
        result = bridge.quality_gate("ws-1", {}, "goal")
        self.assertEqual(result["verdict"], "pass")
        self.assertTrue(result["skipped"])


class TestCoordinatePipeline(unittest.TestCase):

    def test_disabled_returns_none(self):
        bridge = _make_bridge(route_complex_through_p3=False)
        result = bridge.coordinate_pipeline("ws-1", "goal", {})
        self.assertIsNone(result)

    def test_enabled_posts(self):
        bridge = _make_bridge(route_complex_through_p3=True)
        bridge._post = MagicMock(return_value={"pipeline": "started"})
        result = bridge.coordinate_pipeline("ws-1", "goal", {"ctx": True})
        self.assertEqual(result, {"pipeline": "started"})
        call_payload = bridge._post.call_args[0][1]
        self.assertEqual(call_payload["source"], "sadd")


class TestSendFeedback(unittest.TestCase):

    def test_sends_feedback(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value=None)
        phase_outputs = {
            "verification": {"status": "pass"},
            "reflection": {"learnings": ["learned something"]},
            "quality": {"score": 0.9},
        }
        bridge.send_feedback("ws-1", "cycle-1", True, phase_outputs)
        bridge._post.assert_called_once()
        payload = bridge._post.call_args[0][1]
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["verification_status"], "pass")
        self.assertEqual(payload["learnings"], ["learned something"])

    def test_send_feedback_empty_outputs(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value=None)
        bridge.send_feedback("ws-1", "cycle-1", False, {})
        payload = bridge._post.call_args[0][1]
        self.assertEqual(payload["verification_status"], "skip")
        self.assertEqual(payload["learnings"], [])


class TestTrace(unittest.TestCase):

    def test_trace_enabled(self):
        bridge = _make_bridge(trace_enabled=True)
        bridge._post = MagicMock(return_value=None)
        bridge.trace("cycle_start", {"ws_id": "ws-1"})
        bridge._post.assert_called_once()
        payload = bridge._post.call_args[0][1]
        self.assertEqual(payload["event_type"], "cycle_start")
        self.assertIn("timestamp", payload)

    def test_trace_disabled(self):
        bridge = _make_bridge(trace_enabled=False)
        bridge._post = MagicMock()
        bridge.trace("cycle_start", {"ws_id": "ws-1"})
        bridge._post.assert_not_called()


class TestNotifySession(unittest.TestCase):

    def test_sends_session_event(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value=None)
        bridge.notify_session("session_started", {"session_id": "s-1"})
        payload = bridge._post.call_args[0][1]
        self.assertEqual(payload["event_type"], "session_started")
        self.assertEqual(payload["source"], "sadd")
        self.assertEqual(payload["session_id"], "s-1")


class TestNotifyWorkstream(unittest.TestCase):

    def test_sends_workstream_event(self):
        bridge = _make_bridge()
        bridge._post = MagicMock(return_value=None)
        bridge.notify_workstream("ws_completed", {"ws_id": "ws-1", "cycles": 3})
        payload = bridge._post.call_args[0][1]
        self.assertEqual(payload["event_type"], "ws_completed")
        self.assertEqual(payload["ws_id"], "ws-1")
        self.assertEqual(payload["cycles"], 3)


if __name__ == "__main__":
    unittest.main()
