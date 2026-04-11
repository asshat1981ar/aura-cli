"""Tests for core/transport.py — JSON-RPC 2.0 stdio transport."""

import io
import json
import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.transport import (
    StdioTransport,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    SERVER_ERROR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(method, params=None, request_id=1):
    """Build a minimal valid JSON-RPC 2.0 request dict."""
    req = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        req["params"] = params
    return req


def _make_orchestrator(current_goal=None, queue_size=0):
    """Return a MagicMock that mimics LoopOrchestrator's public surface."""
    orch = MagicMock()
    orch.current_goal = current_goal
    orch.last_cycle_summary = None
    goal_queue = MagicMock()
    goal_queue.__len__ = MagicMock(return_value=queue_size)
    orch.goal_queue = goal_queue
    return orch


def _transport(orchestrator=None):
    return StdioTransport(orchestrator=orchestrator)


# ---------------------------------------------------------------------------
# system/health
# ---------------------------------------------------------------------------


class TestSystemHealth:
    def test_health_returns_ok_status(self):
        t = _transport()
        result = t.dispatch(_make_request("system/health"))
        assert result["result"]["status"] == "ok"

    def test_health_returns_timestamp(self):
        t = _transport()
        result = t.dispatch(_make_request("system/health"))
        ts = result["result"]["timestamp"]
        assert isinstance(ts, str) and ts.endswith("Z")

    def test_health_timestamp_is_parseable_iso8601(self):
        t = _transport()
        result = t.dispatch(_make_request("system/health"))
        ts = result["result"]["timestamp"].rstrip("Z")
        # Should not raise
        datetime.datetime.fromisoformat(ts)

    def test_health_no_orchestrator_needed(self):
        t = _transport(orchestrator=None)
        result = t.dispatch(_make_request("system/health"))
        assert result["result"]["status"] == "ok"

    def test_health_response_has_jsonrpc_field(self):
        t = _transport()
        result = t.dispatch(_make_request("system/health"))
        assert result["jsonrpc"] == "2.0"

    def test_health_response_id_matches_request(self):
        t = _transport()
        result = t.dispatch(_make_request("system/health", request_id=42))
        assert result["id"] == 42


# ---------------------------------------------------------------------------
# goal/status
# ---------------------------------------------------------------------------


class TestGoalStatus:
    def test_status_no_orchestrator(self):
        t = _transport(orchestrator=None)
        result = t.dispatch(_make_request("goal/status"))
        r = result["result"]
        assert r["current_goal"] is None
        assert r["queue_size"] == 0
        assert r["cycle_count"] == 0

    def test_status_with_current_goal(self):
        orch = _make_orchestrator(current_goal="Refactor auth")
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/status"))
        assert result["result"]["current_goal"] == "Refactor auth"

    def test_status_queue_size(self):
        orch = _make_orchestrator(queue_size=3)
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/status"))
        assert result["result"]["queue_size"] == 3

    def test_status_cycle_count_from_last_summary(self):
        orch = _make_orchestrator()
        orch.last_cycle_summary = {"cycle_index": 7}
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/status"))
        assert result["result"]["cycle_count"] == 7

    def test_status_cycle_count_zero_when_no_summary(self):
        orch = _make_orchestrator()
        orch.last_cycle_summary = None
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/status"))
        assert result["result"]["cycle_count"] == 0


# ---------------------------------------------------------------------------
# goal/add
# ---------------------------------------------------------------------------


class TestGoalAdd:
    def test_add_queues_goal(self):
        orch = _make_orchestrator()
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/add", {"goal": "Add tests"}))
        orch.goal_queue.add.assert_called_once_with("Add tests")
        assert result["result"]["queued"] is True

    def test_add_missing_goal_param(self):
        orch = _make_orchestrator()
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/add", {}))
        assert "error" in result
        assert result["error"]["code"] == SERVER_ERROR

    def test_add_empty_goal_string(self):
        orch = _make_orchestrator()
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/add", {"goal": ""}))
        assert "error" in result
        assert result["error"]["code"] == SERVER_ERROR

    def test_add_no_orchestrator(self):
        t = _transport(orchestrator=None)
        result = t.dispatch(_make_request("goal/add", {"goal": "A goal"}))
        assert "error" in result
        assert result["error"]["code"] == SERVER_ERROR

    def test_add_no_goal_queue(self):
        orch = _make_orchestrator()
        orch.goal_queue = None
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/add", {"goal": "A goal"}))
        assert "error" in result
        assert result["error"]["code"] == SERVER_ERROR


# ---------------------------------------------------------------------------
# goal/run
# ---------------------------------------------------------------------------


class TestGoalRun:
    def test_run_calls_orchestrator(self):
        orch = _make_orchestrator()
        orch.run_loop.return_value = {"goal": "Refactor", "stop_reason": "PASS", "history": []}
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/run", {"goal": "Refactor"}))
        orch.run_loop.assert_called_once_with("Refactor", max_cycles=5, dry_run=False)
        assert result["result"]["stop_reason"] == "PASS"

    def test_run_passes_max_cycles(self):
        orch = _make_orchestrator()
        orch.run_loop.return_value = {"goal": "x", "stop_reason": "MAX_CYCLES", "history": []}
        t = _transport(orchestrator=orch)
        t.dispatch(_make_request("goal/run", {"goal": "x", "max_cycles": 2}))
        orch.run_loop.assert_called_once_with("x", max_cycles=2, dry_run=False)

    def test_run_passes_dry_run(self):
        orch = _make_orchestrator()
        orch.run_loop.return_value = {"goal": "x", "stop_reason": "PASS", "history": []}
        t = _transport(orchestrator=orch)
        t.dispatch(_make_request("goal/run", {"goal": "x", "dry_run": True}))
        orch.run_loop.assert_called_once_with("x", max_cycles=5, dry_run=True)

    def test_run_missing_goal_param(self):
        orch = _make_orchestrator()
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/run", {}))
        assert "error" in result
        assert result["error"]["code"] == SERVER_ERROR

    def test_run_no_orchestrator(self):
        t = _transport(orchestrator=None)
        result = t.dispatch(_make_request("goal/run", {"goal": "Test"}))
        assert "error" in result
        assert result["error"]["code"] == SERVER_ERROR

    def test_run_orchestrator_raises(self):
        orch = _make_orchestrator()
        orch.run_loop.side_effect = RuntimeError("boom")
        t = _transport(orchestrator=orch)
        result = t.dispatch(_make_request("goal/run", {"goal": "Test"}))
        assert "error" in result
        assert result["error"]["code"] == SERVER_ERROR
        assert "boom" in result["error"]["message"]


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_unknown_method_returns_method_not_found(self):
        t = _transport()
        result = t.dispatch(_make_request("nonexistent/method"))
        assert result["error"]["code"] == METHOD_NOT_FOUND

    def test_unknown_method_message_contains_method_name(self):
        t = _transport()
        result = t.dispatch(_make_request("nonexistent/method"))
        assert "nonexistent/method" in result["error"]["message"]

    def test_missing_method_field_returns_invalid_request(self):
        t = _transport()
        result = t.dispatch({"jsonrpc": "2.0", "id": 1})
        assert result["error"]["code"] == INVALID_REQUEST

    def test_wrong_jsonrpc_version_returns_invalid_request(self):
        t = _transport()
        result = t.dispatch({"jsonrpc": "1.0", "id": 1, "method": "system/health"})
        assert result["error"]["code"] == INVALID_REQUEST

    def test_non_dict_request_returns_invalid_request(self):
        t = _transport()
        result = t._process_line(json.dumps([1, 2, 3]))
        assert result["error"]["code"] == INVALID_REQUEST

    def test_invalid_json_returns_parse_error(self):
        t = _transport()
        result = t._process_line("{not valid json}")
        assert result["error"]["code"] == PARSE_ERROR

    def test_parse_error_id_is_null(self):
        t = _transport()
        result = t._process_line("broken")
        assert result["id"] is None

    def test_notification_no_id_returns_none(self):
        t = _transport()
        notification = {"jsonrpc": "2.0", "method": "system/health"}
        result = t.dispatch(notification)
        assert result is None

    def test_notification_unknown_method_returns_none(self):
        t = _transport()
        notification = {"jsonrpc": "2.0", "method": "unknown/method"}
        result = t.dispatch(notification)
        assert result is None

    def test_error_response_id_echoes_request_id(self):
        t = _transport()
        result = t.dispatch({"jsonrpc": "2.0", "id": 99, "method": "bad/method"})
        assert result["id"] == 99


# ---------------------------------------------------------------------------
# run() — full stdin/stdout loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    def _run_with_lines(self, lines, orchestrator=None):
        """Feed *lines* (list of str) through transport.run() and return stdout lines."""
        stdin = io.StringIO("\n".join(lines) + "\n")
        stdout = io.StringIO()
        t = StdioTransport(orchestrator=orchestrator, stdin=stdin, stdout=stdout)
        t.run()
        stdout.seek(0)
        return [json.loads(l) for l in stdout.read().splitlines() if l.strip()]

    def test_run_processes_single_health_request(self):
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "system/health"})
        responses = self._run_with_lines([req])
        assert len(responses) == 1
        assert responses[0]["result"]["status"] == "ok"

    def test_run_processes_multiple_requests(self):
        r1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "system/health"})
        r2 = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "goal/status"})
        responses = self._run_with_lines([r1, r2])
        assert len(responses) == 2

    def test_run_skips_blank_lines(self):
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "system/health"})
        responses = self._run_with_lines(["", req, ""])
        assert len(responses) == 1

    def test_run_notification_produces_no_output(self):
        notification = json.dumps({"jsonrpc": "2.0", "method": "system/health"})
        responses = self._run_with_lines([notification])
        assert responses == []
