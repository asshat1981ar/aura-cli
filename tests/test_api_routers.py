"""Tests for aura_cli/api/routers/runs.py and aura_cli/api/routers/ws.py."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Path / env bootstrap
# ---------------------------------------------------------------------------
os.environ.setdefault("AURA_SKIP_CHDIR", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine .* was never awaited")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Build a minimal test app that mounts only the two routers under test.
# This avoids loading the full server.py with all its heavy dependencies.
# ---------------------------------------------------------------------------

from aura_cli.api.routers import runs as runs_module  # noqa: E402
from aura_cli.api.routers import ws as ws_module  # noqa: E402

_test_app = FastAPI()
_test_app.include_router(runs_module.router, prefix="/api")
_test_app.include_router(ws_module.router)


def _closing_create_task(coro, *args, **kwargs):
    """Drop-in for asyncio.create_task that closes the coroutine immediately."""
    coro.close()
    return MagicMock()


@pytest.fixture()
def client(monkeypatch):
    """TestClient with no AGENT_API_TOKEN (auth disabled)."""
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.delenv("AGENT_API_ENABLE_RUN", raising=False)
    with TestClient(_test_app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def auth_client(monkeypatch):
    """TestClient with AGENT_API_TOKEN + run enabled."""
    monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    with TestClient(_test_app, raise_server_exceptions=False) as c:
        yield c


_AUTH_HEADERS = {"Authorization": "Bearer test-token"}


# ===========================================================================
# POST /api/run
# ===========================================================================


class TestRunEndpoint:
    def test_run_disabled_by_default(self, client):
        """403 when AGENT_API_ENABLE_RUN is not set."""
        resp = client.post("/api/run", json={"goal": "do something"})
        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"].lower()

    def test_run_disabled_without_env_flag_with_auth(self, monkeypatch):
        """403 even with valid auth when flag missing."""
        monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
        monkeypatch.delenv("AGENT_API_ENABLE_RUN", raising=False)
        with TestClient(_test_app, raise_server_exceptions=False) as c:
            resp = c.post("/api/run", json={"goal": "do something"}, headers=_AUTH_HEADERS)
        assert resp.status_code == 403

    def test_run_requires_auth_when_token_set(self, monkeypatch):
        """401 when AGENT_API_TOKEN is set but no Authorization header provided."""
        monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
        monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
        with TestClient(_test_app, raise_server_exceptions=False) as c:
            resp = c.post("/api/run", json={"goal": "do something"})
        assert resp.status_code == 401

    def test_run_rejects_wrong_token(self, monkeypatch):
        """403 when token is wrong."""
        monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
        monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
        with TestClient(_test_app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/api/run",
                json={"goal": "do something"},
                headers={"Authorization": "Bearer wrong-token"},
            )
        assert resp.status_code == 403

    def test_run_accepted_with_valid_auth(self, auth_client):
        """202-style accepted response with run_id returned."""
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            resp = auth_client.post("/api/run", json={"goal": "fix a bug"}, headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "accepted"
        assert "run_id" in body
        assert len(body["run_id"]) == 24  # secrets.token_hex(12) → 24 hex chars

    def test_run_accepted_no_auth_when_token_absent(self, client):
        """When AGENT_API_TOKEN is unset, run still requires the enable flag."""
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            # Enable run but no token set → auth passes, returns 200
            os.environ["AGENT_API_ENABLE_RUN"] = "1"
            try:
                resp = client.post("/api/run", json={"goal": "my goal"})
            finally:
                os.environ.pop("AGENT_API_ENABLE_RUN", None)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_run_default_max_cycles(self, auth_client):
        """Default max_cycles=1 and dry_run=False are accepted."""
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            resp = auth_client.post("/api/run", json={"goal": "test"}, headers=_AUTH_HEADERS)
        assert resp.status_code == 200

    def test_run_with_dry_run_flag(self, auth_client):
        """dry_run=True is accepted."""
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            resp = auth_client.post(
                "/api/run",
                json={"goal": "test", "max_cycles": 3, "dry_run": True},
                headers=_AUTH_HEADERS,
            )
        assert resp.status_code == 200
        assert "run_id" in resp.json()

    def test_run_missing_goal_field(self, auth_client):
        """422 when goal is absent."""
        resp = auth_client.post("/api/run", json={}, headers=_AUTH_HEADERS)
        assert resp.status_code == 422


# ===========================================================================
# POST /api/webhook/goal
# ===========================================================================


class TestWebhookGoal:
    def _post(self, client, payload=None, headers=None):
        if payload is None:
            payload = {"goal": "process data"}
        return client.post("/api/webhook/goal", json=payload, headers=headers or {})

    def test_webhook_goal_queued_no_auth(self, client):
        """When no AGENT_API_TOKEN, goal is accepted and queued."""
        runs_module._webhook_goal_queue.clear()
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            resp = self._post(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert "goal_id" in body

    def test_webhook_goal_queued_with_auth(self, auth_client):
        """With valid auth, goal is accepted."""
        runs_module._webhook_goal_queue.clear()
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            resp = self._post(auth_client, headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_webhook_goal_requires_auth_when_token_set(self, monkeypatch):
        """401 when token required but header absent."""
        monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
        with TestClient(_test_app, raise_server_exceptions=False) as c:
            resp = c.post("/api/webhook/goal", json={"goal": "test"})
        assert resp.status_code == 401

    def test_webhook_goal_enqueues_entry(self, client):
        """Goal entry is stored in the in-memory queue."""
        runs_module._webhook_goal_queue.clear()
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            resp = self._post(client, {"goal": "my test goal"})
        goal_id = resp.json()["goal_id"]
        assert goal_id in runs_module._webhook_goal_queue
        entry = runs_module._webhook_goal_queue[goal_id]
        assert entry["goal"] == "my test goal"
        assert entry["status"] == "queued"

    def test_webhook_goal_with_metadata(self, client):
        """Metadata is stored in queue entry."""
        runs_module._webhook_goal_queue.clear()
        payload = {
            "goal": "complex task",
            "priority": 3,
            "metadata": {"complexity": "high", "pipeline_run_id": "run-abc"},
        }
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            resp = self._post(client, payload)
        goal_id = resp.json()["goal_id"]
        assert runs_module._webhook_goal_queue[goal_id]["metadata"]["complexity"] == "high"

    def test_webhook_goal_missing_goal(self, client):
        """422 when goal field absent."""
        resp = client.post("/api/webhook/goal", json={"priority": 1})
        assert resp.status_code == 422

    def test_webhook_goal_returns_unique_ids(self, client):
        """Each submission gets a distinct goal_id."""
        runs_module._webhook_goal_queue.clear()
        ids = set()
        with patch("asyncio.create_task", side_effect=_closing_create_task):
            for _ in range(5):
                resp = self._post(client)
                ids.add(resp.json()["goal_id"])
        assert len(ids) == 5


# ===========================================================================
# GET /api/webhook/status/{goal_id}
# ===========================================================================


class TestWebhookStatus:
    def test_status_not_found(self, client):
        """404 for unknown goal_id."""
        resp = client.get("/api/webhook/status/nonexistent-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_status_queued(self, client):
        """Returns queued status for a freshly submitted goal."""
        runs_module._webhook_goal_queue.clear()
        import time

        goal_id = "test-goal-queued"
        runs_module._webhook_goal_queue[goal_id] = {
            "goal_id": goal_id,
            "goal": "test goal",
            "status": "queued",
            "priority": 5,
            "dry_run": False,
            "metadata": {},
            "queued_at": time.time(),
        }
        resp = client.get(f"/api/webhook/status/{goal_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert body["goal_id"] == goal_id
        assert body["goal"] == "test goal"

    def test_status_running(self, client):
        """Returns running status with started_at field."""
        import time

        goal_id = "test-goal-running"
        runs_module._webhook_goal_queue[goal_id] = {
            "goal_id": goal_id,
            "goal": "running goal",
            "status": "running",
            "priority": 5,
            "dry_run": False,
            "metadata": {},
            "queued_at": time.time(),
            "started_at": time.time(),
        }
        resp = client.get(f"/api/webhook/status/{goal_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert "started_at" in body

    def test_status_done_includes_result(self, client):
        """Returns result when goal is done."""
        import time

        goal_id = "test-goal-done"
        runs_module._webhook_goal_queue[goal_id] = {
            "goal_id": goal_id,
            "goal": "done goal",
            "status": "done",
            "priority": 5,
            "dry_run": False,
            "metadata": {},
            "queued_at": time.time(),
            "result": {"output": "success"},
            "completed_at": time.time(),
        }
        resp = client.get(f"/api/webhook/status/{goal_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["result"] == {"output": "success"}
        assert "completed_at" in body

    def test_status_failed_includes_error(self, client):
        """Returns error when goal has failed."""
        import time

        goal_id = "test-goal-failed"
        runs_module._webhook_goal_queue[goal_id] = {
            "goal_id": goal_id,
            "goal": "failed goal",
            "status": "failed",
            "priority": 5,
            "dry_run": False,
            "metadata": {},
            "queued_at": time.time(),
            "error": "something went wrong",
            "completed_at": time.time(),
        }
        resp = client.get(f"/api/webhook/status/{goal_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] == "something went wrong"


# ===========================================================================
# POST /api/webhook/plan-review
# ===========================================================================


class TestWebhookPlanReview:
    def _post(self, client, payload=None):
        if payload is None:
            payload = {
                "task_bundle": {"plan": ["step 1", "step 2"], "file_targets": ["a.py"]},
                "goal": "improve code",
                "pipeline_run_id": "run-123",
            }
        return client.post("/api/webhook/plan-review", json=payload)

    def test_plan_review_ok(self, client):
        resp = self._post(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "review_payload" in body

    def test_plan_review_formats_list_steps(self, client):
        payload = {
            "task_bundle": {"plan": ["write tests", "run lint", "merge PR"]},
            "goal": "quality gate",
        }
        resp = self._post(client, payload)
        review = resp.json()["review_payload"]
        assert "1. write tests" in review["plan_text"]
        assert "2. run lint" in review["plan_text"]
        assert "3. merge PR" in review["plan_text"]

    def test_plan_review_string_plan(self, client):
        """plan field as a plain string is passed through unchanged."""
        payload = {
            "task_bundle": {"plan": "just do everything"},
            "goal": "something",
        }
        resp = self._post(client, payload)
        review = resp.json()["review_payload"]
        assert review["plan_text"] == "just do everything"

    def test_plan_review_includes_goal_and_run_id(self, client):
        resp = self._post(client)
        review = resp.json()["review_payload"]
        assert review["goal"] == "improve code"
        assert review["pipeline_run_id"] == "run-123"

    def test_plan_review_includes_file_targets(self, client):
        resp = self._post(client)
        review = resp.json()["review_payload"]
        assert review["file_targets"] == ["a.py"]

    def test_plan_review_task_bundle_keys_listed(self, client):
        resp = self._post(client)
        review = resp.json()["review_payload"]
        assert "plan" in review["task_bundle_keys"]

    def test_plan_review_missing_goal(self, client):
        resp = client.post(
            "/api/webhook/plan-review",
            json={"task_bundle": {"plan": []}},
        )
        assert resp.status_code == 422

    def test_plan_review_requires_auth_when_token_set(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
        with TestClient(_test_app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/api/webhook/plan-review",
                json={"task_bundle": {}, "goal": "test"},
            )
        assert resp.status_code == 401


# ===========================================================================
# WebSocket /ws
# ===========================================================================


class TestWebSocketEndpoint:
    def test_ws_connect_receives_initial_message(self, client):
        """On connect, server sends initial message with type='initial'."""
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "initial"
            assert "payload" in msg
            assert "goals" in msg["payload"]
            assert "agents" in msg["payload"]

    def test_ws_initial_goals_is_list(self, client):
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert isinstance(msg["payload"]["goals"], list)

    def test_ws_initial_agents_is_list(self, client):
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert isinstance(msg["payload"]["agents"], list)

    def test_ws_ping_pong(self, client):
        """Send ping, receive pong."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # consume initial message
            ws.send_text(json.dumps({"type": "ping"}))
            pong = ws.receive_json()
            assert pong["type"] == "pong"

    def test_ws_unknown_message_type_does_not_crash(self, client):
        """Non-ping messages are silently ignored."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # consume initial message
            ws.send_text(json.dumps({"type": "unknown", "data": "x"}))
            # Send ping afterward to verify server is still alive
            ws.send_text(json.dumps({"type": "ping"}))
            pong = ws.receive_json()
            assert pong["type"] == "pong"

    def test_ws_multiple_pings(self, client):
        """Multiple pings each receive a pong."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial
            for _ in range(3):
                ws.send_text(json.dumps({"type": "ping"}))
                pong = ws.receive_json()
                assert pong["type"] == "pong"

    def test_ws_connection_manager_tracks_connection(self, client):
        """ConnectionManager adds and removes connection."""
        initial_count = len(ws_module.manager.active_connections)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            connected_count = len(ws_module.manager.active_connections)
            assert connected_count == initial_count + 1
        # After disconnect, connection should be removed
        assert len(ws_module.manager.active_connections) == initial_count


# ===========================================================================
# WebSocket /ws/logs
# ===========================================================================


class TestWebSocketLogs:
    def test_ws_logs_connect(self, client):
        """Can connect to /ws/logs without error."""
        with client.websocket_connect("/ws/logs") as ws:
            # No initial message on logs endpoint, just send a ping
            ws.send_text(json.dumps({"type": "ping"}))
            pong = ws.receive_json()
            assert pong["type"] == "pong"

    def test_ws_logs_ping_pong(self, client):
        """Ping/pong works on logs endpoint."""
        with client.websocket_connect("/ws/logs") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_logs_multiple_pings(self, client):
        with client.websocket_connect("/ws/logs") as ws:
            for _ in range(2):
                ws.send_text(json.dumps({"type": "ping"}))
                resp = ws.receive_json()
                assert resp["type"] == "pong"


# ===========================================================================
# ConnectionManager unit tests
# ===========================================================================


class TestConnectionManager:
    def test_broadcast_skips_on_empty(self):
        """Broadcast with no connections does not raise."""
        import asyncio

        mgr = ws_module.ConnectionManager()
        asyncio.run(mgr.broadcast({"type": "test"}))

    def test_disconnect_unknown_websocket_is_safe(self):
        """Disconnecting a websocket not in the list is a no-op."""
        mgr = ws_module.ConnectionManager()
        fake_ws = MagicMock()
        mgr.disconnect(fake_ws)  # Should not raise
