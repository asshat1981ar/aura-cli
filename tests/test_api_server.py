"""Comprehensive unit tests for aura_cli/api_server.py."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Pre-import: mock optional heavy dependencies so GOAL_QUEUE_AVAILABLE /
# REGISTRY_AVAILABLE / AUTH_AVAILABLE are set at import time (we override
# them per-test via monkeypatch anyway).
# ---------------------------------------------------------------------------
for _dep in ("core.goal_queue", "core.mcp_agent_registry", "core.auth"):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

import aura_cli.api_server as api_server  # noqa: E402
from aura_cli.api_server import (  # noqa: E402
    ConnectionManager,
    GoalCreate,
    GoalCycleRecord,
    GoalDetailResponse,
    GoalPrioritizeResponse,
    GoalResponse,
    _goal_id_for_archived,
    _goal_id_for_inflight,
    _goal_id_for_queued,
    app,
    broadcast_log,
    format_goals_for_api,
    get_agents_from_registry,
    get_goal_archive,
    get_goal_queue_data,
    get_telemetry_data,
    get_in_flight_goal,
    get_notifications_status,
    get_performance_stats,
    get_metrics_summary,
    get_goal_trends,
    get_system_performance_analytics,
    get_analytics_insights,
    get_daily_report,
    get_agents_performance,
    get_agent_performance,
    reset_agent_performance,
    list_pull_requests,
    get_pull_request,
    get_pr_reviews,
    get_pr_comments,
    github_callback,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
_EMPTY_QUEUE: dict = {"queue": [], "in_flight": {}}
_SAMPLE_QUEUE: dict = {"queue": ["write tests", "add logging"], "in_flight": {}}
_SAMPLE_ARCHIVE: list = [
    {
        "goal": "improve coverage",
        "status": "completed",
        "timestamp": "2024-01-01T00:00:00",
        "cycles": 3,
        "history": [
            {
                "phase": "act",
                "outcome": "ok",
                "duration_s": 1.0,
                "timestamp": "2024-01-01T00:00:01",
            }
        ],
    }
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Disable optional features and clear mutable state before every test."""
    monkeypatch.setattr(api_server, "AUTH_AVAILABLE", False)
    monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", False)
    monkeypatch.setattr(api_server, "REGISTRY_AVAILABLE", False)
    api_server.sadd_sessions_store.clear()
    yield
    api_server.sadd_sessions_store.clear()


# ===========================================================================
# Pydantic model tests
# ===========================================================================


class TestGoalCreate:
    def test_valid_defaults(self):
        g = GoalCreate(description="do something")
        assert g.description == "do something"
        assert g.priority == 1
        assert g.max_cycles == 10

    def test_custom_fields(self):
        g = GoalCreate(description="task", priority=7, max_cycles=5)
        assert g.priority == 7
        assert g.max_cycles == 5

    def test_stripped_description(self):
        g = GoalCreate(description="  trim me  ")
        assert g.stripped_description == "trim me"

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="blank"):
            GoalCreate(description="   ")

    def test_empty_string_raises_validation_error(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GoalCreate(description="")

    def test_priority_zero_raises(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GoalCreate(description="ok", priority=0)

    def test_priority_eleven_raises(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GoalCreate(description="ok", priority=11)

    def test_max_cycles_zero_raises(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GoalCreate(description="ok", max_cycles=0)

    def test_priority_boundary_valid(self):
        g1 = GoalCreate(description="x", priority=1)
        g2 = GoalCreate(description="x", priority=10)
        assert g1.priority == 1
        assert g2.priority == 10


class TestGoalCycleRecord:
    def test_minimal_fields(self):
        r = GoalCycleRecord(cycle=1)
        assert r.cycle == 1
        assert r.phase is None
        assert r.outcome is None

    def test_full_fields(self):
        r = GoalCycleRecord(cycle=2, phase="act", outcome="ok", duration_s=2.5, timestamp="2024-01-01")
        assert r.phase == "act"
        assert r.duration_s == 2.5

    def test_model_dump(self):
        r = GoalCycleRecord(cycle=3, phase="verify")
        d = r.model_dump()
        assert d["cycle"] == 3
        assert d["phase"] == "verify"


class TestGoalResponse:
    def _now(self):
        return datetime.utcnow().isoformat()

    def test_valid_defaults(self):
        n = self._now()
        r = GoalResponse(id="g1", description="d", status="pending", created_at=n, updated_at=n)
        assert r.priority == 1
        assert r.progress == 0
        assert r.cycles == 0
        assert r.max_cycles == 10

    def test_progress_bounds_negative(self):
        import pydantic

        n = self._now()
        with pytest.raises(pydantic.ValidationError):
            GoalResponse(id="x", description="x", status="pending", created_at=n, updated_at=n, progress=-1)


class TestGoalPrioritizeResponse:
    def test_defaults(self):
        r = GoalPrioritizeResponse(success=True, id="goal-q-0")
        assert r.position == 0
        assert r.message == ""

    def test_with_message_and_position(self):
        r = GoalPrioritizeResponse(success=False, id="g", position=2, message="err")
        assert r.message == "err"
        assert r.position == 2


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestGoalIdHelpers:
    def test_queued_id_format(self):
        assert _goal_id_for_queued(0, "any") == "goal-q-0"
        assert _goal_id_for_queued(3, "any") == "goal-q-3"

    def test_queued_id_ignores_description(self):
        assert _goal_id_for_queued(1, "x") == _goal_id_for_queued(1, "y")

    def test_inflight_id_prefix(self):
        assert _goal_id_for_inflight("my goal").startswith("goal-f-")

    def test_inflight_id_deterministic(self):
        assert _goal_id_for_inflight("x") == _goal_id_for_inflight("x")

    def test_inflight_id_unique_per_desc(self):
        assert _goal_id_for_inflight("a") != _goal_id_for_inflight("b")

    def test_archived_id_prefix(self):
        assert _goal_id_for_archived("done").startswith("goal-a-")

    def test_archived_id_deterministic(self):
        assert _goal_id_for_archived("x") == _goal_id_for_archived("x")


# ===========================================================================
# ConnectionManager tests
# ===========================================================================


class TestConnectionManager:
    def test_initial_empty(self):
        m = ConnectionManager()
        assert m.active_connections == []

    def test_connect_accepts_and_stores(self):
        m = ConnectionManager()
        ws = AsyncMock()
        asyncio.run(m.connect(ws))
        ws.accept.assert_called_once()
        assert ws in m.active_connections

    def test_disconnect_removes(self):
        m = ConnectionManager()
        ws = MagicMock()
        m.active_connections.append(ws)
        m.disconnect(ws)
        assert ws not in m.active_connections

    def test_disconnect_noop_on_unknown(self):
        m = ConnectionManager()
        ws = MagicMock()
        m.disconnect(ws)  # must not raise

    def test_broadcast_sends_json_to_all(self):
        m = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        m.active_connections.extend([ws1, ws2])
        asyncio.run(m.broadcast({"type": "ping"}))
        ws1.send_json.assert_called_once_with({"type": "ping"})
        ws2.send_json.assert_called_once_with({"type": "ping"})

    def test_broadcast_removes_failed_connection(self):
        m = ConnectionManager()
        good = AsyncMock()
        bad = AsyncMock()
        bad.send_json.side_effect = RuntimeError("closed")
        m.active_connections.extend([good, bad])
        asyncio.run(m.broadcast({"type": "update"}))
        assert bad not in m.active_connections
        assert good in m.active_connections

    def test_broadcast_empty_connections_no_error(self):
        m = ConnectionManager()
        asyncio.run(m.broadcast({"type": "test"}))  # no error


# ===========================================================================
# Data access function tests
# ===========================================================================


class TestGetGoalQueueData:
    def test_missing_file_returns_default(self):
        with patch.object(Path, "exists", return_value=False):
            result = get_goal_queue_data()
        assert result == {"queue": [], "in_flight": {}}

    def test_valid_dict_json(self):
        data = {"queue": ["t1"], "in_flight": {"t2": 1234567890.0}}
        with patch.object(Path, "exists", return_value=True), patch("builtins.open", mock_open()), patch("json.load", return_value=data):
            result = get_goal_queue_data()
        assert result == data

    def test_list_json_wraps_in_dict(self):
        lst = ["item1", "item2"]
        with patch.object(Path, "exists", return_value=True), patch("builtins.open", mock_open()), patch("json.load", return_value=lst):
            result = get_goal_queue_data()
        assert result["queue"] == lst
        assert result["in_flight"] == {}

    def test_json_decode_error_returns_default(self):
        with patch.object(Path, "exists", return_value=True), patch("builtins.open", mock_open()), patch("json.load", side_effect=json.JSONDecodeError("err", "", 0)):
            result = get_goal_queue_data()
        assert result == {"queue": [], "in_flight": {}}

    def test_io_error_returns_default(self):
        with patch.object(Path, "exists", return_value=True), patch("builtins.open", side_effect=IOError("no")):
            result = get_goal_queue_data()
        assert result == {"queue": [], "in_flight": {}}


class TestGetGoalArchive:
    def test_missing_file_returns_empty(self):
        with patch.object(Path, "exists", return_value=False):
            result = get_goal_archive()
        assert result == []

    def test_valid_jsonl_lines(self):
        lines = '{"goal": "g1"}\n{"goal": "g2"}\n'
        with patch.object(Path, "exists", return_value=True), patch("builtins.open", mock_open(read_data=lines)):
            result = get_goal_archive()
        assert len(result) == 2
        assert result[0]["goal"] == "g1"

    def test_blank_lines_skipped(self):
        lines = '{"goal": "g1"}\n\n{"goal": "g2"}\n'
        with patch.object(Path, "exists", return_value=True), patch("builtins.open", mock_open(read_data=lines)):
            result = get_goal_archive()
        assert len(result) == 2

    def test_io_error_returns_empty(self):
        with patch.object(Path, "exists", return_value=True), patch("builtins.open", side_effect=IOError):
            result = get_goal_archive()
        assert result == []


class TestGetTelemetryData:
    def test_missing_db_returns_empty(self):
        with patch.object(Path, "exists", return_value=False):
            result = get_telemetry_data()
        assert result == []

    def test_valid_db_returns_records(self):
        row = ("2024-01-01T00:00:00", "planner", 1.5, 100)
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [row]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        with patch.object(Path, "exists", return_value=True), patch("sqlite3.connect", return_value=mock_conn):
            result = get_telemetry_data()
        assert len(result) == 1
        assert result[0]["agent_name"] == "planner"
        assert result[0]["latency"] == 1.5

    def test_sqlite_error_returns_empty(self):
        with patch.object(Path, "exists", return_value=True), patch("sqlite3.connect", side_effect=sqlite3.Error("db error")):
            result = get_telemetry_data()
        assert result == []


class TestGetAgentsFromRegistry:
    def test_registry_unavailable_returns_defaults(self, monkeypatch):
        monkeypatch.setattr(api_server, "REGISTRY_AVAILABLE", False)
        agents = get_agents_from_registry()
        ids = [a["id"] for a in agents]
        assert "ingest" in ids
        assert len(agents) >= 4

    def test_registry_available_returns_spec_agents(self, monkeypatch):
        monkeypatch.setattr(api_server, "REGISTRY_AVAILABLE", True)
        spec = MagicMock()
        spec.name = "test-agent"
        spec.source = "mcp"
        spec.capabilities = ["cap1"]
        mock_registry = MagicMock()
        mock_registry.list_agents.return_value = [spec]
        with patch.object(api_server, "agent_registry", mock_registry, create=True):
            agents = get_agents_from_registry()
        assert any(a["id"] == "test-agent" for a in agents)

    def test_registry_exception_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setattr(api_server, "REGISTRY_AVAILABLE", True)
        mock_registry = MagicMock()
        mock_registry.list_agents.side_effect = RuntimeError("broken")
        with patch.object(api_server, "agent_registry", mock_registry, create=True):
            agents = get_agents_from_registry()
        assert len(agents) >= 4


class TestFormatGoalsForApi:
    def test_empty_data_returns_empty_list(self):
        assert format_goals_for_api({"queue": [], "in_flight": {}}, []) == []

    def test_queued_goals_have_correct_ids(self):
        result = format_goals_for_api({"queue": ["task1", "task2"], "in_flight": {}}, [])
        assert len(result) == 2
        assert result[0]["id"] == "goal-q-0"
        assert result[0]["status"] == "pending"
        assert result[1]["id"] == "goal-q-1"

    def test_inflight_goals_have_running_status(self):
        import time

        ts = time.time()
        result = format_goals_for_api({"queue": [], "in_flight": {"running task": ts}}, [])
        assert len(result) == 1
        assert result[0]["status"] == "running"
        assert result[0]["progress"] == 50

    def test_archived_goals_string_type(self):
        archive = [{"goal": "done task", "status": "completed", "timestamp": "2024-01-01", "cycles": 2}]
        result = format_goals_for_api({"queue": [], "in_flight": {}}, archive)
        assert len(result) == 1
        assert result[0]["status"] == "completed"
        assert result[0]["progress"] == 100

    def test_archived_goals_dict_goal_type(self):
        # Dict-type goals are not handled by format_goals_for_api (only strings are)
        archive = [{"goal": {"key": "val"}, "status": "failed", "timestamp": "2024-01-01", "cycles": 1}]
        result = format_goals_for_api({"queue": [], "in_flight": {}}, archive)
        assert len(result) == 0

    def test_archive_capped_at_50(self):
        archive = [{"goal": f"g{i}", "status": "completed", "timestamp": "2024-01-01"} for i in range(60)]
        result = format_goals_for_api({"queue": [], "in_flight": {}}, archive)
        assert len(result) == 50


# ===========================================================================
# HTTP route tests
# ===========================================================================


class TestHealthCheckRoute:
    def test_returns_healthy(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_EMPTY_QUEUE):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["goals_queued"] == 0

    def test_reflects_queue_count(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_SAMPLE_QUEUE):
            resp = client.get("/health")
        assert resp.json()["goals_queued"] == 2


class TestAuthLoginRoute:
    def test_no_auth_returns_mock_token(self, client):
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "pass"})
        assert resp.status_code == 200
        assert resp.json()["access_token"] == "mock-token"

    def test_auth_valid_credentials(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "AUTH_AVAILABLE", True)
        mock_user = MagicMock()
        mock_user.to_dict.return_value = {"username": "admin", "role": "admin"}
        mock_auth = MagicMock()
        mock_auth.authenticate_user.return_value = mock_user
        mock_auth.create_access_token.return_value = "real-token"
        with patch("aura_cli.api_server.get_auth_manager", return_value=mock_auth):
            resp = client.post("/api/auth/login", json={"username": "admin", "password": "pass"})
        assert resp.status_code == 200
        assert resp.json()["access_token"] == "real-token"

    def test_auth_invalid_credentials_returns_401(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "AUTH_AVAILABLE", True)
        mock_auth = MagicMock()
        mock_auth.authenticate_user.return_value = None
        with patch("aura_cli.api_server.get_auth_manager", return_value=mock_auth):
            resp = client.post("/api/auth/login", json={"username": "bad", "password": "bad"})
        assert resp.status_code == 401


class TestGetGoalsRoute:
    def test_returns_all_goals(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_SAMPLE_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_filter_by_status(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_SAMPLE_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals?status=pending")
        assert resp.status_code == 200
        assert all(g["status"] == "pending" for g in resp.json())

    def test_empty_goals(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_EMPTY_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filter_running_returns_empty_when_no_inflight(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_SAMPLE_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals?status=running")
        assert resp.status_code == 200
        assert resp.json() == []


class TestCreateGoalRoute:
    def test_create_valid_goal(self, client):
        resp = client.post("/api/goals", json={"description": "New test goal"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["description"] == "New test goal"
        assert data["status"] == "pending"

    def test_create_with_custom_priority(self, client):
        resp = client.post("/api/goals", json={"description": "priority task", "priority": 3, "max_cycles": 5})
        assert resp.status_code == 201
        assert resp.json()["priority"] == 3

    def test_create_blank_description_fails(self, client):
        resp = client.post("/api/goals", json={"description": "   "})
        assert resp.status_code in (400, 422, 500)

    def test_create_with_queue_available(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        mock_gq = MagicMock()
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals", json={"description": "queued goal"})
        assert resp.status_code == 201
        mock_gq.add.assert_called_once_with("queued goal")

    def test_create_queue_error_returns_500(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        mock_gq = MagicMock()
        mock_gq.add.side_effect = RuntimeError("disk full")
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals", json={"description": "will fail"})
        assert resp.status_code == 500


class TestGetGoalDetailRoute:
    def test_queued_goal_found(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_SAMPLE_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals/goal-q-0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "goal-q-0"
        assert data["status"] == "pending"

    def test_inflight_goal_found(self, client):
        import time

        desc = "running task"
        gid = _goal_id_for_inflight(desc)
        queue_data = {"queue": [], "in_flight": {desc: time.time()}}
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=queue_data), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get(f"/api/goals/{gid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_archived_goal_found_with_history(self, client):
        desc = "improve coverage"
        gid = _goal_id_for_archived(desc)
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_EMPTY_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=_SAMPLE_ARCHIVE):
            resp = client.get(f"/api/goals/{gid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["history"]) == 1

    def test_not_found_returns_404(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_EMPTY_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals/nonexistent-id")
        assert resp.status_code == 404


class TestDeleteGoalRoute:
    def test_queue_unavailable_returns_503(self, client):
        resp = client.delete("/api/goals/goal-q-0")
        assert resp.status_code == 503

    def test_inflight_goal_returns_409(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        desc = "running"
        gid = _goal_id_for_inflight(desc)
        mock_gq = MagicMock()
        mock_gq.in_flight_keys.return_value = [desc]
        mock_gq.queue = []
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.delete(f"/api/goals/{gid}")
        assert resp.status_code == 409

    def test_goal_not_in_queue_returns_404(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        mock_gq = MagicMock()
        mock_gq.in_flight_keys.return_value = []
        mock_gq.queue = []
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.delete("/api/goals/goal-q-0")
        assert resp.status_code == 404

    def test_cancel_success(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        mock_gq = MagicMock()
        mock_gq.in_flight_keys.return_value = []
        mock_gq.queue = ["write tests"]
        mock_gq.cancel.return_value = "write tests"
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.delete("/api/goals/goal-q-0")
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestPrioritizeGoalRoute:
    def test_queue_unavailable_returns_503(self, client):
        resp = client.post("/api/goals/goal-q-0/prioritize")
        assert resp.status_code == 503

    def test_inflight_returns_409(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        desc = "running"
        gid = _goal_id_for_inflight(desc)
        mock_gq = MagicMock()
        mock_gq.in_flight_keys.return_value = [desc]
        mock_gq.queue = []
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post(f"/api/goals/{gid}/prioritize")
        assert resp.status_code == 409

    def test_not_found_returns_404(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        mock_gq = MagicMock()
        mock_gq.in_flight_keys.return_value = []
        mock_gq.queue = []
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals/goal-q-99/prioritize")
        assert resp.status_code == 404

    def test_already_at_front(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        mock_gq = MagicMock()
        mock_gq.in_flight_keys.return_value = []
        mock_gq.queue = ["front task"]
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals/goal-q-0/prioritize")
        assert resp.status_code == 200
        data = resp.json()
        assert data["position"] == 0
        assert "already" in data["message"].lower()

    def test_promote_to_front(self, client, monkeypatch):
        monkeypatch.setattr(api_server, "GOAL_QUEUE_AVAILABLE", True)
        mock_gq = MagicMock()
        mock_gq.in_flight_keys.return_value = []
        mock_gq.queue = ["first", "second task"]
        mock_gq.promote.return_value = "second task"
        with patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals/goal-q-1/prioritize")
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestCancelGoalLegacyRoute:
    def test_delegates_to_delete_503(self, client):
        resp = client.post("/api/goals/goal-q-0/cancel")
        assert resp.status_code == 503


class TestInFlightGoalRoute:
    def _mock_tracker(self, exists=False, summary=None):
        tracker = MagicMock()
        tracker.exists.return_value = exists
        tracker.get_summary.return_value = summary or {}
        mod = MagicMock()
        mod.InFlightTracker.return_value = tracker
        return mod

    def test_no_inflight(self):
        # Route is shadowed by /api/goals/{goal_id}; call handler directly
        mod = self._mock_tracker(exists=False)
        with patch.dict(sys.modules, {"core.in_flight_tracker": mod}):
            result = asyncio.run(get_in_flight_goal(user={"user_id": "dev"}))
        assert result["exists"] is False

    def test_with_inflight(self):
        # Route is shadowed by /api/goals/{goal_id}; call handler directly
        mod = self._mock_tracker(exists=True, summary={"goal": "working on it"})
        with patch.dict(sys.modules, {"core.in_flight_tracker": mod}):
            result = asyncio.run(get_in_flight_goal(user={"user_id": "dev"}))
        assert result["exists"] is True


class TestResumeGoalRoute:
    def _make_modules(self, goal_text=None):
        tracker = MagicMock()
        tracker.read.return_value = {"goal": goal_text} if goal_text is not None else None
        tracker.clear = MagicMock()
        in_flight_mod = MagicMock()
        in_flight_mod.InFlightTracker.return_value = tracker
        gq = MagicMock()
        goal_queue_mod = MagicMock()
        goal_queue_mod.GoalQueue.return_value = gq
        return in_flight_mod, goal_queue_mod, tracker, gq

    def test_no_interrupted_goal_returns_404(self, client):
        in_flight_mod, gq_mod, _, _ = self._make_modules(goal_text=None)
        with patch.dict(sys.modules, {"core.in_flight_tracker": in_flight_mod, "core.goal_queue": gq_mod}):
            resp = client.post("/api/goals/resume", json={})
        assert resp.status_code == 404

    def test_empty_goal_field_returns_400(self, client):
        tracker = MagicMock()
        tracker.read.return_value = {"goal": ""}
        in_flight_mod = MagicMock()
        in_flight_mod.InFlightTracker.return_value = tracker
        gq_mod = MagicMock()
        with patch.dict(sys.modules, {"core.in_flight_tracker": in_flight_mod, "core.goal_queue": gq_mod}):
            resp = client.post("/api/goals/resume", json={})
        assert resp.status_code == 400

    def test_resume_success(self, client):
        in_flight_mod, gq_mod, _, _ = self._make_modules(goal_text="resume this")
        with patch.dict(sys.modules, {"core.in_flight_tracker": in_flight_mod, "core.goal_queue": gq_mod}):
            resp = client.post("/api/goals/resume", json={"run": False})
        assert resp.status_code == 200
        assert resp.json()["status"] == "resumed"

    def test_resume_and_run(self, client):
        in_flight_mod, gq_mod, _, _ = self._make_modules(goal_text="run this now")
        with patch.dict(sys.modules, {"core.in_flight_tracker": in_flight_mod, "core.goal_queue": gq_mod}):
            resp = client.post("/api/goals/resume", json={"run": True})
        assert resp.status_code == 200
        assert resp.json()["status"] == "resumed_and_running"


class TestAgentRoutes:
    def test_get_agents_returns_list(self, client):
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        assert len(resp.json()) >= 4

    def test_get_agent_logs_empty(self, client):
        with patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/agents/planner/logs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_agent_logs_filtered(self, client):
        telemetry = [
            {"agent_name": "planner", "timestamp": "2024-01-01", "latency": 1.0},
            {"agent_name": "act", "timestamp": "2024-01-01", "latency": 0.5},
        ]
        with patch("aura_cli.api_server.get_telemetry_data", return_value=telemetry):
            resp = client.get("/api/agents/planner/logs")
        assert len(resp.json()) == 1

    def test_get_agent_metrics_no_data(self, client):
        with patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/agents/planner/metrics")
        assert resp.status_code == 200
        assert resp.json()["total_executions"] == 0

    def test_get_agent_metrics_with_data(self, client):
        telemetry = [
            {"agent_name": "planner", "latency": 2.0, "tokens": 100, "status": "success"},
            {"agent_name": "planner", "latency": 1.5, "tokens": 50, "status": "error"},
        ]
        with patch("aura_cli.api_server.get_telemetry_data", return_value=telemetry):
            resp = client.get("/api/agents/planner/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_executions"] == 2
        assert data["errors"] == 1

    def test_get_agent_history(self, client):
        with patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/agents/act/history")
        assert resp.status_code == 200

    def test_pause_agent(self, client):
        resp = client.post("/api/agents/planner/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_resume_agent(self, client):
        resp = client.post("/api/agents/planner/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_restart_agent(self, client):
        resp = client.post("/api/agents/act/restart")
        assert resp.status_code == 200
        assert resp.json()["status"] == "restarting"

    def test_agents_overview(self, client):
        with patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/agents/overview")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestTelemetryRoutes:
    def test_get_telemetry_empty(self, client):
        with patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/telemetry")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_telemetry_summary_empty(self, client):
        with patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/telemetry/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 0

    def test_get_telemetry_summary_with_data(self, client):
        telemetry = [
            {"agent_name": "p", "latency": 2.0, "tokens": 100, "status": "success", "timestamp": "2024-01-01T10:30:00"},
            {"agent_name": "a", "latency": 1.0, "tokens": 50, "status": "error", "timestamp": "2024-01-01T11:30:00"},
        ]
        with patch("aura_cli.api_server.get_telemetry_data", return_value=telemetry):
            resp = client.get("/api/telemetry/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 2
        assert "by_agent" in data
        assert "by_hour" in data


class TestStatsRoute:
    def test_stats_all_empty(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_EMPTY_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=[]), patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "goals" in data
        assert "agents" in data
        assert "telemetry" in data
        assert "system" in data

    def test_stats_with_queue_data(self, client):
        archive = [{"goal": "done", "status": "completed", "timestamp": "2024-01-01"}]
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_SAMPLE_QUEUE), patch("aura_cli.api_server.get_goal_archive", return_value=archive), patch("aura_cli.api_server.get_telemetry_data", return_value=[]):
            resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["goals"]["pending"] == 2


class TestCoverageRoutes:
    def test_coverage_falls_back_to_mock_on_error(self, client):
        with patch("subprocess.run", side_effect=Exception("no coverage")):
            resp = client.get("/api/coverage")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall" in data

    def test_coverage_gaps_returns_list(self, client):
        resp = client.get("/api/coverage/gaps")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["severity"] in ("critical", "high", "medium", "low")


class TestTestsRoutes:
    def test_get_tests_subprocess_error(self, client):
        with patch("subprocess.run", side_effect=Exception("pytest not found")):
            resp = client.get("/api/tests")
        assert resp.status_code == 200
        assert "total" in resp.json()

    def test_run_tests_started(self, client):
        resp = client.post("/api/tests/run")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


class TestApiHealthRoute:
    def test_healthy_no_failed_agents(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_EMPTY_QUEUE):
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded", "critical")
        assert "checks" in data

    def test_health_checks_structure(self, client):
        with patch("aura_cli.api_server.get_goal_queue_data", return_value=_EMPTY_QUEUE):
            resp = client.get("/api/health")
        checks = resp.json()["checks"]
        assert "agents" in checks
        assert "queue" in checks
        assert "api" in checks


class TestChatRoute:
    def test_chat_import_error_returns_error_response(self, client):
        with patch.dict(sys.modules, {"agents.registry": None, "core.model_adapter": None, "memory.brain": None}):
            resp = client.post("/api/chat", json={"message": "hello", "agent": "planner"})
        assert resp.status_code == 200
        assert "response" in resp.json()

    def test_chat_agent_not_found(self, client):
        mock_reg = MagicMock()
        mock_reg.default_agents.return_value = {}
        mock_model = MagicMock()
        mock_model.ModelAdapter.return_value = MagicMock()
        mock_brain = MagicMock()
        mock_brain.Brain.return_value = MagicMock()
        with patch.dict(
            sys.modules,
            {"agents.registry": mock_reg, "core.model_adapter": mock_model, "memory.brain": mock_brain},
        ):
            resp = client.post("/api/chat", json={"message": "hello", "agent": "unknown"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_chat_agent_found_and_run(self, client):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value={"response": "I can help"})
        mock_reg = MagicMock()
        mock_reg.default_agents.return_value = {"planner": mock_agent}
        mock_model = MagicMock()
        mock_model.ModelAdapter.return_value = MagicMock()
        mock_brain = MagicMock()
        mock_brain.Brain.return_value = MagicMock()
        with patch.dict(
            sys.modules,
            {"agents.registry": mock_reg, "core.model_adapter": mock_model, "memory.brain": mock_brain},
        ):
            resp = client.post("/api/chat", json={"message": "help me", "agent": "planner"})
        assert resp.status_code == 200
        assert "response" in resp.json()


class TestSaddRoutes:
    def test_get_sessions_empty(self, client):
        resp = client.get("/api/sadd/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_session(self, client):
        resp = client.post("/api/sadd/sessions", json={"title": "My Session", "design_spec": "some spec"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "My Session"
        assert data["status"] == "idle"
        assert data["id"].startswith("sadd-")

    def test_create_session_default_title(self, client):
        resp = client.post("/api/sadd/sessions", json={})
        assert resp.status_code == 200
        assert resp.json()["title"] == "Untitled Session"

    def test_get_sessions_after_create(self, client):
        client.post("/api/sadd/sessions", json={"title": "S1"})
        client.post("/api/sadd/sessions", json={"title": "S2"})
        resp = client.get("/api/sadd/sessions")
        assert len(resp.json()) == 2

    def test_start_session(self, client):
        session_id = client.post("/api/sadd/sessions", json={"title": "To start"}).json()["id"]
        resp = client.post(f"/api/sadd/sessions/{session_id}/start")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_start_nonexistent_session_returns_404(self, client):
        resp = client.post("/api/sadd/sessions/nope/start")
        assert resp.status_code == 404

    def test_pause_session(self, client):
        session_id = client.post("/api/sadd/sessions", json={"title": "Active"}).json()["id"]
        resp = client.post(f"/api/sadd/sessions/{session_id}/pause")
        assert resp.status_code == 200

    def test_pause_nonexistent_session_returns_404(self, client):
        resp = client.post("/api/sadd/sessions/nope/pause")
        assert resp.status_code == 404

    def test_resume_session(self, client):
        session_id = client.post("/api/sadd/sessions", json={"title": "Paused"}).json()["id"]
        client.post(f"/api/sadd/sessions/{session_id}/pause")
        resp = client.post(f"/api/sadd/sessions/{session_id}/resume")
        assert resp.status_code == 200

    def test_resume_nonexistent_returns_404(self, client):
        resp = client.post("/api/sadd/sessions/nope/resume")
        assert resp.status_code == 404

    def test_stop_session(self, client):
        session_id = client.post("/api/sadd/sessions", json={"title": "Running"}).json()["id"]
        resp = client.post(f"/api/sadd/sessions/{session_id}/stop")
        assert resp.status_code == 200

    def test_stop_nonexistent_returns_404(self, client):
        resp = client.post("/api/sadd/sessions/nope/stop")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        session_id = client.post("/api/sadd/sessions", json={"title": "Deletable"}).json()["id"]
        resp = client.delete(f"/api/sadd/sessions/{session_id}")
        assert resp.status_code == 200
        ids = [s["id"] for s in client.get("/api/sadd/sessions").json()]
        assert session_id not in ids


class TestWorkflowRoutes:
    def test_get_workflows_no_dir(self, client):
        with patch.object(Path, "exists", return_value=False):
            resp = client.get("/api/workflows")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_workflow_not_found(self, client):
        with patch.object(Path, "exists", return_value=False):
            resp = client.get("/api/workflows/nonexistent")
        assert resp.status_code == 404

    def test_execute_workflow_returns_execution_id(self, client):
        resp = client.post("/api/workflows/my-workflow/execute", json={"param": "value"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "execution_id" in data
        assert data["workflow_id"] == "my-workflow"

    def test_get_workflow_executions_empty(self, client):
        resp = client.get("/api/workflows/my-workflow/executions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_activate_workflow(self, client):
        resp = client.post("/api/workflows/my-workflow/activate")
        assert resp.status_code == 200
        assert resp.json()["active"] is True

    def test_deactivate_workflow(self, client):
        resp = client.post("/api/workflows/my-workflow/deactivate")
        assert resp.status_code == 200
        assert resp.json()["active"] is False


class TestMcpRoutes:
    def test_get_servers_no_config_file(self, client):
        with patch.object(Path, "exists", return_value=False):
            resp = client.get("/api/mcp/servers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_servers_with_config(self, client):
        config = {"mcpServers": {"filesystem": {"type": "stdio", "command": "npx", "args": ["-y", "fs"]}}}
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value=json.dumps(config)):
            resp = client.get("/api/mcp/servers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "filesystem"

    def test_get_mcp_tools_known_server(self, client):
        resp = client.get("/api/mcp/servers/filesystem/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["server_id"] == "filesystem"
        assert len(data["tools"]) > 0

    def test_get_mcp_tools_unknown_server(self, client):
        resp = client.get("/api/mcp/servers/unknown-xyz/tools")
        assert resp.status_code == 200
        assert resp.json()["tools"] == []

    def test_execute_mcp_tool(self, client):
        resp = client.post("/api/mcp/servers/filesystem/tools/read_file/execute", json={"path": "test.txt"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["server_id"] == "filesystem"

    def test_get_server_status(self, client):
        resp = client.get("/api/mcp/servers/filesystem/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "connected"
        assert data["server_id"] == "filesystem"


class TestGithubRoutes:
    def test_list_prs_no_app_client(self):
        # GET routes after catch-all are shadowed; call handler directly
        mock_app = MagicMock()
        mock_app._app_client = None
        mock_gh_mod = MagicMock()
        mock_gh_mod.get_github_app.return_value = mock_app
        with patch.dict(sys.modules, {"aura_cli.github_integration": mock_gh_mod}):
            result = asyncio.run(list_pull_requests(state="open", user={"user_id": "dev"}))
        assert isinstance(result, list)

    def test_list_prs_no_github_app(self):
        # GET routes after catch-all are shadowed; call handler directly
        mock_gh_mod = MagicMock()
        mock_gh_mod.get_github_app.return_value = None
        with patch.dict(sys.modules, {"aura_cli.github_integration": mock_gh_mod}):
            result = asyncio.run(list_pull_requests(state="open", user={"user_id": "dev"}))
        assert isinstance(result, list)

    def test_get_pr_detail(self):
        result = asyncio.run(get_pull_request(pr_number=42, user={"user_id": "dev"}))
        assert result["number"] == 42

    def test_get_pr_reviews(self):
        result = asyncio.run(get_pr_reviews(pr_number=42, user={"user_id": "dev"}))
        assert isinstance(result, list)
        assert result[0]["state"] == "APPROVED"

    def test_get_pr_comments(self):
        result = asyncio.run(get_pr_comments(pr_number=42, user={"user_id": "dev"}))
        assert isinstance(result, list)

    def test_webhook_no_github_app_returns_503(self, client):
        mock_gh_mod = MagicMock()
        mock_gh_mod.get_github_app.return_value = None
        with patch.dict(sys.modules, {"aura_cli.github_integration": mock_gh_mod}):
            resp = client.post(
                "/api/github/webhook",
                content=b'{"action": "opened"}',
                headers={"x-github-event": "pull_request"},
            )
        assert resp.status_code == 503

    def test_webhook_invalid_signature_returns_401(self, client):
        mock_app = MagicMock()
        mock_app.verify_webhook_signature.return_value = False
        mock_gh_mod = MagicMock()
        mock_gh_mod.get_github_app.return_value = mock_app
        with patch.dict(sys.modules, {"aura_cli.github_integration": mock_gh_mod}):
            resp = client.post(
                "/api/github/webhook",
                content=b'{"action": "opened"}',
                headers={
                    "x-github-event": "pull_request",
                    "x-hub-signature-256": "sha256=bad",
                },
            )
        assert resp.status_code == 401

    def test_webhook_success(self, client):
        mock_app = MagicMock()
        mock_app.verify_webhook_signature.return_value = True
        mock_app.handle_webhook.return_value = {"status": "processed"}
        mock_gh_mod = MagicMock()
        mock_gh_mod.get_github_app.return_value = mock_app
        with patch.dict(sys.modules, {"aura_cli.github_integration": mock_gh_mod}):
            resp = client.post(
                "/api/github/webhook",
                content=b'{"action": "opened"}',
                headers={"x-github-event": "issues"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "processed"

    def test_webhook_pr_event_broadcasts(self, client):
        mock_app = MagicMock()
        mock_app.verify_webhook_signature.return_value = True
        mock_app.handle_webhook.return_value = {"status": "queued"}
        mock_gh_mod = MagicMock()
        mock_gh_mod.get_github_app.return_value = mock_app
        payload = json.dumps({"action": "opened", "pull_request": {"number": 1, "title": "PR"}, "repository": {"full_name": "o/r"}, "sender": {"login": "dev"}}).encode()
        with patch.dict(sys.modules, {"aura_cli.github_integration": mock_gh_mod}):
            resp = client.post(
                "/api/github/webhook",
                content=payload,
                headers={"x-github-event": "pull_request"},
            )
        assert resp.status_code == 200

    def test_github_callback(self):
        result = asyncio.run(github_callback(code="abc123", installation_id="999"))
        assert result["status"] == "success"
        assert result["installation_id"] == "999"


class TestBroadcastLog:
    def test_broadcast_log_creates_task(self):
        """broadcast_log should call asyncio.create_task without raising."""
        with patch("asyncio.create_task") as mock_task:
            broadcast_log({"level": "INFO", "msg": "hello"})
            mock_task.assert_called_once()


class TestNotificationRoutes:
    def test_notifications_status(self):
        # GET route after catch-all is shadowed; call handler directly
        mock_mgr = MagicMock()
        mock_mgr.get_status.return_value = {"enabled": True}
        mock_mod = MagicMock()
        mock_mod.get_notification_manager.return_value = mock_mgr
        with patch.dict(sys.modules, {"core.notifications": mock_mod}):
            result = asyncio.run(get_notifications_status(user={"user_id": "dev"}))
        assert result == {"enabled": True}

    def test_test_notification(self, client):
        mock_result = {"sent": True}
        mock_mod = MagicMock()
        mock_mod.notify = AsyncMock(return_value=mock_result)
        mock_mod.NotificationChannel.SLACK = "slack"
        mock_mod.NotificationChannel.DISCORD = "discord"
        mock_mod.NotificationChannel.LOG = "log"
        with patch.dict(sys.modules, {"core.notifications": mock_mod}):
            resp = client.post("/api/notifications/test", json="slack")
        assert resp.status_code == 200


class TestPerformanceRoutes:
    def test_get_performance_stats(self):
        # GET route after catch-all is shadowed; call handler directly
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = {"hits": 10}
        mock_cache_mod = MagicMock()
        mock_cache_mod.get_cache.return_value = mock_cache
        mock_profiler = MagicMock()
        mock_profiler.get_memory_usage.return_value = {"rss_mb": 128}
        with patch.dict(sys.modules, {"core.cache": mock_cache_mod, "core.memory_profiler": mock_profiler}):
            result = asyncio.run(get_performance_stats(user={"user_id": "dev"}))
        assert "cache" in result
        assert "memory" in result

    def test_clear_cache(self, client):
        mock_cache = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_cache.return_value = mock_cache
        with patch.dict(sys.modules, {"core.cache": mock_mod}):
            resp = client.post("/api/performance/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        mock_cache.clear.assert_called_once()


class TestAnalyticsRoutes:
    def _mock_metrics(self, engine_methods=None):
        engine = MagicMock()
        if engine_methods:
            for method, return_val in engine_methods.items():
                getattr(engine, method).return_value = return_val
        mod = MagicMock()
        mod.get_analytics_engine.return_value = engine
        mod.get_metrics_collector.return_value = MagicMock(get_summary=MagicMock(return_value={"total": 0}))
        mod.get_agent_tracker.return_value = MagicMock(
            get_all_performances=MagicMock(return_value=[]),
            get_summary=MagicMock(return_value={}),
            get_recommendations=MagicMock(return_value=[]),
        )
        return mod

    def test_get_metrics_summary(self):
        # GET route after catch-all is shadowed; call handler directly
        mock_mod = self._mock_metrics()
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            result = asyncio.run(get_metrics_summary(user={"user_id": "dev"}))
        assert result == {"total": 0}

    def test_get_goal_trends(self):
        mock_mod = self._mock_metrics({"analyze_goal_trends": {"trend": "up"}})
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            result = asyncio.run(get_goal_trends(days=7, user={"user_id": "dev"}))
        assert result == {"trend": "up"}

    def test_get_system_performance(self):
        mock_mod = self._mock_metrics({"analyze_system_performance": {"cpu": 10}})
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            result = asyncio.run(get_system_performance_analytics(hours=24, user={"user_id": "dev"}))
        assert result == {"cpu": 10}

    def test_get_analytics_insights(self):
        mock_mod = self._mock_metrics({"get_insights": [{"text": "improve coverage"}]})
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            result = asyncio.run(get_analytics_insights(user={"user_id": "dev"}))
        assert result["insights"] == [{"text": "improve coverage"}]

    def test_get_daily_report(self):
        mock_mod = self._mock_metrics({"generate_daily_report": {"date": "2024-01-01"}})
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            result = asyncio.run(get_daily_report(user={"user_id": "dev"}))
        assert result == {"date": "2024-01-01"}

    def test_get_agents_performance(self):
        mock_mod = self._mock_metrics()
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            result = asyncio.run(get_agents_performance(user={"user_id": "dev"}))
        assert "performances" in result

    def test_get_agent_performance_not_found(self, client):
        tracker = MagicMock()
        tracker.get_performance.return_value = None
        mock_mod = MagicMock()
        mock_mod.get_agent_tracker.return_value = tracker
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            resp = client.get("/api/agents/performance/unknown-agent")
        assert resp.status_code == 404

    def test_get_agent_performance_found(self):
        # GET route after catch-all shadowed; call handler directly
        perf = MagicMock()
        perf.to_dict.return_value = {"agent_id": "planner", "score": 95}
        tracker = MagicMock()
        tracker.get_performance.return_value = perf
        mock_mod = MagicMock()
        mock_mod.get_agent_tracker.return_value = tracker
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            result = asyncio.run(get_agent_performance(agent_id="planner", user={"user_id": "dev"}))
        assert result["agent_id"] == "planner"

    def test_reset_agent_performance_success(self, client):
        tracker = MagicMock()
        tracker.reset_agent_stats.return_value = True
        mock_mod = MagicMock()
        mock_mod.get_agent_tracker.return_value = tracker
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            resp = client.post("/api/agents/performance/planner/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"

    def test_reset_agent_performance_not_found(self, client):
        tracker = MagicMock()
        tracker.reset_agent_stats.return_value = False
        mock_mod = MagicMock()
        mock_mod.get_agent_tracker.return_value = tracker
        with patch.dict(sys.modules, {"core.metrics": mock_mod}):
            resp = client.post("/api/agents/performance/unknown/reset")
        assert resp.status_code == 404
