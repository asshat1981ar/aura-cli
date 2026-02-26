"""R7: Integration tests for the AURA HTTP API server (aura_cli/server.py).

Uses FastAPI TestClient to exercise all public endpoints:
  GET  /health
  GET  /tools
  GET  /metrics
  POST /execute  (tool_name=env)

The tests mock out the heavyweight runtime objects so no real model calls
are made and no filesystem mutations occur.

Always sets AURA_SKIP_CHDIR=1 to prevent os.chdir() side-effects.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Prevent chdir side-effects that break other tests
os.environ.setdefault("AURA_SKIP_CHDIR", "1")
# Disable bearer-token auth for all tests in this module (re-enabled per test)
os.environ.pop("AGENT_API_TOKEN", None)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake runtime so server.py can be imported
# ---------------------------------------------------------------------------

def _make_fake_runtime() -> Dict[str, Any]:
    """Return a dict that satisfies all server.py accesses of `runtime`."""
    fake_policy = MagicMock()
    fake_policy.evaluate.return_value = "CONVERGED"

    fake_orchestrator = MagicMock()
    fake_orchestrator.policy = fake_policy
    fake_orchestrator.run_cycle.return_value = {
        "cycle_id": "test-cycle-1",
        "phase_outputs": {"verification": {"status": "pass", "failures": []}},
        "stop_reason": "CONVERGED",
    }
    fake_orchestrator.run_loop.return_value = {"stop_reason": "CONVERGED", "cycles": 1}

    fake_model = MagicMock()
    fake_model.respond.return_value = "stub answer"

    fake_memory = MagicMock()
    fake_memory.put.return_value = None

    return {
        "orchestrator": fake_orchestrator,
        "model_adapter": fake_model,
        "memory_store": fake_memory,
        "config_api_key": None,
    }


# ---------------------------------------------------------------------------
# Module-level fixture: import server with mocked runtime
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_module():
    """Import aura_cli.server with a fake runtime, return the module."""
    fake_rt = _make_fake_runtime()
    with patch("aura_cli.cli_main.create_runtime", return_value=fake_rt):
        import importlib
        import aura_cli.server as _srv
        # Patch module-level objects in case module was already imported
        _srv.runtime = fake_rt
        _srv.orchestrator = fake_rt["orchestrator"]
        _srv.model_adapter = fake_rt["model_adapter"]
        _srv.memory_store = fake_rt["memory_store"]
        return _srv


@pytest.fixture(scope="module")
def client(server_module):
    from fastapi.testclient import TestClient
    return TestClient(server_module.app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        os.environ.pop("AGENT_API_TOKEN", None)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status(self, client):
        os.environ.pop("AGENT_API_TOKEN", None)
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_body_has_providers(self, client):
        os.environ.pop("AGENT_API_TOKEN", None)
        data = client.get("/health").json()
        assert "providers" in data
        assert isinstance(data["providers"], dict)

    def test_health_requires_auth_when_token_set(self, client):
        os.environ["AGENT_API_TOKEN"] = "mysecret"
        try:
            r = client.get("/health")
            assert r.status_code == 401
            r2 = client.get("/health", headers={"Authorization": "Bearer mysecret"})
            assert r2.status_code == 200
        finally:
            os.environ.pop("AGENT_API_TOKEN", None)


# ---------------------------------------------------------------------------
# GET /tools
# ---------------------------------------------------------------------------

class TestToolsEndpoint:
    def test_tools_returns_200(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200

    def test_tools_body_has_tools_list(self, client):
        data = client.get("/tools").json()
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_tools_list_not_empty(self, client):
        data = client.get("/tools").json()
        assert len(data["tools"]) > 0

    def test_tools_each_has_name_and_description(self, client):
        data = client.get("/tools").json()
        for tool in data["tools"]:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"

    def test_tools_contains_expected_names(self, client):
        data = client.get("/tools").json()
        names = {t["name"] for t in data["tools"]}
        for expected in ("ask", "run", "env", "goal"):
            assert expected in names, f"Expected tool '{expected}' not found in {names}"


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_has_status_ok(self, client):
        data = client.get("/metrics").json()
        assert data.get("status") == "ok"

    def test_metrics_has_skill_metrics(self, client):
        data = client.get("/metrics").json()
        assert "skill_metrics" in data


# ---------------------------------------------------------------------------
# POST /execute — env tool (no external deps)
# ---------------------------------------------------------------------------

class TestExecuteEnvTool:
    def test_env_returns_200(self, client):
        resp = client.post("/execute", json={"tool_name": "env", "args": []})
        assert resp.status_code == 200

    def test_env_body_has_status_success(self, client):
        data = client.post("/execute", json={"tool_name": "env", "args": []}).json()
        assert data.get("status") == "success"

    def test_env_body_has_data(self, client):
        data = client.post("/execute", json={"tool_name": "env", "args": []}).json()
        assert "data" in data
        assert isinstance(data["data"], dict)


# ---------------------------------------------------------------------------
# POST /execute — unknown tool → 404
# ---------------------------------------------------------------------------

class TestExecuteUnknownTool:
    def test_unknown_tool_returns_404(self, client):
        resp = client.post("/execute", json={"tool_name": "no_such_tool", "args": []})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /execute — ask tool (mocked model adapter)
# ---------------------------------------------------------------------------

class TestExecuteAskTool:
    def test_ask_without_args_returns_400(self, client):
        resp = client.post("/execute", json={"tool_name": "ask", "args": []})
        assert resp.status_code == 400

    def test_ask_with_question_returns_200(self, client, server_module):
        server_module.model_adapter.respond.return_value = "Hello from stub"
        resp = client.post("/execute", json={"tool_name": "ask", "args": ["Hello?"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "success"
        assert "data" in data


# ---------------------------------------------------------------------------
# POST /execute — run tool disabled by default
# ---------------------------------------------------------------------------

class TestExecuteRunToolDisabled:
    def test_run_disabled_without_flag(self, client):
        os.environ.pop("AGENT_API_ENABLE_RUN", None)
        resp = client.post("/execute", json={"tool_name": "run", "args": ["echo hi"]})
        assert resp.status_code == 403

    def test_run_without_args_is_400_when_enabled(self, client):
        os.environ["AGENT_API_ENABLE_RUN"] = "1"
        try:
            resp = client.post("/execute", json={"tool_name": "run", "args": []})
            assert resp.status_code == 400
        finally:
            os.environ.pop("AGENT_API_ENABLE_RUN", None)


# ---------------------------------------------------------------------------
# Auth guard on all endpoints
# ---------------------------------------------------------------------------

class TestAuthGuard:
    def test_all_endpoints_reject_bad_token(self, client):
        os.environ["AGENT_API_TOKEN"] = "correct"
        try:
            for path, method in [
                ("/health", "GET"),
                ("/tools", "GET"),
                ("/metrics", "GET"),
            ]:
                r = client.request(method, path, headers={"Authorization": "Bearer wrong"})
                assert r.status_code == 403, f"{method} {path} should return 403 for bad token"
        finally:
            os.environ.pop("AGENT_API_TOKEN", None)
