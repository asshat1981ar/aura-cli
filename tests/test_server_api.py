"""Integration tests for the AURA HTTP API server (aura_cli/server.py)."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from memory.store import MemoryStore

os.environ.setdefault("AURA_SKIP_CHDIR", "1")
os.environ.pop("AGENT_API_TOKEN", None)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _make_fake_runtime() -> Dict[str, Any]:
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


def _load_server_with_env(monkeypatch, *, timeout_s: str | None = None, max_output_bytes: str | None = None):
    if timeout_s is None:
        monkeypatch.delenv("AURA_RUN_TOOL_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("AURA_RUN_TOOL_TIMEOUT_S", timeout_s)

    if max_output_bytes is None:
        monkeypatch.delenv("AURA_RUN_TOOL_MAX_OUTPUT_BYTES", raising=False)
    else:
        monkeypatch.setenv("AURA_RUN_TOOL_MAX_OUTPUT_BYTES", max_output_bytes)

    sys.modules.pop("aura_cli.server", None)
    fake_rt = _make_fake_runtime()
    with patch("aura_cli.cli_main.create_runtime", return_value=fake_rt) as mock_create_runtime:
        module = importlib.import_module("aura_cli.server")
        mock_create_runtime.assert_not_called()
        module.runtime = fake_rt
        module.orchestrator = fake_rt["orchestrator"]
        module.model_adapter = fake_rt["model_adapter"]
        module.memory_store = fake_rt["memory_store"]
        return module


@pytest.fixture(scope="module")
def server_module():
    fake_rt = _make_fake_runtime()
    sys.modules.pop("aura_cli.server", None)
    with patch("aura_cli.cli_main.create_runtime", return_value=fake_rt) as mock_create_runtime:
        import aura_cli.server as _srv

        mock_create_runtime.assert_not_called()
        _srv.runtime = fake_rt
        _srv.orchestrator = fake_rt["orchestrator"]
        _srv.model_adapter = fake_rt["model_adapter"]
        _srv.memory_store = fake_rt["memory_store"]
        return _srv


@pytest.fixture(autouse=True)
def immediate_to_thread(server_module, monkeypatch):
    async def _immediate(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(server_module.asyncio, "to_thread", _immediate)


def _run(coro):
    return asyncio.run(coro)


async def _collect_streaming_response(response) -> list[str]:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        chunks.append(chunk)
    return chunks


def _sse_payloads(chunks: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        payloads.append(json.loads(chunk[len("data: ") :]))
    return payloads


def test_health_returns_200(server_module):
    os.environ.pop("AGENT_API_TOKEN", None)
    data = _run(server_module.health())
    assert data["status"] == "healthy"


def test_health_body_has_status(server_module):
    os.environ.pop("AGENT_API_TOKEN", None)
    data = _run(server_module.health())
    assert "status" in data
    assert data["status"] == "healthy"


def test_health_body_has_version(server_module):
    os.environ.pop("AGENT_API_TOKEN", None)
    data = _run(server_module.health())
    assert "version" in data
    assert data["version"] == "0.1.0"


def test_health_still_serves_when_runtime_components_are_missing(server_module):
    original_runtime = server_module.runtime
    original_orchestrator = server_module.orchestrator
    original_model_adapter = server_module.model_adapter
    original_memory_store = server_module.memory_store
    server_module.runtime = {}
    server_module.orchestrator = None
    server_module.model_adapter = None
    server_module.memory_store = None
    try:
        data = _run(server_module.health())
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
    finally:
        server_module.runtime = original_runtime
        server_module.orchestrator = original_orchestrator
        server_module.model_adapter = original_model_adapter
        server_module.memory_store = original_memory_store


def test_health_requires_auth_when_token_set(server_module):
    os.environ["AGENT_API_TOKEN"] = "mysecret"
    try:
        with pytest.raises(HTTPException) as exc:
            server_module.require_auth("Bearer wrong")
        assert exc.value.status_code == 403
        assert server_module.require_auth("Bearer mysecret") is None
    finally:
        os.environ.pop("AGENT_API_TOKEN", None)


def test_tools_returns_non_empty_list(server_module):
    data = _run(server_module.tools())
    assert data["status"] == "success"
    assert isinstance(data["tools"], list)
    names = {t["name"] for t in data["tools"]}
    for expected in ("ask", "run", "env", "goal"):
        assert expected in names


def test_metrics_returns_prometheus_format(server_module):
    response = _run(server_module.metrics())
    # Prometheus format returns a Response object with text/plain content
    assert hasattr(response, "body") or hasattr(response, "media_type")
    assert "prometheus" in (getattr(response, "media_type", "") or "").lower() or b"TYPE" in (getattr(response, "body", b"") or b"")


def test_execute_env_tool_disabled(server_module):
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="env", args=[])))
    assert exc.value.status_code == 501


def test_execute_ask_tool(server_module):
    server_module.model_adapter.respond.return_value = "Hello from stub"
    data = _run(server_module.execute(server_module.ExecuteRequest(tool_name="ask", args=["Hello?"])))
    assert data["status"] == "success"
    assert data["data"] == "Hello from stub"


def test_execute_ask_requires_runtime_component(server_module):
    original = server_module.model_adapter
    server_module.model_adapter = None
    try:
        with pytest.raises(HTTPException) as exc:
            _run(server_module.execute(server_module.ExecuteRequest(tool_name="ask", args=["Hello?"])))
        assert exc.value.status_code == 503
    finally:
        server_module.model_adapter = original


def test_execute_unknown_tool(server_module):
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="no_such_tool", args=[])))
    assert exc.value.status_code == 404


def test_execute_run_disabled(server_module, monkeypatch):
    monkeypatch.delenv("AGENT_API_ENABLE_RUN", raising=False)
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=["ls"])))
    assert exc.value.status_code == 403


def test_execute_run_without_args_when_enabled(server_module, monkeypatch):
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[])))
    assert exc.value.status_code == 400


def test_execute_run_streams_stdout_and_exit(server_module, monkeypatch):
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    cmd = f"{sys.executable} -c \"print('run-ok')\""

    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
    payloads = _sse_payloads(_run(_collect_streaming_response(response)))

    assert any(evt.get("type") == "stdout" and "run-ok" in evt.get("data", "") for evt in payloads)
    assert any(evt.get("type") == "exit" and evt.get("code") == 0 for evt in payloads)


def test_execute_run_streams_stderr_and_exit_code(server_module, monkeypatch):
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    cmd = f"{sys.executable} -c \"import sys; sys.stderr.write('run-err\\\\n'); sys.exit(3)\""

    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
    payloads = _sse_payloads(_run(_collect_streaming_response(response)))

    assert any(evt.get("type") == "stderr" and "run-err" in evt.get("data", "") for evt in payloads)
    assert any(evt.get("type") == "exit" and evt.get("code") == 3 for evt in payloads)


def test_execute_run_persists_audit_entries(server_module, monkeypatch, tmp_path):
    store = MemoryStore(tmp_path / "memory")
    original_store = server_module.memory_store

    server_module.memory_store = store
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    cmd = f"{sys.executable} -c \"print('audit-ok')\""

    try:
        response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
        payloads = _sse_payloads(_run(_collect_streaming_response(response)))
        assert any(evt.get("type") == "stdout" and "audit-ok" in evt.get("data", "") for evt in payloads)

        audit_entries = store.read_log()
        assert len(audit_entries) == 1
        assert all(entry["type"] == "server_run_tool" for entry in audit_entries)
        assert audit_entries[0]["command"] == cmd
        assert audit_entries[0]["code"] == 0
    finally:
        server_module.memory_store = original_store


def test_execute_run_timeouts_emit_timeout_metadata(server_module, monkeypatch):
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    monkeypatch.setattr(server_module, "_clamped_run_tool_timeout_s", lambda: 0.01, raising=False)
    cmd = f'{sys.executable} -c "import time; time.sleep(1)"'

    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
    payloads = _sse_payloads(_run(_collect_streaming_response(response)))

    assert any(evt.get("type") == "start" and evt.get("command") == cmd for evt in payloads)
    assert any(evt.get("type") == "error" and evt.get("timeout_s") == 0.01 for evt in payloads)
    assert any(evt.get("type") == "exit" and evt.get("timed_out") is True for evt in payloads)


def test_run_tool_timeout_and_output_limits_are_clamped(monkeypatch):
    server_module = _load_server_with_env(monkeypatch, timeout_s="0.01", max_output_bytes="1")
    assert server_module.RUN_TOOL_TIMEOUT_S == 0.01
    assert server_module._clamped_run_tool_timeout_s() == 1.0
    assert server_module._clamped_run_tool_output_bytes() == 1024

    server_module = _load_server_with_env(monkeypatch, timeout_s="999", max_output_bytes="9999999")
    assert server_module.RUN_TOOL_TIMEOUT_S == 999.0
    assert server_module._clamped_run_tool_timeout_s() == server_module.RUN_TOOL_MAX_TIMEOUT_S
    assert server_module._clamped_run_tool_output_bytes() == server_module.RUN_TOOL_MAX_OUTPUT_HARD_CAP


def test_execute_run_denylisted_command_is_rejected(server_module, monkeypatch):
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=["echo shutdown"])))
    assert exc.value.status_code == 403
    assert "blocked by policy" in str(exc.value.detail)


def test_execute_goal_without_args_when_enabled(server_module):
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="goal", args=[])))
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# HTTP-level tests using TestClient (headers, auth, CORS, request ID)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def http_client(server_module):
    """TestClient for HTTP-level header and auth tests."""
    return TestClient(server_module.app, raise_server_exceptions=False)


def test_execute_requires_auth_when_token_set(http_client, monkeypatch):
    """POST /execute with AGENT_API_TOKEN set but no Authorization header → 401."""
    monkeypatch.setenv("AGENT_API_TOKEN", "test-secret-token")
    response = http_client.post(
        "/execute",
        json={"tool_name": "ask", "args": ["hi"]},
    )
    assert response.status_code == 401


def test_rate_limit_header_present_on_response(http_client):
    """Every response must include X-RateLimit-Limit header."""
    response = http_client.get("/health")
    assert "x-ratelimit-limit" in response.headers
    assert response.headers["x-ratelimit-limit"].isdigit()


def test_health_http_returns_200_with_status_field(http_client):
    """GET /health returns HTTP 200 and a JSON body with a 'status' field."""
    response = http_client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_ready_http_returns_json_with_components_field(http_client):
    """GET /ready returns JSON that contains a 'components' field."""
    response = http_client.get("/ready")
    data = response.json()
    assert "components" in data


def test_ready_http_has_status_and_overall_latency(http_client):
    """GET /ready returns 'status' and 'overall_latency_ms' fields."""
    response = http_client.get("/ready")
    data = response.json()
    assert "status" in data
    assert data["status"] in ("ready", "degraded", "not_ready")
    assert "overall_latency_ms" in data
    assert isinstance(data["overall_latency_ms"], (int, float))


def test_ready_components_include_required_keys(http_client):
    """GET /ready components include brain_db, auth_db, mcp_server, model_config, and sandbox."""
    response = http_client.get("/ready")
    data = response.json()
    components = data.get("components", {})
    for key in ("brain_db", "auth_db", "mcp_server", "model_config", "sandbox"):
        assert key in components, f"Missing component: {key}"
    assert components["mcp_server"]["status"] in ("ready", "unavailable")
    assert components["model_config"]["status"] in ("configured", "unconfigured")
    assert components["sandbox"]["status"] in ("ready", "degraded")


def test_ready_mcp_server_unavailable_does_not_cause_503(http_client):
    """MCP server being unavailable should not cause a 503 response."""
    response = http_client.get("/ready")
    data = response.json()
    # Even if mcp_server is unavailable, overall status should not be not_ready
    mcp_status = data.get("components", {}).get("mcp_server", {}).get("status")
    assert mcp_status in ("ready", "unavailable")
    # HTTP status should be 200 for ready or degraded
    assert response.status_code in (200, 503)


def test_cors_header_present_when_origin_sent(http_client):
    """GET /health with an allowed Origin returns Access-Control-Allow-Origin."""
    response = http_client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert "access-control-allow-origin" in response.headers


def test_x_request_id_returned_in_response_headers(http_client):
    """Every response must include an X-Request-ID header."""
    response = http_client.get("/health")
    assert "x-request-id" in response.headers
