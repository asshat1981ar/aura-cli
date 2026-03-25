"""Comprehensive tests for AURA Agent API (aura_cli/server.py)."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


def _load_server_module():
    fake_rt = {
        "orchestrator": MagicMock(),
        "model_adapter": MagicMock(),
        "memory_store": MagicMock(),
        "config_api_key": None,
    }
    fake_rt["orchestrator"].policy = MagicMock()
    fake_rt["orchestrator"].run_cycle.return_value = {
        "cycle_id": "test-cycle-1",
        "phase_outputs": {"verification": {"status": "pass", "failures": []}},
        "stop_reason": "CONVERGED",
    }
    fake_rt["orchestrator"].run_loop.return_value = {"stop_reason": "CONVERGED", "cycles": 1}
    fake_rt["model_adapter"].respond.return_value = "stub answer"

    sys.modules.pop("aura_cli.server", None)
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
    return _load_server_module()


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


def _sse_payloads(chunks: list[str]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for chunk in chunks:
        if chunk.startswith("data: "):
            payloads.append(json.loads(chunk[len("data: ") :]))
    return payloads


@pytest.fixture(autouse=True)
def api_env(monkeypatch):
    monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")


def test_server_import_does_not_eagerly_create_runtime(monkeypatch):
    sys.modules.pop("aura_cli.server", None)
    with patch("aura_cli.cli_main.create_runtime") as mock_create_runtime:
        importlib.import_module("aura_cli.server")
        mock_create_runtime.assert_not_called()


def test_health_endpoint_auth_failure(server_module):
    with pytest.raises(HTTPException) as exc:
        server_module.require_auth(None)
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        server_module.require_auth("Bearer wrong-token")
    assert exc.value.status_code == 403


def test_health_endpoint_success(server_module):
    data = _run(server_module.health())
    assert data["status"] == "ok"
    assert "providers" in data
    assert "run_enabled" in data


def test_health_endpoint_succeeds_without_bound_runtime_components(server_module):
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
        assert data["status"] == "ok"
        assert "providers" in data
    finally:
        server_module.runtime = original_runtime
        server_module.orchestrator = original_orchestrator
        server_module.model_adapter = original_model_adapter
        server_module.memory_store = original_memory_store


def test_metrics_endpoint_success(server_module):
    data = _run(server_module.metrics())
    assert data["status"] == "ok"
    assert "skill_metrics" in data
    assert data["skill_metrics"]["registered_services"] >= 0


def test_tools_endpoint_success(server_module):
    data = _run(server_module.tools())
    assert data["status"] == "success"
    assert isinstance(data["tools"], list)


def test_execute_ask_success(server_module):
    with patch.object(server_module.model_adapter, "respond", return_value="Test answer"):
        response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="ask", args=["What is 2+2?"])))
    assert response["status"] == "success"
    assert response["data"] == "Test answer"


def test_execute_ask_requires_runtime_component(server_module):
    original = server_module.model_adapter
    server_module.model_adapter = None
    try:
        with pytest.raises(HTTPException) as exc:
            _run(server_module.execute(server_module.ExecuteRequest(tool_name="ask", args=["What is 2+2?"])))
        assert exc.value.status_code == 503
    finally:
        server_module.model_adapter = original


def test_execute_env_is_disabled(server_module):
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="env", args=[])))
    assert exc.value.status_code == 501


def test_execute_run_streaming(server_module):
    cmd = f"{sys.executable} -c \"print('comprehensive-run-ok')\""
    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
    payloads = _sse_payloads(_run(_collect_streaming_response(response)))
    start_evt = next(evt for evt in payloads if evt.get("type") == "start")
    assert start_evt["command"] == cmd
    assert isinstance(start_evt["pid"], int)
    assert any(evt.get("type") == "stdout" and "comprehensive-run-ok" in evt.get("data", "") for evt in payloads)
    assert any(evt.get("type") == "exit" and evt.get("code") == 0 for evt in payloads)


def test_execute_run_scrubs_environment_variables(server_module, monkeypatch):
    monkeypatch.setenv("AURA_SECRET_TOKEN", "super-secret")
    cmd = f"{sys.executable} -c \"import os; print(os.getenv('AURA_SECRET_TOKEN', 'missing'))\""

    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
    payloads = _sse_payloads(_run(_collect_streaming_response(response)))
    stdout_chunks = [evt.get("data", "") for evt in payloads if evt.get("type") == "stdout"]

    assert any("missing" in chunk for chunk in stdout_chunks)
    assert all("super-secret" not in chunk for chunk in stdout_chunks)


def test_execute_run_truncates_output_when_limit_is_hit(server_module, monkeypatch):
    monkeypatch.setattr(server_module, "RUN_TOOL_MAX_OUTPUT_BYTES", 64, raising=False)
    cmd = f"{sys.executable} -c \"print('x' * 512)\""

    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
    payloads = _sse_payloads(_run(_collect_streaming_response(response)))

    assert any(evt.get("type") == "truncated" for evt in payloads)
    stdout_chunks = [evt.get("data", "") for evt in payloads if evt.get("type") == "stdout"]
    assert sum(len(chunk) for chunk in stdout_chunks) <= 64
    assert any(evt.get("type") == "exit" and evt.get("truncated") is True for evt in payloads)


def test_execute_goal_streaming(server_module):
    mock_entry = {
        "cycle_id": "c1",
        "stop_reason": "test_stop",
        "phase_outputs": {"verification": {"status": "pass"}},
    }
    server_module.orchestrator.run_cycle.return_value = mock_entry
    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="goal", args=["Fix the bug"])))
    chunks = _run(_collect_streaming_response(response))
    payloads = [json.loads(chunk[len("data: ") :]) for chunk in chunks if chunk.startswith("data: ")]
    assert any(evt.get("type") == "start" for evt in payloads if isinstance(evt, dict))
    assert any(evt.get("type") == "cycle" for evt in payloads if isinstance(evt, dict))
    assert any(evt.get("type") == "complete" for evt in payloads if isinstance(evt, dict))


def test_execute_goal_requires_args(server_module):
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="goal", args=[])))
    assert exc.value.status_code == 400


def test_execute_goal_requires_runtime_component(server_module):
    original = server_module.orchestrator
    server_module.orchestrator = None
    try:
        with pytest.raises(HTTPException) as exc:
            _run(server_module.execute(server_module.ExecuteRequest(tool_name="goal", args=["Fix the bug"])))
        assert exc.value.status_code == 503
    finally:
        server_module.orchestrator = original


def test_execute_unknown_tool(server_module):
    with pytest.raises(HTTPException):
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="invalid", args=[])))


def test_execute_run_disabled(server_module, monkeypatch):
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "0")
    with pytest.raises(HTTPException):
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=["ls"])))
