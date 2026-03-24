"""Integration tests for the AURA HTTP API server (aura_cli/server.py)."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

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


def test_health_returns_200(server_module):
    os.environ.pop("AGENT_API_TOKEN", None)
    data = _run(server_module.health())
    assert data["status"] == "ok"


def test_health_body_has_status(server_module):
    os.environ.pop("AGENT_API_TOKEN", None)
    data = _run(server_module.health())
    assert "status" in data
    assert data["status"] in ("ok", "degraded")


def test_health_body_has_providers(server_module):
    os.environ.pop("AGENT_API_TOKEN", None)
    data = _run(server_module.health())
    assert "providers" in data
    assert isinstance(data["providers"], dict)


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


def test_metrics_has_skill_metrics(server_module):
    data = _run(server_module.metrics())
    assert data["status"] == "ok"
    assert "skill_metrics" in data


def test_execute_env_tool(server_module):
    data = _run(server_module.execute(server_module.ExecuteRequest(tool_name="env", args=[])))
    assert data["status"] == "success"
    assert isinstance(data["data"], dict)


def test_execute_ask_tool(server_module):
    server_module.model_adapter.respond.return_value = "Hello from stub"
    data = _run(server_module.execute(server_module.ExecuteRequest(tool_name="ask", args=["Hello?"])))
    assert data["status"] == "success"
    assert data["data"] == "Hello from stub"


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
