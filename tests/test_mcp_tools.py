from __future__ import annotations

import asyncio
import importlib
import sys
from unittest.mock import patch

import pytest


def _load_server_module():
    fake_rt = {
        "orchestrator": object(),
        "model_adapter": object(),
        "memory_store": None,
        "config_api_key": None,
    }
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


def _run(coro):
    return asyncio.run(coro)


def test_server_import_does_not_call_create_runtime():
    sys.modules.pop("aura_cli.server", None)
    with patch("aura_cli.cli_main.create_runtime") as mock_create_runtime:
        importlib.import_module("aura_cli.server")
        mock_create_runtime.assert_not_called()


def test_rate_limit_and_health(monkeypatch, server_module):
    monkeypatch.setenv("AGENT_API_TOKEN", "t")
    data = _run(server_module.health())
    assert "status" in data and "providers" in data


def test_list_tools(monkeypatch, server_module):
    monkeypatch.setenv("AGENT_API_TOKEN", "t")
    tools_data = _run(server_module.tools())["tools"]
    names = {t["name"] for t in tools_data}
    expected = {"ask", "run", "env", "goal"}
    for expected_tool in expected:
        assert expected_tool in names
    for tool in tools_data:
        assert "inputSchema" in tool


def test_discovery(monkeypatch, server_module):
    monkeypatch.setenv("AGENT_API_TOKEN", "t")
    data = _run(server_module.discovery())
    assert data["current_server"]["name"] == "aura-dev-tools"
    assert any(server["name"] == "aura-copilot" for server in data["servers"])
    assert any(env["name"] == "claude-code" for env in data["supported_environments"])


def test_environments(monkeypatch, server_module):
    monkeypatch.setenv("AGENT_API_TOKEN", "t")
    data = _run(server_module.environments())
    envs = {env["name"]: env for env in data["environments"]}
    assert {"gemini-cli", "claude-code", "codex-cli"} <= set(envs)
    assert envs["claude-code"]["cli_command"] == "claude"


def test_architecture(monkeypatch, server_module):
    monkeypatch.setenv("AGENT_API_TOKEN", "t")
    data = _run(server_module.architecture())
    assert data["routing"]["strategy"] == "health-aware-round-robin"
    assert any(backend["name"] == "neo4j" for backend in data["knowledge_backends"])
    assert any(env["name"] == "codex-cli" for env in data["supported_environments"])
