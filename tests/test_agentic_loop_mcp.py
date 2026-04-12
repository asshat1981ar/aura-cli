"""Tests for tools/agentic_loop_mcp.py."""

from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _DirectClient:
    def __init__(self, module):
        self._module = module

    def get(self, path):
        if path == "/health":
            return _Response(200, asyncio.run(self._module.health(None)))
        if path == "/discovery":
            return _Response(200, asyncio.run(self._module.discovery(None)))
        if path == "/tools":
            return _Response(200, asyncio.run(self._module.list_tools(None)))
        raise AssertionError(f"Unhandled GET path in test harness: {path}")


def _make_engine_mock():
    engine = MagicMock()
    engine.list_executions.return_value = [{"status": "running"}, {"status": "completed"}]
    engine.list_loops.return_value = [{"status": "running"}, {"status": "paused"}]
    engine.list_definitions.return_value = [{"name": "wf1"}, {"name": "wf2"}]
    return engine


def test_health_and_discovery():
    with patch("tools.agentic_loop_mcp.get_engine", return_value=_make_engine_mock()):
        import tools.agentic_loop_mcp as mod

        client = _DirectClient(mod)

        health = client.get("/health")
        assert health.status_code == 200
        health_payload = health.json()
        assert health_payload["status"] == "ok"
        assert health_payload["tool_count"] >= 1
        assert health_payload["server"] == "aura-agentic-loop"

        discovery = client.get("/discovery")
        assert discovery.status_code == 200
        discovery_payload = discovery.json()
        assert discovery_payload["current_server"]["name"] == "aura-agentic-loop"
        assert any(server["name"] == "aura-control" for server in discovery_payload["servers"])


def test_tools_use_canonical_descriptor_shape():
    with patch("tools.agentic_loop_mcp.get_engine", return_value=_make_engine_mock()):
        import tools.agentic_loop_mcp as mod

        client = _DirectClient(mod)
        response = client.get("/tools")
        assert response.status_code == 200
        tools = response.json()["tools"]
        assert any(tool["name"] == "workflow_run" for tool in tools)
        for tool in tools:
            assert "description" in tool
            assert "inputSchema" in tool
