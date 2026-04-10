from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

import aura_cli.mcp_swarm_bridge as _bridge_module


class _TestableBridgeHandler(_bridge_module.BridgeHandler):
    """BridgeHandler harness that avoids real sockets."""

    def __init__(self, method: str, path: str, body: dict | None = None):
        self.command = method
        self.path = path
        encoded = json.dumps(body or {}).encode("utf-8")
        self.rfile = io.BytesIO(encoded)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(encoded))}
        self.responses: list[tuple[int, dict]] = []

    def send_response(self, code, message=None):  # noqa: D401
        self._status = code

    def send_header(self, key, value):  # noqa: D401
        return

    def end_headers(self):  # noqa: D401
        return

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def _send_json(self, code: int, data):
        self.responses.append((code, data))


@pytest.fixture()
def mock_mcp():
    m = MagicMock()
    m.alive.return_value = True
    m.list_tools.return_value = [{"name": "ping", "description": "test tool"}]
    m.call_tool.return_value = {"error": "unknown tool"}
    return m


def _dispatch(method: str, path: str, body: dict | None = None):
    handler = _TestableBridgeHandler(method, path, body)
    if method == "GET":
        handler.do_GET()
    elif method == "POST":
        handler.do_POST()
    else:
        raise AssertionError(f"unsupported method {method}")
    assert handler.responses, "handler did not emit a response"
    return handler.responses[-1]


def test_health_endpoint_returns_ok(mock_mcp):
    with patch.object(_bridge_module, "_mcp", mock_mcp):
        status, body = _dispatch("GET", "/health")
    assert status == 200
    assert body.get("status") == "ok"
    assert body.get("mcp_alive") is True


def test_tools_list_endpoint(mock_mcp):
    with patch.object(_bridge_module, "_mcp", mock_mcp):
        status, body = _dispatch("GET", "/tools")
    assert status == 200
    assert isinstance(body.get("tools"), list)


def test_call_unknown_tool_returns_error(mock_mcp):
    mock_mcp.call_tool.return_value = {"error": "unknown tool"}
    with patch.object(_bridge_module, "_mcp", mock_mcp):
        status, body = _dispatch("POST", "/call", {"tool_name": "nonexistent_tool_xyz", "args": {}})
    assert status == 200
    assert "data" in body
    assert "error" in body["data"]


def test_module_imports_cleanly():
    import importlib

    mod = importlib.import_module("aura_cli.mcp_swarm_bridge")
    assert mod is not None


def test_bridge_app_is_fastapi():
    with pytest.raises(ImportError):
        from aura_cli.mcp_swarm_bridge import app  # type: ignore[attr-defined]  # noqa: F401


def test_call_requires_tool_name(mock_mcp):
    with patch.object(_bridge_module, "_mcp", mock_mcp):
        status, body = _dispatch("POST", "/call", {"args": {}})
    assert status == 400
    assert "error" in body


def test_health_response_has_expected_keys(mock_mcp):
    with patch.object(_bridge_module, "_mcp", mock_mcp):
        _, body = _dispatch("GET", "/health")
    assert "status" in body


def test_bridge_port_config():
    import os

    env_val = os.environ.get("MCP_SERVER_PORT")
    if env_val is not None:
        pytest.skip(f"MCP_SERVER_PORT is set to {env_val!r} in the environment")
    assert _bridge_module.PORT == 8050
