from __future__ import annotations

"""Tests for aura_cli.mcp_swarm_bridge HTTP bridge.

Note: mcp_swarm_bridge uses Python's stdlib http.server (not FastAPI), so we
spin up a real HTTPServer on an ephemeral port in a daemon thread and exercise
it with urllib.  The module-level _mcp MCPProcess is patched so no Node.js
process is ever spawned.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level import helpers
# ---------------------------------------------------------------------------

# The task spec mentions importing `app` (FastAPI); the actual module exposes
# BridgeHandler / HTTPServer instead, so we try gracefully.
try:
    from aura_cli.mcp_swarm_bridge import app as _fastapi_app  # type: ignore[attr-defined]
    _HAS_FASTAPI_APP = True
except ImportError:
    _fastapi_app = None
    _HAS_FASTAPI_APP = False

import aura_cli.mcp_swarm_bridge as _bridge_module  # noqa: E402 – always available


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_mcp():
    """A fully mocked MCPProcess that never touches Node.js."""
    m = MagicMock()
    m.alive.return_value = True
    m.list_tools.return_value = [{"name": "ping", "description": "test tool"}]
    m.call_tool.return_value = {"error": "unknown tool"}
    return m


@pytest.fixture()
def bridge_server(mock_mcp):
    """Spin up a real BridgeHandler HTTPServer on an ephemeral port.

    Patches the module-level _mcp so no Node.js process is spawned.
    Yields (base_url, mock_mcp) and shuts the server down afterwards.
    """
    with patch.object(_bridge_module, "_mcp", mock_mcp):
        server = HTTPServer(("127.0.0.1", 0), _bridge_module.BridgeHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        yield base_url, mock_mcp
        server.shutdown()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url) as resp:
        return resp.status, json.loads(resp.read())


def _post(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_ok(bridge_server):
    """GET /health → 200 with status ok."""
    base_url, _ = bridge_server
    status, body = _get(f"{base_url}/health")
    assert status == 200
    assert body.get("status") == "ok"


def test_tools_list_endpoint(bridge_server):
    """GET /tools → 200 with a list under 'tools' key."""
    base_url, _ = bridge_server
    status, body = _get(f"{base_url}/tools")
    assert status == 200
    assert isinstance(body.get("tools"), list)


def test_call_unknown_tool_returns_error(bridge_server):
    """POST /call with an unknown tool_name → response contains 'error' or data with error."""
    base_url, mock_mcp = bridge_server
    # call_tool returns {"error": "unknown tool"} from the mock
    mock_mcp.call_tool.return_value = {"error": "unknown tool"}
    status, body = _post(f"{base_url}/call", {"tool_name": "nonexistent_tool_xyz", "args": {}})
    # Bridge wraps result in {"data": ...}; data should carry the error
    assert status in (200, 400, 404, 422, 500)
    # Either top-level error key, or nested under "data"
    has_error = (
        "error" in body
        or ("data" in body and "error" in body["data"])
    )
    assert has_error, f"Expected error in response, got: {body}"


def test_module_imports_cleanly():
    """Importing aura_cli.mcp_swarm_bridge must not raise."""
    import importlib
    # Already imported above; re-import to confirm it's clean
    mod = importlib.import_module("aura_cli.mcp_swarm_bridge")
    assert mod is not None


def test_bridge_app_is_fastapi():
    """The bridge uses stdlib http.server, not FastAPI.  Skip if no 'app' attr."""
    if not _HAS_FASTAPI_APP:
        pytest.skip(
            "aura_cli.mcp_swarm_bridge does not expose a FastAPI 'app'; "
            "it uses http.server.HTTPServer instead"
        )
    from fastapi import FastAPI  # type: ignore[import]
    assert isinstance(_fastapi_app, FastAPI)


def test_call_requires_tool_name(bridge_server):
    """POST /call with missing tool_name → 400 with error payload."""
    base_url, _ = bridge_server
    status, body = _post(f"{base_url}/call", {"args": {}})  # no tool_name
    assert status == 400
    assert "error" in body


def test_health_response_has_expected_keys(bridge_server):
    """GET /health response must contain at least the 'status' key."""
    base_url, _ = bridge_server
    _, body = _get(f"{base_url}/health")
    assert "status" in body, f"'status' key missing from health response: {body}"


def test_bridge_port_config():
    """PORT constant defaults to 8050 when MCP_SERVER_PORT env var is unset."""
    import os
    # Only verify the default when the env var is not overriding it
    env_val = os.environ.get("MCP_SERVER_PORT")
    if env_val is not None:
        pytest.skip(f"MCP_SERVER_PORT is set to {env_val!r} in the environment")
    assert _bridge_module.PORT == 8050
