"""
Comprehensive pytest tests for aura_cli.mcp_swarm_bridge.

Tests cover:
- MCPProcess lifecycle and subprocess communication
- JSON-RPC 2.0 message handling
- HTTP endpoint routing and responses
- Error scenarios (timeouts, broken pipes)
- Threading and synchronization
- Tool caching behavior
- Node.js subprocess management
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from unittest.mock import MagicMock, Mock, call, patch

import pytest

import aura_cli.mcp_swarm_bridge as swarm_bridge_module


# =============================================================================
# Test Harness for HTTP Handler
# =============================================================================


class _TestableBridgeHandler(swarm_bridge_module.BridgeHandler):
    """BridgeHandler test harness that avoids real sockets."""

    def __init__(self, method: str, path: str, body: dict | None = None):
        self.command = method
        self.path = path
        encoded = json.dumps(body or {}).encode("utf-8")
        self.rfile = io.BytesIO(encoded)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(encoded))}
        self.responses: list[tuple[int, dict]] = []

    def send_response(self, code, message=None):
        self._status = code

    def send_header(self, key, value):
        return

    def end_headers(self):
        return

    def log_message(self, fmt, *args):
        return

    def _send_json(self, code: int, data):
        self.responses.append((code, data))


def _dispatch(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Dispatch an HTTP request to the handler."""
    handler = _TestableBridgeHandler(method, path, body)
    if method == "GET":
        handler.do_GET()
    elif method == "POST":
        handler.do_POST()
    else:
        raise AssertionError(f"unsupported method {method}")
    assert handler.responses, "handler did not emit a response"
    return handler.responses[-1]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def mock_mcp():
    """Mock MCPProcess with common behaviors."""
    m = MagicMock()
    m.alive.return_value = True
    m.list_tools.return_value = [
        {"name": "swarm_run", "description": "Run a swarm agent"},
        {"name": "swarm_status", "description": "Get agent status"},
    ]
    m.call_tool.return_value = {"result": "swarm tool executed"}
    return m


@pytest.fixture()
def mock_process():
    """Mock subprocess.Popen object."""
    proc = MagicMock()
    proc.pid = 54321
    proc.poll.return_value = None
    proc.stdin = MagicMock()
    proc.stdout = iter([])
    proc.stderr = iter([])
    return proc


# =============================================================================
# MCPProcess Tests: Initialization and Lifecycle
# =============================================================================


class TestMCPProcessInitialization:
    """Test MCPProcess.__init__ and basic state."""

    def test_mcp_process_init_creates_empty_state(self):
        """Verify MCPProcess initializes with correct state."""
        mcp = swarm_bridge_module.MCPProcess()
        assert mcp._proc is None
        assert mcp._req_id == 0
        assert mcp._pending == {}
        assert mcp._results == {}
        assert mcp._tools_cache is None
        assert isinstance(mcp._lock, type(threading.Lock()))

    def test_mcp_process_has_required_methods(self):
        """Verify MCPProcess has all required methods."""
        mcp = swarm_bridge_module.MCPProcess()
        assert hasattr(mcp, "start")
        assert hasattr(mcp, "list_tools")
        assert hasattr(mcp, "call_tool")
        assert hasattr(mcp, "alive")


class TestMCPProcessStart:
    """Test MCPProcess.start() subprocess spawning."""

    @patch("aura_cli.mcp_swarm_bridge.subprocess.Popen")
    @patch("aura_cli.mcp_swarm_bridge.os.environ", {})
    def test_start_spawns_node_process(self, mock_popen):
        """Verify start() spawns node swarm-mcp-server."""
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        with patch.object(swarm_bridge_module.MCPProcess, "_initialize"):
            mcp = swarm_bridge_module.MCPProcess()
            mcp.start()

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert args[0][0] == "node"
        assert "swarm-mcp-server" in args[0][1]
        assert "dist" in args[0][1]
        assert "index.js" in args[0][1]

    @patch("aura_cli.mcp_swarm_bridge.subprocess.Popen")
    def test_start_sets_cwd_to_project_root(self, mock_popen):
        """Verify start() runs from project root."""
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        with patch.object(swarm_bridge_module.MCPProcess, "_initialize"):
            mcp = swarm_bridge_module.MCPProcess()
            mcp.start()

        _, kwargs = mock_popen.call_args
        assert "cwd" in kwargs
        assert "swarm-mcp-server" not in kwargs["cwd"]

    @patch("aura_cli.mcp_swarm_bridge.subprocess.Popen")
    def test_start_configures_stdio_pipes(self, mock_popen):
        """Verify start() configures stdin/stdout/stderr."""
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        with patch.object(swarm_bridge_module.MCPProcess, "_initialize"):
            mcp = swarm_bridge_module.MCPProcess()
            mcp.start()

        _, kwargs = mock_popen.call_args
        assert kwargs["stdin"] == swarm_bridge_module.subprocess.PIPE
        assert kwargs["stdout"] == swarm_bridge_module.subprocess.PIPE
        assert kwargs["stderr"] == swarm_bridge_module.subprocess.PIPE
        assert kwargs["text"] is True

    @patch("aura_cli.mcp_swarm_bridge.subprocess.Popen")
    def test_start_creates_reader_thread(self, mock_popen):
        """Verify start() creates a reader thread."""
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        with patch.object(swarm_bridge_module.MCPProcess, "_initialize"):
            with patch("threading.Thread") as mock_thread:
                mcp = swarm_bridge_module.MCPProcess()
                mcp.start()
                mock_thread.assert_called_once()
                args, kwargs = mock_thread.call_args
                assert kwargs.get("daemon") is True

    @patch("aura_cli.mcp_swarm_bridge.subprocess.Popen")
    def test_start_sets_proc_attribute(self, mock_popen):
        """Verify start() assigns _proc."""
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        with patch.object(swarm_bridge_module.MCPProcess, "_initialize"):
            mcp = swarm_bridge_module.MCPProcess()
            mcp.start()
            assert mcp._proc is mock_proc


# =============================================================================
# MCPProcess Tests: JSON-RPC Communication
# =============================================================================


class TestMCPProcessRPC:
    """Test JSON-RPC 2.0 communication (_rpc, _rpc_notify)."""

    def test_rpc_increments_request_id(self):
        """Verify _rpc increments request ID."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()
        mcp._proc.poll.return_value = None

        event_mock = MagicMock()
        event_mock.wait.return_value = False

        with patch("threading.Event", return_value=event_mock):
            mcp._rpc("test_method", {})
            assert mcp._req_id == 1

    def test_rpc_sends_json_to_stdin(self):
        """Verify _rpc writes JSON-RPC message to stdin."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        stdin_mock = MagicMock()
        mcp._proc.stdin = stdin_mock

        event_mock = MagicMock()
        event_mock.wait.return_value = False

        with patch("threading.Event", return_value=event_mock):
            mcp._rpc("test_method", {"param": "value"})

        stdin_mock.write.assert_called_once()
        written_data = stdin_mock.write.call_args[0][0]
        msg = json.loads(written_data.strip())
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "test_method"
        assert msg["params"] == {"param": "value"}
        assert "id" in msg

    def test_rpc_flushes_stdin(self):
        """Verify _rpc flushes stdin after write."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        stdin_mock = MagicMock()
        mcp._proc.stdin = stdin_mock

        event_mock = MagicMock()
        event_mock.wait.return_value = False

        with patch("threading.Event", return_value=event_mock):
            mcp._rpc("test_method", {})

        stdin_mock.flush.assert_called_once()

    def test_rpc_waits_for_response(self):
        """Verify _rpc waits for response event."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()

        event_mock = MagicMock()
        event_mock.wait.return_value = True

        req_id = mcp._req_id + 1
        mcp._results[req_id] = {"jsonrpc": "2.0", "result": {"data": "test"}, "id": req_id}

        with patch("threading.Event", return_value=event_mock):
            result = mcp._rpc("test_method", {}, timeout=5.0)

        event_mock.wait.assert_called_once_with(5.0)
        assert result == {"jsonrpc": "2.0", "result": {"data": "test"}, "id": req_id}

    def test_rpc_timeout_returns_none(self):
        """Verify _rpc returns None on timeout."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()

        event_mock = MagicMock()
        event_mock.wait.return_value = False

        with patch("threading.Event", return_value=event_mock):
            result = mcp._rpc("test_method", {}, timeout=0.1)

        assert result is None

    def test_rpc_broken_pipe_returns_none(self):
        """Verify _rpc handles BrokenPipeError."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()
        mcp._proc.stdin.write.side_effect = BrokenPipeError("pipe broken")

        result = mcp._rpc("test_method", {})

        assert result is None

    def test_rpc_notify_sends_no_id(self):
        """Verify _rpc_notify sends message without id."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        stdin_mock = MagicMock()
        mcp._proc.stdin = stdin_mock

        mcp._rpc_notify("notifications/initialized", {"foo": "bar"})

        stdin_mock.write.assert_called_once()
        written_data = stdin_mock.write.call_args[0][0]
        msg = json.loads(written_data.strip())
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "notifications/initialized"
        assert "id" not in msg

    def test_rpc_notify_broken_pipe_is_silent(self):
        """Verify _rpc_notify ignores BrokenPipeError."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()
        mcp._proc.stdin.write.side_effect = BrokenPipeError()

        mcp._rpc_notify("test", {})


class TestMCPProcessReader:
    """Test _reader thread and JSON-RPC response parsing."""

    def test_reader_parses_json_response(self):
        """Verify _reader parses JSON-RPC responses."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()

        req_id = 1
        event = threading.Event()
        mcp._pending[req_id] = event

        response_msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"data": "test"}})
        mcp._proc.stdout = iter([response_msg + "\n"])

        mcp._reader()

        assert req_id in mcp._results
        assert mcp._results[req_id]["result"]["data"] == "test"
        assert event.is_set()

    def test_reader_ignores_empty_lines(self):
        """Verify _reader skips empty lines."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdout = iter(["  \n", "\n", ""])

        mcp._reader()

    def test_reader_ignores_malformed_json(self):
        """Verify _reader handles JSONDecodeError gracefully."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdout = iter(["not valid json\n"])

        mcp._reader()

    def test_reader_ignores_response_for_unknown_request_id(self):
        """Verify _reader ignores responses for unknown request IDs."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()

        mcp._pending = {}
        response_msg = json.dumps({"jsonrpc": "2.0", "id": 999, "result": {}})
        mcp._proc.stdout = iter([response_msg + "\n"])

        mcp._reader()

        assert 999 not in mcp._results


# =============================================================================
# MCPProcess Tests: Tool Management
# =============================================================================


class TestMCPProcessTools:
    """Test list_tools() and call_tool() methods."""

    def test_list_tools_calls_rpc(self):
        """Verify list_tools calls _rpc with tools/list."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value={"result": {"tools": [{"name": "test"}]}})

        tools = mcp.list_tools()

        mcp._rpc.assert_called_once_with("tools/list", {})
        assert len(tools) == 1
        assert tools[0]["name"] == "test"

    def test_list_tools_caches_result(self):
        """Verify list_tools caches the result."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value={"result": {"tools": [{"name": "cached"}]}})

        tools1 = mcp.list_tools()
        tools2 = mcp.list_tools()

        mcp._rpc.assert_called_once()
        assert tools1 is tools2
        assert tools1[0]["name"] == "cached"

    def test_list_tools_returns_empty_on_rpc_failure(self):
        """Verify list_tools returns [] on RPC failure."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value=None)

        tools = mcp.list_tools()

        assert tools == []

    def test_list_tools_returns_empty_on_missing_result(self):
        """Verify list_tools returns [] if result key is missing."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value={"error": "something"})

        tools = mcp.list_tools()

        assert tools == []

    def test_call_tool_sends_name_and_arguments(self):
        """Verify call_tool sends tool_name and args to _rpc."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value={"result": {"status": "success"}})

        result = mcp.call_tool("swarm_run", {"agent_id": "123"})

        mcp._rpc.assert_called_once_with("tools/call", {"name": "swarm_run", "arguments": {"agent_id": "123"}})

    def test_call_tool_returns_error_on_timeout(self):
        """Verify call_tool returns error dict on timeout."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value=None)

        result = mcp.call_tool("test", {})

        assert result == {"error": "timeout or process failure"}

    def test_call_tool_returns_rpc_error(self):
        """Verify call_tool returns RPC error response."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value={"error": {"code": -1, "message": "Invalid tool"}})

        result = mcp.call_tool("nonexistent", {})

        assert "error" in result

    def test_call_tool_returns_result_directly(self):
        """Verify call_tool returns result as-is (no flattening like GitHub bridge)."""
        mcp = swarm_bridge_module.MCPProcess()
        response_result = {"status": "running", "output": [1, 2, 3]}
        mcp._rpc = MagicMock(return_value={"result": response_result})

        result = mcp.call_tool("test", {})

        assert result == response_result

    def test_alive_returns_true_when_process_running(self):
        """Verify alive() returns True when process is running."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.poll.return_value = None

        assert mcp.alive() is True

    def test_alive_returns_false_when_process_terminated(self):
        """Verify alive() returns False when process exits."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.poll.return_value = 0

        assert mcp.alive() is False

    def test_alive_returns_false_when_no_process(self):
        """Verify alive() returns False when _proc is None."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = None

        assert mcp.alive() is False


# =============================================================================
# MCPProcess Tests: Initialization Handshake
# =============================================================================


class TestMCPProcessInitialize:
    """Test _initialize() MCP handshake."""

    def test_initialize_sends_initialize_method(self):
        """Verify _initialize sends MCP initialize request."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()
        mcp._rpc = MagicMock(return_value={"result": {"serverInfo": {}}})

        mcp._initialize()

        mcp._rpc.assert_called_once()
        args = mcp._rpc.call_args[0]
        assert args[0] == "initialize"
        params = args[1]
        assert params["protocolVersion"] == "2024-11-05"
        assert params["clientInfo"]["name"] == "aura-mcp-swarm-bridge"

    def test_initialize_sends_notification_on_success(self):
        """Verify _initialize sends initialized notification."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value={"result": {"serverInfo": {}}})
        mcp._rpc_notify = MagicMock()

        mcp._initialize()

        mcp._rpc_notify.assert_called_once_with("notifications/initialized", {})

    def test_initialize_handles_none_response(self):
        """Verify _initialize handles RPC timeout."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value=None)
        mcp._rpc_notify = MagicMock()

        mcp._initialize()

        mcp._rpc_notify.assert_not_called()


# =============================================================================
# HTTP Handler Tests: GET Endpoints
# =============================================================================


class TestBridgeHandlerGetEndpoints:
    """Test GET /health, /tools endpoints."""

    def test_get_health_endpoint(self, mock_mcp):
        """Verify GET /health returns status and mcp_alive."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/health")

        assert status == 200
        assert body["status"] == "ok"
        assert body["mcp_alive"] is True

    def test_get_health_shows_mcp_dead(self, mock_mcp):
        """Verify GET /health reports when MCP is dead."""
        mock_mcp.alive.return_value = False
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/health")

        assert status == 200
        assert body["mcp_alive"] is False

    def test_get_tools_endpoint(self, mock_mcp):
        """Verify GET /tools returns tool list."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/tools")

        assert status == 200
        assert isinstance(body["tools"], list)
        assert len(body["tools"]) == 2

    def test_get_tools_empty_list(self, mock_mcp):
        """Verify GET /tools handles empty tool list."""
        mock_mcp.list_tools.return_value = []
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/tools")

        assert status == 200
        assert body["tools"] == []

    def test_get_unknown_endpoint_404(self, mock_mcp):
        """Verify GET unknown path returns 404."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/unknown")

        assert status == 404
        assert "error" in body

    def test_get_root_404(self, mock_mcp):
        """Verify GET / returns 404."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/")

        assert status == 404

    def test_get_metrics_returns_404(self, mock_mcp):
        """Verify swarm bridge does NOT have /metrics endpoint (unlike github bridge)."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/metrics")

        assert status == 404


# =============================================================================
# HTTP Handler Tests: POST Endpoints
# =============================================================================


class TestBridgeHandlerPostEndpoints:
    """Test POST /call endpoint."""

    def test_post_call_with_tool_name(self, mock_mcp):
        """Verify POST /call executes tool."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "swarm_run", "args": {"agent": "a1"}})

        assert status == 200
        assert "data" in body
        mock_mcp.call_tool.assert_called_once_with("swarm_run", {"agent": "a1"})

    def test_post_call_with_name_field(self, mock_mcp):
        """Verify POST /call accepts 'name' field."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"name": "swarm_run", "args": {}})

        assert status == 200
        mock_mcp.call_tool.assert_called_once_with("swarm_run", {})

    def test_post_call_with_arguments_field(self, mock_mcp):
        """Verify POST /call accepts 'arguments' field."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "swarm_run", "arguments": {"id": "xyz"}})

        assert status == 200
        mock_mcp.call_tool.assert_called_once_with("swarm_run", {"id": "xyz"})

    def test_post_call_missing_tool_name_400(self, mock_mcp):
        """Verify POST /call without tool_name returns 400."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"args": {}})

        assert status == 400
        assert "error" in body

    def test_post_call_empty_tool_name_400(self, mock_mcp):
        """Verify POST /call with empty tool_name returns 400."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "", "args": {}})

        assert status == 400

    def test_post_call_no_args_defaults_to_empty(self, mock_mcp):
        """Verify POST /call without args defaults to {}."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "swarm_run"})

        assert status == 200
        mock_mcp.call_tool.assert_called_once_with("swarm_run", {})

    def test_post_unknown_endpoint_404(self, mock_mcp):
        """Verify POST unknown path returns 404."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/unknown", {})

        assert status == 404

    def test_post_call_returns_tool_error(self, mock_mcp):
        """Verify POST /call returns tool errors wrapped in 'data'."""
        mock_mcp.call_tool.return_value = {"error": "tool not found"}
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "missing", "args": {}})

        assert status == 200
        assert "data" in body
        assert body["data"]["error"] == "tool not found"

    def test_post_call_wraps_result_in_data_field(self, mock_mcp):
        """Verify POST /call wraps result in 'data' field (swarm-specific behavior)."""
        mock_mcp.call_tool.return_value = {"output": "test output", "status": "done"}
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "test", "args": {}})

        assert status == 200
        assert "data" in body
        assert body["data"]["output"] == "test output"


# =============================================================================
# HTTP Handler Tests: Header and Content Handling
# =============================================================================


class TestBridgeHandlerHeaderHandling:
    """Test HTTP header and body parsing."""

    def test_read_body_with_content_length(self):
        """Verify _read_body parses Content-Length header."""
        handler = _TestableBridgeHandler("POST", "/call", {"test": "data"})
        body = handler._read_body()

        assert body["test"] == "data"

    def test_read_body_missing_content_length(self):
        """Verify _read_body handles missing Content-Length."""
        handler = _TestableBridgeHandler("POST", "/call")
        handler.headers = {}
        body = handler._read_body()

        assert body == {}

    def test_send_json_response_code(self, mock_mcp):
        """Verify _send_json sets correct status code."""
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/health")

        assert status == 200


# =============================================================================
# Module-level Globals and Configuration
# =============================================================================


class TestModuleGlobals:
    """Test module-level configuration."""

    def test_port_from_env(self):
        """Verify PORT respects MCP_SERVER_PORT env var."""
        with patch.dict(os.environ, {"MCP_SERVER_PORT": "9050"}):
            import importlib

            mod = importlib.reload(swarm_bridge_module)
            assert mod.PORT == 9050

    def test_port_default_8050(self):
        """Verify PORT defaults to 8050."""
        env_copy = dict(os.environ)
        env_copy.pop("MCP_SERVER_PORT", None)
        with patch.dict(os.environ, env_copy, clear=True):
            import importlib

            mod = importlib.reload(swarm_bridge_module)
            assert mod.PORT == 8050

    def test_global_mcp_instance_exists(self):
        """Verify module has global _mcp instance."""
        assert hasattr(swarm_bridge_module, "_mcp")
        assert isinstance(swarm_bridge_module._mcp, swarm_bridge_module.MCPProcess)

    def test_project_root_computation(self):
        """Verify PROJECT_ROOT is computed correctly."""
        assert hasattr(swarm_bridge_module, "PROJECT_ROOT")
        assert swarm_bridge_module.PROJECT_ROOT.endswith("aura-cli") or "aura" in swarm_bridge_module.PROJECT_ROOT


# =============================================================================
# Threading and Concurrency Tests
# =============================================================================


class TestThreading:
    """Test threading behavior and thread safety."""

    def test_rpc_lock_protects_request_id(self):
        """Verify _lock protects concurrent _req_id access."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()

        event_mock = MagicMock()
        event_mock.wait.return_value = False

        with patch("threading.Event", return_value=event_mock):
            mcp._rpc("method1", {})
            id1 = mcp._req_id

            mcp._rpc("method2", {})
            id2 = mcp._req_id

            assert id2 > id1

    def test_reader_is_daemon_thread(self):
        """Verify reader thread is created as daemon."""
        with patch("aura_cli.mcp_swarm_bridge.subprocess.Popen"):
            with patch.object(swarm_bridge_module.MCPProcess, "_initialize"):
                with patch("threading.Thread") as mock_thread:
                    mcp = swarm_bridge_module.MCPProcess()
                    mcp._proc = MagicMock()
                    mcp._proc.stdout = iter([])

                    t = threading.Thread(target=mcp._reader, daemon=True)
                    assert t.daemon


# =============================================================================
# Error Handling and Edge Cases
# =============================================================================


class TestErrorHandling:
    """Test error scenarios and edge cases."""

    def test_call_tool_with_complex_result(self):
        """Verify call_tool returns complex results as-is."""
        mcp = swarm_bridge_module.MCPProcess()
        complex_result = {
            "agent_id": "abc123",
            "status": "completed",
            "iterations": 5,
            "messages": [{"role": "user", "content": "hello"}],
        }
        mcp._rpc = MagicMock(return_value={"result": complex_result})

        result = mcp.call_tool("test", {})

        assert result == complex_result

    def test_list_tools_missing_tools_key(self):
        """Verify list_tools handles result without 'tools' key."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._rpc = MagicMock(return_value={"result": {}})

        tools = mcp.list_tools()

        assert tools == []

    def test_rpc_clears_pending_on_timeout(self):
        """Verify _rpc cleans up _pending on timeout."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()

        event_mock = MagicMock()
        event_mock.wait.return_value = False

        with patch("threading.Event", return_value=event_mock):
            mcp._rpc("test", {}, timeout=0.1)

        assert len(mcp._pending) == 0

    def test_read_body_with_zero_content_length(self):
        """Verify _read_body handles Content-Length: 0."""
        handler = _TestableBridgeHandler("POST", "/call")
        handler.headers = {"Content-Length": "0"}
        body = handler._read_body()

        assert body == {}

    def test_rpc_clears_results_after_retrieval(self):
        """Verify _rpc removes result after returning."""
        mcp = swarm_bridge_module.MCPProcess()
        mcp._proc = MagicMock()
        mcp._proc.stdin = MagicMock()

        event_mock = MagicMock()
        event_mock.wait.return_value = True

        req_id = mcp._req_id + 1
        mcp._results[req_id] = {"result": "data"}

        with patch("threading.Event", return_value=event_mock):
            mcp._rpc("test", {})

        assert req_id not in mcp._results


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_handler_with_valid_tool_call(self, mock_mcp):
        """Verify full flow: POST /call -> handler -> mock_mcp."""
        mock_mcp.call_tool.return_value = {"status": "success", "result": "done"}
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "test", "args": {"x": 1}})

        assert status == 200
        assert body["data"]["status"] == "success"
        mock_mcp.call_tool.assert_called_once_with("test", {"x": 1})

    def test_handler_mcp_dead_health_check(self, mock_mcp):
        """Verify health endpoint reports dead MCP."""
        mock_mcp.alive.return_value = False
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("GET", "/health")

        assert status == 200
        assert body["mcp_alive"] is False

    def test_post_call_mcp_error_propagates(self, mock_mcp):
        """Verify POST /call returns MCP errors wrapped in 'data'."""
        mock_mcp.call_tool.return_value = {"error": "MCP server error"}
        with patch.object(swarm_bridge_module, "_mcp", mock_mcp):
            status, body = _dispatch("POST", "/call", {"tool_name": "fail", "args": {}})

        assert status == 200
        assert body["data"]["error"] == "MCP server error"


# =============================================================================
# Module Import and Basic Structure Tests
# =============================================================================


class TestModuleStructure:
    """Test module imports and basic structure."""

    def test_module_imports_cleanly(self):
        """Verify module can be imported."""
        import importlib

        mod = importlib.import_module("aura_cli.mcp_swarm_bridge")
        assert mod is not None

    def test_has_bridge_handler_class(self):
        """Verify BridgeHandler exists."""
        assert hasattr(swarm_bridge_module, "BridgeHandler")
        assert hasattr(swarm_bridge_module.BridgeHandler, "do_GET")
        assert hasattr(swarm_bridge_module.BridgeHandler, "do_POST")

    def test_has_mcp_process_class(self):
        """Verify MCPProcess exists."""
        assert hasattr(swarm_bridge_module, "MCPProcess")

    def test_has_main_function(self):
        """Verify main() exists."""
        assert hasattr(swarm_bridge_module, "main")
        assert callable(swarm_bridge_module.main)


# =============================================================================
# Swarm-specific Behavior Tests
# =============================================================================


class TestSwarmSpecificBehavior:
    """Test behaviors specific to swarm bridge."""

    def test_no_github_token_check(self):
        """Verify swarm bridge doesn't check for GITHUB_PERSONAL_ACCESS_TOKEN."""
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.stdout = iter([])
        with patch("aura_cli.mcp_swarm_bridge.subprocess.Popen", return_value=mock_proc):
            with patch.object(swarm_bridge_module.MCPProcess, "_initialize"):
                mcp = swarm_bridge_module.MCPProcess()
                mcp.start()
                assert mcp._proc is not None

    def test_node_server_path_construction(self):
        """Verify swarm server path is constructed correctly."""
        project_root = swarm_bridge_module.PROJECT_ROOT
        server_path = os.path.join(project_root, "swarm-mcp-server", "dist", "index.js")
        assert "swarm-mcp-server" in server_path
        assert "dist" in server_path
        assert "index.js" in server_path
