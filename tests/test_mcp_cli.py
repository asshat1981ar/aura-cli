from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import aura_cli.mcp_cli as mcp_cli


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, config_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["AURA_MCP_CONFIG"] = str(config_path)
    return subprocess.run(
        [sys.executable, "-m", "aura_cli.mcp_cli", *args],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _build_stdio_server_script(tmpdir: Path) -> Path:
    script_path = tmpdir / "fake_stdio_mcp.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)
                req_id = message.get("id")
                method = message.get("method")
                if req_id is None:
                    continue
                if method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "serverInfo": {"name": "fake-stdio", "version": "1.0"},
                            "capabilities": {},
                        },
                    }
                elif method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": "Echo back text.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                        "required": ["text"],
                                    },
                                }
                            ]
                        },
                    }
                elif method == "tools/call":
                    arguments = message["params"].get("arguments", {})
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {"type": "text", "text": f"Echo: {arguments.get('text', '')}"}
                            ]
                        },
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Unknown method: {method}"},
                    }
                sys.stdout.write(json.dumps(response) + "\\n")
                sys.stdout.flush()
            """
        ),
        encoding="utf-8",
    )
    return script_path


def test_cli_lists_servers_and_tool_names(monkeypatch, capsys) -> None:
    fake_specs = {
        "stdio-server": mcp_cli.MCPServerSpec(name="stdio-server", config={"type": "stdio"}),
        "http-server": mcp_cli.MCPServerSpec(name="http-server", config={"type": "http"}),
    }

    class _FakeClient:
        def __init__(self, spec: mcp_cli.MCPServerSpec):
            self.spec = spec

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def list_tools(self) -> list[dict]:
            if self.spec.name == "stdio-server":
                return [{"name": "echo", "description": "Echo back text."}]
            return [{"name": "status", "description": "Return service status."}]

    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _path: ROOT / ".mcp.json")
    monkeypatch.setattr(mcp_cli, "_load_server_specs", lambda _path: fake_specs)
    monkeypatch.setattr(mcp_cli, "_create_client", lambda spec: _FakeClient(spec))

    result = mcp_cli.main([])

    captured = capsys.readouterr()
    assert result == 0
    assert "stdio-server/echo" in captured.out
    assert "http-server/status" in captured.out


def test_cli_can_show_tool_schema_and_call_stdio_tool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        script_path = _build_stdio_server_script(tmpdir)
        config_path = tmpdir / "mcp.json"
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "stdio-server": {
                            "type": "stdio",
                            "command": sys.executable,
                            "args": [str(script_path)],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        schema_result = _run_cli("stdio-server/echo", "--json", config_path=config_path)
        call_result = _run_cli("stdio-server/echo", '{"text": "hello"}', "--raw", config_path=config_path)

        assert schema_result.returncode == 0, schema_result.stderr
        schema = json.loads(schema_result.stdout)
        assert schema["name"] == "echo"
        assert "text" in schema["inputSchema"]["properties"]

        assert call_result.returncode == 0, call_result.stderr
        assert call_result.stdout.strip() == "Echo: hello"


def test_cli_can_grep_tools() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        script_path = _build_stdio_server_script(tmpdir)
        config_path = tmpdir / "mcp.json"
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "stdio-server": {
                            "type": "stdio",
                            "command": sys.executable,
                            "args": [str(script_path)],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        result = _run_cli("grep", "*echo*", config_path=config_path)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "stdio-server/echo"


def test_http_client_aura_routes_without_network(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="http-server", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)

    def fake_request(method: str, path: str, payload: dict | None = None, timeout: float | None = None):
        if method == "GET" and path == "/tools":
            return {
                "tools": [
                    {
                        "name": "status",
                        "description": "Return service status.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        if method == "GET" and path == "/tool/status":
            return {
                "name": "status",
                "description": "Return service status.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        if method == "POST" and path == "/call":
            return {"result": f"called {payload['tool_name']}"}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(client, "_request_aura", fake_request)

    assert client.list_tools()[0]["name"] == "status"
    assert client.get_tool("status")["name"] == "status"
    assert client.call_tool("status", {})["result"] == "called status"


def test_cli_listing_times_out_hanging_stdio_server(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        script_path = tmpdir / "hang_stdio_mcp.py"
        script_path.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        config_path = tmpdir / "mcp.json"
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "hang-server": {
                            "type": "stdio",
                            "command": sys.executable,
                            "args": [str(script_path)],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("AURA_MCP_LIST_TIMEOUT_SECONDS", "0.2")
        monkeypatch.setenv("AURA_MCP_INIT_TIMEOUT_SECONDS", "0.2")

        started_at = time.monotonic()
        rows = mcp_cli._list_servers(mcp_cli._load_server_specs(config_path), include_description=False)
        elapsed = time.monotonic() - started_at

        assert elapsed < 2
        assert rows[0]["server"] == "hang-server"
        assert rows[0]["tools"] == []
        assert "Timed out waiting for 'initialize'" in rows[0]["error"]


# ---------------------------------------------------------------------------
# Pure unit tests — no subprocesses, no network
# ---------------------------------------------------------------------------

# ── MCPServerSpec ──────────────────────────────────────────────────────────


def test_mcp_server_spec_default_transport() -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={})
    assert spec.transport == "stdio"


def test_mcp_server_spec_explicit_transport() -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "http"})
    assert spec.transport == "http"


# ── _expand_env_vars ───────────────────────────────────────────────────────


def test_expand_env_vars_string(monkeypatch) -> None:
    monkeypatch.setenv("MY_VAR", "hello")
    assert mcp_cli._expand_env_vars("${MY_VAR}") == "hello"


def test_expand_env_vars_env_prefix(monkeypatch) -> None:
    monkeypatch.setenv("MY_VAR", "world")
    assert mcp_cli._expand_env_vars("${env:MY_VAR}") == "world"


def test_expand_env_vars_missing_var(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert mcp_cli._expand_env_vars("${MISSING_VAR}") == ""


def test_expand_env_vars_list(monkeypatch) -> None:
    monkeypatch.setenv("A", "1")
    result = mcp_cli._expand_env_vars(["${A}", "literal"])
    assert result == ["1", "literal"]


def test_expand_env_vars_dict(monkeypatch) -> None:
    monkeypatch.setenv("B", "2")
    result = mcp_cli._expand_env_vars({"key": "${B}"})
    assert result == {"key": "2"}


def test_expand_env_vars_non_string_passthrough() -> None:
    assert mcp_cli._expand_env_vars(42) == 42
    assert mcp_cli._expand_env_vars(None) is None


# ── _resolve_timeout ───────────────────────────────────────────────────────


def test_resolve_timeout_uses_env_var(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TIMEOUT", "7.5")
    result = mcp_cli._resolve_timeout({}, "TEST_TIMEOUT", "timeout_seconds", 3.0)
    assert result == 7.5


def test_resolve_timeout_uses_config_key(monkeypatch) -> None:
    monkeypatch.delenv("TEST_TIMEOUT", raising=False)
    result = mcp_cli._resolve_timeout({"timeout_seconds": 12.0}, "TEST_TIMEOUT", "timeout_seconds", 3.0)
    assert result == 12.0


def test_resolve_timeout_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("TEST_TIMEOUT", raising=False)
    result = mcp_cli._resolve_timeout({}, "TEST_TIMEOUT", "timeout_seconds", 3.0)
    assert result == 3.0


def test_resolve_timeout_invalid_value_returns_default(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TIMEOUT", "not-a-number")
    result = mcp_cli._resolve_timeout({}, "TEST_TIMEOUT", "timeout_seconds", 5.0)
    assert result == 5.0


def test_resolve_timeout_non_positive_returns_default(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TIMEOUT", "0")
    result = mcp_cli._resolve_timeout({}, "TEST_TIMEOUT", "timeout_seconds", 5.0)
    assert result == 5.0


# ── _load_config / _load_server_specs ─────────────────────────────────────


def test_load_config_file_not_found(tmp_path) -> None:
    with pytest.raises(mcp_cli.MCPCLIError, match="MCP config not found"):
        mcp_cli._load_config(tmp_path / "nonexistent.json")


def test_load_server_specs_valid(tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"s1": {"type": "stdio", "command": "echo"}}}))
    specs = mcp_cli._load_server_specs(cfg)
    assert "s1" in specs
    assert specs["s1"].transport == "stdio"


def test_load_server_specs_invalid_mcp_servers(tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": "not-a-dict"}))
    with pytest.raises(mcp_cli.MCPCLIError, match="Invalid MCP config"):
        mcp_cli._load_server_specs(cfg)


def test_load_server_specs_empty_mcp_servers(tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({}))
    specs = mcp_cli._load_server_specs(cfg)
    assert specs == {}


# ── _resolve_config_path ──────────────────────────────────────────────────


def test_resolve_config_path_explicit(tmp_path) -> None:
    cfg = tmp_path / "my.json"
    cfg.write_text("{}")
    result = mcp_cli._resolve_config_path(str(cfg))
    assert result == cfg.resolve()


def test_resolve_config_path_env_var(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "env.json"
    cfg.write_text("{}")
    monkeypatch.setenv("AURA_MCP_CONFIG", str(cfg))
    result = mcp_cli._resolve_config_path(None)
    assert result == cfg.resolve()


# ── _split_target ─────────────────────────────────────────────────────────


def test_split_target_server_only() -> None:
    server, tool = mcp_cli._split_target("myserver")
    assert server == "myserver"
    assert tool is None


def test_split_target_server_and_tool() -> None:
    server, tool = mcp_cli._split_target("myserver/mytool")
    assert server == "myserver"
    assert tool == "mytool"


def test_split_target_tool_with_slash_in_name() -> None:
    server, tool = mcp_cli._split_target("s/a/b")
    assert server == "s"
    assert tool == "a/b"


# ── _format_tool_line ─────────────────────────────────────────────────────


def test_format_tool_line_with_description() -> None:
    tool = {"name": "echo", "description": "Echoes text."}
    line = mcp_cli._format_tool_line("srv", tool, include_description=True)
    assert line == "srv/echo - Echoes text."


def test_format_tool_line_without_description() -> None:
    tool = {"name": "echo", "description": "Echoes text."}
    line = mcp_cli._format_tool_line("srv", tool, include_description=False)
    assert line == "srv/echo"


def test_format_tool_line_missing_name() -> None:
    tool = {}
    line = mcp_cli._format_tool_line("srv", tool, include_description=False)
    assert line == "srv/<unknown>"


# ── _extract_raw_text ─────────────────────────────────────────────────────


def test_extract_raw_text_string() -> None:
    assert mcp_cli._extract_raw_text("hello") == "hello"


def test_extract_raw_text_dict_content_list() -> None:
    result = {"content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]}
    assert mcp_cli._extract_raw_text(result) == "hi\nthere"


def test_extract_raw_text_dict_skips_non_text_blocks() -> None:
    result = {"content": [{"type": "image", "text": "skip"}, {"type": "text", "text": "keep"}]}
    assert mcp_cli._extract_raw_text(result) == "keep"


def test_extract_raw_text_dict_result_key() -> None:
    assert mcp_cli._extract_raw_text({"result": "done"}) == "done"


def test_extract_raw_text_returns_none_for_unknown() -> None:
    assert mcp_cli._extract_raw_text({"other": "data"}) is None
    assert mcp_cli._extract_raw_text(123) is None


# ── _create_client ────────────────────────────────────────────────────────


def test_create_client_returns_http_for_http_type() -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://localhost"})
    client = mcp_cli._create_client(spec)
    assert isinstance(client, mcp_cli.HttpMCPClient)


def test_create_client_returns_stdio_by_default() -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    client = mcp_cli._create_client(spec)
    assert isinstance(client, mcp_cli.StdioMCPClient)


# ── HttpMCPClient — constructor ──────────────────────────────────────────


def test_http_client_raises_without_url() -> None:
    spec = mcp_cli.MCPServerSpec(name="no-url", config={"type": "http"})
    with pytest.raises(mcp_cli.MCPCLIError, match="missing a url"):
        mcp_cli.HttpMCPClient(spec)


def test_http_client_context_manager_noop() -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)
    with client as c:
        assert c is client


# ── HttpMCPClient — _request error paths ────────────────────────────────


def test_http_client_request_http_error(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)

    import io

    err = urllib.error.HTTPError(
        url="http://example.test/tools",
        code=500,
        msg="Internal Server Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"error": "oops"}'),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(mcp_cli.MCPCLIError, match="HTTP 500"):
            client._request("GET", "http://example.test/tools")


def test_http_client_request_url_error(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)

    err = urllib.error.URLError(reason="connection refused")
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(mcp_cli.MCPCLIError, match="Failed to reach server"):
            client._request("GET", "http://example.test/tools")


# ── HttpMCPClient — list_tools cache ────────────────────────────────────


def test_http_client_list_tools_uses_cache() -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)
    client._tools_cache = [{"name": "cached-tool"}]
    result = client.list_tools()
    assert result[0]["name"] == "cached-tool"


# ── HttpMCPClient — list_tools jsonrpc fallback ──────────────────────────


def test_http_client_list_tools_falls_back_to_jsonrpc(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)

    def fake_aura(*_a, **_kw):
        raise mcp_cli.MCPCLIError("no aura route")

    jsonrpc_response = {"result": {"tools": [{"name": "j-tool"}]}}

    monkeypatch.setattr(client, "_aura_tool_list", fake_aura)
    monkeypatch.setattr(client, "_request_jsonrpc", lambda *_a, **_kw: jsonrpc_response)

    tools = client.list_tools()
    assert tools[0]["name"] == "j-tool"


# ── HttpMCPClient — get_tool fallback to list ────────────────────────────


def test_http_client_get_tool_falls_back_to_list(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)

    def fake_aura_request(method, path, payload=None, timeout=None):
        raise mcp_cli.MCPCLIError("404")

    monkeypatch.setattr(client, "_request_aura", fake_aura_request)
    client._tools_cache = [{"name": "fb-tool"}]

    tool = client.get_tool("fb-tool")
    assert tool["name"] == "fb-tool"


def test_http_client_get_tool_not_found(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)

    def fake_aura_request(method, path, payload=None, timeout=None):
        raise mcp_cli.MCPCLIError("404")

    monkeypatch.setattr(client, "_request_aura", fake_aura_request)
    client._tools_cache = []

    with pytest.raises(mcp_cli.MCPCLIError, match="not found"):
        client.get_tool("missing")


# ── HttpMCPClient — call_tool jsonrpc fallback ───────────────────────────


def test_http_client_call_tool_falls_back_to_jsonrpc(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="h", config={"type": "http", "url": "http://example.test"})
    client = mcp_cli.HttpMCPClient(spec)

    def fake_aura(*_a, **_kw):
        raise mcp_cli.MCPCLIError("no aura")

    jsonrpc_response = {"result": {"output": "done"}}
    monkeypatch.setattr(client, "_request_aura", fake_aura)
    monkeypatch.setattr(client, "_request_jsonrpc", lambda *_a, **_kw: jsonrpc_response)

    result = client.call_tool("t", {})
    assert result == {"output": "done"}


# ── StdioMCPClient — start without command ───────────────────────────────


def test_stdio_client_start_raises_without_command() -> None:
    spec = mcp_cli.MCPServerSpec(name="no-cmd", config={"type": "stdio"})
    client = mcp_cli.StdioMCPClient(spec)
    with pytest.raises(mcp_cli.MCPCLIError, match="missing a command"):
        client.start()


def test_stdio_client_start_raises_on_popen_failure() -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "nonexistent-cmd-xyz"})
    client = mcp_cli.StdioMCPClient(spec)
    with patch("subprocess.Popen", side_effect=OSError("not found")):
        with pytest.raises(mcp_cli.MCPCLIError, match="Failed to start server"):
            client.start()


def test_stdio_client_start_is_idempotent() -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    client = mcp_cli.StdioMCPClient(spec)
    mock_proc = MagicMock()
    client._proc = mock_proc
    client.start()  # should not call Popen again
    assert client._proc is mock_proc


# ── StdioMCPClient — _rpc not running ────────────────────────────────────


def test_stdio_client_rpc_raises_when_not_started() -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    client = mcp_cli.StdioMCPClient(spec)
    with pytest.raises(mcp_cli.MCPCLIError, match="is not running"):
        client._rpc("ping", {})


# ── StdioMCPClient — list_tools cache ────────────────────────────────────


def test_stdio_client_list_tools_uses_cache(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    client = mcp_cli.StdioMCPClient(spec)
    client._tools_cache = [{"name": "cached"}]
    assert client.list_tools()[0]["name"] == "cached"


def test_stdio_client_list_tools_error_in_response(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    client = mcp_cli.StdioMCPClient(spec)
    monkeypatch.setattr(client, "_rpc", lambda *_a, **_kw: {"error": "bad"})
    with pytest.raises(mcp_cli.MCPCLIError, match="tools/list failed"):
        client.list_tools()


def test_stdio_client_get_tool_not_found(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    client = mcp_cli.StdioMCPClient(spec)
    client._tools_cache = [{"name": "other"}]
    with pytest.raises(mcp_cli.MCPCLIError, match="not found"):
        client.get_tool("missing")


def test_stdio_client_call_tool_error_in_response(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    client = mcp_cli.StdioMCPClient(spec)
    monkeypatch.setattr(client, "_rpc", lambda *_a, **_kw: {"error": {"code": -1, "message": "fail"}})
    with pytest.raises(mcp_cli.MCPCLIError, match="tools/call failed"):
        client.call_tool("t", {})


# ── _handle_grep ──────────────────────────────────────────────────────────


def test_handle_grep_no_match(capsys, monkeypatch) -> None:
    specs = {"srv": mcp_cli.MCPServerSpec(name="srv", config={"type": "stdio"})}
    rows = [{"server": "srv", "transport": "stdio", "tools": [{"name": "echo", "description": ""}]}]
    monkeypatch.setattr(mcp_cli, "_list_servers", lambda *_a, **_kw: rows)

    args = argparse.Namespace(json=False, descriptions=False)
    rc = mcp_cli._handle_grep("*nomatch*", specs, args)
    assert rc == 1


def test_handle_grep_matches_server_name(capsys, monkeypatch) -> None:
    specs = {"myserver": mcp_cli.MCPServerSpec(name="myserver", config={"type": "stdio"})}
    rows = [{"server": "myserver", "transport": "stdio", "tools": []}]
    monkeypatch.setattr(mcp_cli, "_list_servers", lambda *_a, **_kw: rows)

    args = argparse.Namespace(json=False, descriptions=False)
    rc = mcp_cli._handle_grep("myserver", specs, args)
    captured = capsys.readouterr()
    assert rc == 0
    assert "myserver" in captured.out


def test_handle_grep_json_output(capsys, monkeypatch) -> None:
    specs = {"s": mcp_cli.MCPServerSpec(name="s", config={"type": "stdio"})}
    rows = [{"server": "s", "transport": "stdio", "tools": [{"name": "ping", "description": ""}]}]
    monkeypatch.setattr(mcp_cli, "_list_servers", lambda *_a, **_kw: rows)

    args = argparse.Namespace(json=True, descriptions=False)
    rc = mcp_cli._handle_grep("s/ping", specs, args)
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["pattern"] == "s/ping"


# ── _handle_server ────────────────────────────────────────────────────────


def test_handle_server_unknown_server(monkeypatch) -> None:
    args = argparse.Namespace(json=False, raw=False, descriptions=False, payload=None)
    with pytest.raises(mcp_cli.MCPCLIError, match="Unknown MCP server"):
        mcp_cli._handle_server("ghost", {}, args)


# ── _handle_tool ──────────────────────────────────────────────────────────


def test_handle_tool_unknown_server() -> None:
    args = argparse.Namespace(json=False, raw=False, descriptions=False, payload=None)
    with pytest.raises(mcp_cli.MCPCLIError, match="Unknown MCP server"):
        mcp_cli._handle_tool("ghost", "ping", {}, args)


def test_handle_tool_invalid_json_payload(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="s", config={"type": "stdio", "command": "echo"})
    specs = {"s": spec}
    args = argparse.Namespace(json=False, raw=False, descriptions=False, payload="{bad json}")

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(mcp_cli, "_create_client", lambda _spec: _FakeClient())
    with pytest.raises(mcp_cli.MCPCLIError, match="Invalid JSON payload"):
        mcp_cli._handle_tool("s", "ping", specs, args)


# ── _print_default_listing ────────────────────────────────────────────────


def test_print_default_listing_raw_mode(capsys) -> None:
    rows = [{"server": "s", "transport": "stdio", "tools": [{"name": "echo", "description": "hi"}]}]
    mcp_cli._print_default_listing(rows, raw=True, include_description=False)
    out = capsys.readouterr().out
    assert "s/echo" in out


def test_print_default_listing_shows_error(capsys) -> None:
    rows = [{"server": "s", "transport": "stdio", "tools": [], "error": "boom"}]
    mcp_cli._print_default_listing(rows, raw=False, include_description=False)
    out = capsys.readouterr().out
    assert "error: boom" in out


def test_print_default_listing_raw_no_tools_prints_server(capsys) -> None:
    rows = [{"server": "empty-srv", "transport": "stdio", "tools": []}]
    mcp_cli._print_default_listing(rows, raw=True, include_description=False)
    out = capsys.readouterr().out
    assert "empty-srv" in out


# ── main — error paths ────────────────────────────────────────────────────


def test_main_grep_without_pattern(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    rc = mcp_cli.main(["grep"])
    assert rc == 1


def test_main_unknown_server_returns_1(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    rc = mcp_cli.main(["no-such-server"])
    assert rc == 1


def test_main_json_listing_output(monkeypatch, capsys, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    monkeypatch.setattr(mcp_cli, "_list_servers", lambda *_a, **_kw: [])
    rc = mcp_cli.main(["--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "servers" in data


# ============================================================================
# Additional comprehensive tests for high coverage
# ============================================================================


# ── _resolve_config_path: fallback when no files exist ──
def test_resolve_config_path_no_existing_uses_first_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AURA_MCP_CONFIG", raising=False)
    result = mcp_cli._resolve_config_path(None)
    assert str(result).endswith(".mcp.json")


def test_resolve_config_path_env_priority(tmp_path, monkeypatch) -> None:
    env_cfg = tmp_path / "env_cfg.json"
    env_cfg.write_text("{}")
    explicit_cfg = tmp_path / "explicit_cfg.json"
    explicit_cfg.write_text("{}")
    monkeypatch.setenv("AURA_MCP_CONFIG", str(env_cfg))
    result = mcp_cli._resolve_config_path(str(explicit_cfg))
    assert result == explicit_cfg.resolve()


# ── _expand_env_vars: comprehensive tests ──
def test_expand_env_vars_empty_dict(monkeypatch) -> None:
    result = mcp_cli._expand_env_vars({})
    assert result == {}


def test_expand_env_vars_empty_list(monkeypatch) -> None:
    result = mcp_cli._expand_env_vars([])
    assert result == []


def test_expand_env_vars_multiple_vars_in_string(monkeypatch) -> None:
    monkeypatch.setenv("VAR1", "value1")
    monkeypatch.setenv("VAR2", "value2")
    result = mcp_cli._expand_env_vars("${VAR1}_${VAR2}")
    assert result == "value1_value2"


def test_expand_env_vars_with_env_prefix(monkeypatch) -> None:
    monkeypatch.setenv("MYVAR", "myvalue")
    result = mcp_cli._expand_env_vars("${env:MYVAR}")
    assert result == "myvalue"


def test_expand_env_vars_missing_variable(monkeypatch) -> None:
    monkeypatch.delenv("NONEXISTENT", raising=False)
    result = mcp_cli._expand_env_vars("prefix_${NONEXISTENT}_suffix")
    assert result == "prefix__suffix"


def test_expand_env_vars_nested_dict_and_list(monkeypatch) -> None:
    monkeypatch.setenv("VAR", "val")
    result = mcp_cli._expand_env_vars({"a": [{"b": "${VAR}"}]})
    assert result == {"a": [{"b": "val"}]}


def test_expand_env_vars_with_numbers(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "3000")
    result = mcp_cli._expand_env_vars({"url": "http://localhost:${PORT}"})
    assert result == {"url": "http://localhost:3000"}


# ── _load_config: error cases ──
def test_load_config_missing_file(tmp_path) -> None:
    missing_file = tmp_path / "nonexistent.json"
    with pytest.raises(mcp_cli.MCPCLIError, match="not found"):
        mcp_cli._load_config(missing_file)


def test_load_config_valid_complex_json(tmp_path) -> None:
    cfg_file = tmp_path / "config.json"
    data = {"mcpServers": {"s1": {"type": "http", "url": "http://localhost"}}}
    cfg_file.write_text(json.dumps(data))
    result = mcp_cli._load_config(cfg_file)
    assert result == data
    assert "mcpServers" in result


# ── _load_server_specs: edge cases ──
def test_load_server_specs_with_multiple_servers(tmp_path) -> None:
    cfg_file = tmp_path / "config.json"
    data = {
        "mcpServers": {
            "s1": {"type": "stdio", "command": "cmd1"},
            "s2": {"type": "http", "url": "http://localhost:3000"},
        }
    }
    cfg_file.write_text(json.dumps(data))
    result = mcp_cli._load_server_specs(cfg_file)
    assert len(result) == 2
    assert result["s1"].name == "s1"
    assert result["s2"].name == "s2"


def test_load_server_specs_mcpservers_is_not_dict(tmp_path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"mcpServers": "not a dict"}))
    with pytest.raises(mcp_cli.MCPCLIError, match="Invalid MCP config"):
        mcp_cli._load_server_specs(cfg_file)


# ── _json_dump: formatting tests ──
def test_json_dump_indentation_and_sorting(monkeypatch) -> None:
    data = {"z": 1, "a": 2}
    result = mcp_cli._json_dump(data)
    lines = result.split("\n")
    assert "a" in result and "z" in result
    a_idx = next(i for i, l in enumerate(lines) if '"a"' in l)
    z_idx = next(i for i, l in enumerate(lines) if '"z"' in l)
    assert a_idx < z_idx


def test_json_dump_list_of_objects(monkeypatch) -> None:
    data = [{"name": "tool1"}, {"name": "tool2"}]
    result = mcp_cli._json_dump(data)
    parsed = json.loads(result)
    assert len(parsed) == 2


# ── _resolve_timeout: edge cases ──
def test_resolve_timeout_string_float_in_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TIMEOUT", "5.5")
    result = mcp_cli._resolve_timeout({}, "TEST_TIMEOUT", "key", 1.0)
    assert result == 5.5


def test_resolve_timeout_config_dict_has_value(monkeypatch) -> None:
    config = {"timeout_seconds": 7.0}
    result = mcp_cli._resolve_timeout(config, "MISSING", "timeout_seconds", 3.0)
    assert result == 7.0


def test_resolve_timeout_invalid_type_uses_default(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TIMEOUT", "not_a_number")
    result = mcp_cli._resolve_timeout({}, "TEST_TIMEOUT", "key", 5.0)
    assert result == 5.0


def test_resolve_timeout_float_as_string(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TIMEOUT", "3.14159")
    result = mcp_cli._resolve_timeout({}, "TEST_TIMEOUT", "key", 1.0)
    assert abs(result - 3.14159) < 0.0001


# ── StdioMCPClient: additional tests ──
def test_stdio_client_init_with_all_config(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(
        name="test",
        config={
            "command": "test_cmd",
            "args": ["arg1", "arg2"],
            "env": {"CUSTOM": "value"},
            "initialize_timeout_seconds": 10.0,
            "list_timeout_seconds": 15.0,
            "call_timeout_seconds": 25.0,
        },
    )
    client = mcp_cli.StdioMCPClient(spec)
    assert client.command == "test_cmd"
    assert client.args == ["arg1", "arg2"]
    assert "CUSTOM" in client.env
    assert client._initialize_timeout == 10.0
    assert client._list_timeout == 15.0
    assert client._call_timeout == 25.0


def test_stdio_client_env_includes_os_environ(monkeypatch) -> None:
    monkeypatch.setenv("SYSTEM_VAR", "system_value")
    spec = mcp_cli.MCPServerSpec(
        name="test",
        config={"command": "test", "env": {"CUSTOM": "custom_value"}},
    )
    client = mcp_cli.StdioMCPClient(spec)
    assert "SYSTEM_VAR" in client.env
    assert "CUSTOM" in client.env


def test_stdio_client_start_with_popen_success(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"command": "test"})
    client = mcp_cli.StdioMCPClient(spec)
    mock_proc = MagicMock()
    mock_proc.stdout = iter([])
    mock_proc.stderr = iter([])
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)
    with patch.object(client, "_initialize"):
        client.start()
    assert client._proc is not None


def test_stdio_client_close_with_timeout_then_kill(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"command": "test"})
    client = mcp_cli.StdioMCPClient(spec)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.wait.side_effect = subprocess.TimeoutExpired("test", 2)
    client._proc = mock_proc
    client.close()
    mock_proc.kill.assert_called_once()


def test_stdio_client_read_stdout_valid_json(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"command": "test"})
    client = mcp_cli.StdioMCPClient(spec)
    mock_proc = MagicMock()
    client._proc = mock_proc
    client._pending[1] = threading.Event()
    mock_proc.stdout = iter(['{"id": 1, "result": "ok"}\n'])
    client._read_stdout()
    assert 1 in client._responses


def test_stdio_client_read_stdout_empty_lines(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"command": "test"})
    client = mcp_cli.StdioMCPClient(spec)
    mock_proc = MagicMock()
    client._proc = mock_proc
    mock_proc.stdout = iter(["  \n", "\n", ""])
    client._read_stdout()
    assert len(client._responses) == 0


def test_stdio_client_read_stderr_collects_lines(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"command": "test"})
    client = mcp_cli.StdioMCPClient(spec)
    mock_proc = MagicMock()
    client._proc = mock_proc
    mock_proc.stderr = iter(["error line 1\n", "error line 2\n"])
    client._read_stderr()
    assert "error line 1" in client._stderr_lines
    assert "error line 2" in client._stderr_lines


def test_stdio_client_list_tools_returns_cache(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"command": "test"})
    client = mcp_cli.StdioMCPClient(spec)
    cached = [{"name": "tool1"}]
    client._tools_cache = cached
    result = client.list_tools()
    assert result is cached


# ── HttpMCPClient: additional tests ──
def test_http_client_init_strips_slash(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(
        name="test",
        config={"url": "http://localhost:3000/", "headers": {"X-Auth": "token"}},
    )
    client = mcp_cli.HttpMCPClient(spec)
    assert client.base_url == "http://localhost:3000"
    assert client.headers == {"X-Auth": "token"}


def test_http_client_missing_url_raises(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"url": ""})
    with pytest.raises(mcp_cli.MCPCLIError, match="missing a url"):
        mcp_cli.HttpMCPClient(spec)


def test_http_client_request_with_headers(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(
        name="test",
        config={"url": "http://localhost:3000", "headers": {"X-Token": "secret"}},
    )
    client = mcp_cli.HttpMCPClient(spec)
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"result": "ok"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        client._request("GET", "http://localhost:3000/test")
        called_request = mock_urlopen.call_args[0][0]
        # Headers are case-insensitive in HTTP
        assert "X-token" in called_request.headers or "X-Token" in called_request.headers


def test_http_client_request_post_with_payload(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"url": "http://localhost:3000"})
    client = mcp_cli.HttpMCPClient(spec)
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"result": "created"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result = client._request("POST", "http://localhost:3000/test", {"key": "value"})
        assert result == {"result": "created"}
        called_request = mock_urlopen.call_args[0][0]
        assert called_request.data is not None


def test_http_client_request_http_error_with_json(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"url": "http://localhost:3000"})
    client = mcp_cli.HttpMCPClient(spec)
    with patch("urllib.request.urlopen") as mock_urlopen:
        error = urllib.error.HTTPError("http://localhost", 500, "Internal Server Error", {}, None)
        error.read = MagicMock(return_value=b'{"error": "Database connection failed"}')
        mock_urlopen.side_effect = error
        with pytest.raises(mcp_cli.MCPCLIError, match="HTTP 500"):
            client._request("GET", "http://localhost:3000/test")


def test_http_client_list_tools_aura_fallback_to_jsonrpc(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"url": "http://localhost:3000"})
    client = mcp_cli.HttpMCPClient(spec)
    with patch.object(client, "_aura_tool_list", side_effect=mcp_cli.MCPCLIError("Not found")):
        with patch.object(client, "_request_jsonrpc") as mock_jsonrpc:
            mock_jsonrpc.return_value = {"result": {"tools": [{"name": "tool1"}]}}
            result = client.list_tools()
            assert len(result) == 1


def test_http_client_get_tool_aura_endpoint(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"url": "http://localhost:3000"})
    client = mcp_cli.HttpMCPClient(spec)
    with patch.object(client, "_request_aura") as mock_request:
        mock_request.return_value = {"name": "test_tool", "description": "Test"}
        result = client.get_tool("test_tool")
        assert result["name"] == "test_tool"


def test_http_client_call_tool_aura_endpoint(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"url": "http://localhost:3000"})
    client = mcp_cli.HttpMCPClient(spec)
    with patch.object(client, "_request_aura") as mock_request:
        mock_request.return_value = {"result": "success"}
        result = client.call_tool("test_tool", {"arg": "value"})
        assert result == {"result": "success"}


def test_http_client_aura_tool_list_unexpected_response(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"url": "http://localhost:3000"})
    client = mcp_cli.HttpMCPClient(spec)
    with patch.object(client, "_request_aura") as mock_request:
        mock_request.return_value = "unexpected string"
        with pytest.raises(mcp_cli.MCPCLIError, match="Unexpected"):
            client._aura_tool_list()


# ── _create_client ──
def test_create_client_http_transport(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(
        name="test",
        config={"type": "http", "url": "http://localhost:3000"},
    )
    client = mcp_cli._create_client(spec)
    assert isinstance(client, mcp_cli.HttpMCPClient)


def test_create_client_stdio_transport(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"type": "stdio", "command": "test"})
    client = mcp_cli._create_client(spec)
    assert isinstance(client, mcp_cli.StdioMCPClient)


def test_create_client_default_stdio(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="test", config={"command": "test"})
    client = mcp_cli._create_client(spec)
    assert isinstance(client, mcp_cli.StdioMCPClient)


# ── _format_tool_line ──
def test_format_tool_line_all_fields(monkeypatch) -> None:
    tool = {"name": "mytool", "description": "Does something"}
    result = mcp_cli._format_tool_line("myserver", tool, include_description=True)
    assert "myserver/mytool" in result
    assert "Does something" in result


def test_format_tool_line_no_description(monkeypatch) -> None:
    tool = {"name": "mytool", "description": "Does something"}
    result = mcp_cli._format_tool_line("myserver", tool, include_description=False)
    assert result == "myserver/mytool"
    assert "Does something" not in result


def test_format_tool_line_missing_fields(monkeypatch) -> None:
    tool = {}
    result = mcp_cli._format_tool_line("myserver", tool, include_description=True)
    assert "myserver/<unknown>" in result


# ── _list_servers ──
def test_list_servers_with_tools(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="s1", config={"type": "stdio"})
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.list_tools.return_value = [
            {"name": "tool1", "description": "desc1"},
            {"name": "tool2", "description": "desc2"},
        ]
        mock_create.return_value = mock_client
        result = mcp_cli._list_servers({"s1": spec}, include_description=True)
        assert len(result) == 1
        assert len(result[0]["tools"]) == 2


def test_list_servers_error_handling(monkeypatch) -> None:
    spec = mcp_cli.MCPServerSpec(name="s1", config={"type": "stdio"})
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.list_tools.side_effect = mcp_cli.MCPCLIError("Connection error")
        mock_create.return_value = mock_client
        result = mcp_cli._list_servers({"s1": spec}, include_description=False)
        assert len(result) == 1
        assert "error" in result[0]
        assert result[0]["tools"] == []


def test_list_servers_multiple_servers(monkeypatch) -> None:
    spec1 = mcp_cli.MCPServerSpec(name="s1", config={"type": "stdio"})
    spec2 = mcp_cli.MCPServerSpec(name="s2", config={"type": "stdio"})
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.list_tools.return_value = []
        mock_create.return_value = mock_client
        result = mcp_cli._list_servers({"s1": spec1, "s2": spec2}, include_description=False)
        assert len(result) == 2
        assert result[0]["server"] == "s1"
        assert result[1]["server"] == "s2"


# ── _print_default_listing ──
def test_print_default_listing_standard_mode(capsys, monkeypatch) -> None:
    rows = [
        {
            "server": "myserver",
            "transport": "stdio",
            "tools": [{"name": "tool1", "description": "Does stuff"}],
        }
    ]
    mcp_cli._print_default_listing(rows, raw=False, include_description=True)
    out = capsys.readouterr().out
    assert "myserver" in out
    assert "tool1" in out
    assert "Does stuff" in out


def test_print_default_listing_raw_with_description(capsys, monkeypatch) -> None:
    rows = [
        {
            "server": "myserver",
            "transport": "stdio",
            "tools": [{"name": "tool1", "description": "Does stuff"}],
        }
    ]
    mcp_cli._print_default_listing(rows, raw=True, include_description=True)
    out = capsys.readouterr().out
    assert "myserver/tool1" in out
    assert "Does stuff" in out


def test_print_default_listing_no_tools(capsys, monkeypatch) -> None:
    rows = [
        {
            "server": "myserver",
            "transport": "stdio",
            "tools": [],
        }
    ]
    mcp_cli._print_default_listing(rows, raw=False, include_description=False)
    out = capsys.readouterr().out
    assert "no tools" in out


def test_print_default_listing_with_error_raw(capsys, monkeypatch) -> None:
    rows = [
        {
            "server": "myserver",
            "transport": "stdio",
            "error": "Failed to connect",
            "tools": [],
        }
    ]
    mcp_cli._print_default_listing(rows, raw=True, include_description=False)
    out = capsys.readouterr().out
    assert "myserver" in out


# ── _parse_args ──
def test_parse_args_all_flags(monkeypatch) -> None:
    args = mcp_cli._parse_args(["-c", "/path/config.json", "-j", "-r", "-d", "grep", "pattern"])
    assert args.config == "/path/config.json"
    assert args.json is True
    assert args.raw is True
    assert args.descriptions is True
    assert args.target == "grep"
    assert args.payload == "pattern"


def test_parse_args_minimal(monkeypatch) -> None:
    args = mcp_cli._parse_args([])
    assert args.target is None
    assert args.payload is None
    assert args.config is None
    assert args.json is False


# ── _split_target ──
def test_split_target_server_only_no_tool(monkeypatch) -> None:
    server, tool = mcp_cli._split_target("myserver")
    assert server == "myserver"
    assert tool is None


def test_split_target_with_tool(monkeypatch) -> None:
    server, tool = mcp_cli._split_target("myserver/mytool")
    assert server == "myserver"
    assert tool == "mytool"


def test_split_target_multiple_slashes(monkeypatch) -> None:
    server, tool = mcp_cli._split_target("server/path/to/tool")
    assert server == "server"
    assert tool == "path/to/tool"


# ── _extract_raw_text ──
def test_extract_raw_text_string_returns_as_is(monkeypatch) -> None:
    result = mcp_cli._extract_raw_text("raw string result")
    assert result == "raw string result"


def test_extract_raw_text_dict_content_list_multiline(monkeypatch) -> None:
    data = {
        "content": [
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
        ]
    }
    result = mcp_cli._extract_raw_text(data)
    assert "line1" in result
    assert "line2" in result


def test_extract_raw_text_dict_with_result_field(monkeypatch) -> None:
    data = {"result": "extracted"}
    result = mcp_cli._extract_raw_text(data)
    assert result == "extracted"


def test_extract_raw_text_unextractable(monkeypatch) -> None:
    result = mcp_cli._extract_raw_text({"unrelated": "data"})
    assert result is None


def test_extract_raw_text_content_mixed_types(monkeypatch) -> None:
    data = {
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "text", "text": "world"},
        ]
    }
    result = mcp_cli._extract_raw_text(data)
    assert "hello" in result
    assert "world" in result
    assert "http" not in result


# ── _handle_grep ──
def test_handle_grep_match_exact_server(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="test_server", config={"type": "stdio"})
    args = argparse.Namespace(json=False, descriptions=False)
    with patch("aura_cli.mcp_cli._list_servers") as mock_list:
        mock_list.return_value = [{"server": "test_server", "tools": []}]
        rc = mcp_cli._handle_grep("test_server", {"test_server": spec}, args)
        assert rc == 0


def test_handle_grep_match_wildcard(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="test_server", config={"type": "stdio"})
    args = argparse.Namespace(json=False, descriptions=False)
    with patch("aura_cli.mcp_cli._list_servers") as mock_list:
        mock_list.return_value = [{"server": "test_server", "tools": []}]
        rc = mcp_cli._handle_grep("test_*", {"test_server": spec}, args)
        assert rc == 0


def test_handle_grep_match_tool_name(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="server", config={"type": "stdio"})
    args = argparse.Namespace(json=False, descriptions=False)
    with patch("aura_cli.mcp_cli._list_servers") as mock_list:
        mock_list.return_value = [
            {
                "server": "server",
                "tools": [{"name": "query_tool", "description": ""}],
            }
        ]
        rc = mcp_cli._handle_grep("*query*", {"server": spec}, args)
        assert rc == 0


def test_handle_grep_no_match_returns_1(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="server", config={"type": "stdio"})
    args = argparse.Namespace(json=False, descriptions=False)
    with patch("aura_cli.mcp_cli._list_servers") as mock_list:
        mock_list.return_value = [{"server": "server", "tools": []}]
        rc = mcp_cli._handle_grep("nonexistent_*", {"server": spec}, args)
        assert rc == 1


def test_handle_grep_json_output_format(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="test_server", config={"type": "stdio"})
    args = argparse.Namespace(json=True, descriptions=False)
    with patch("aura_cli.mcp_cli._list_servers") as mock_list:
        mock_list.return_value = [{"server": "test_server", "tools": []}]
        rc = mcp_cli._handle_grep("test_*", {"test_server": spec}, args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "pattern" in data
        assert "matches" in data


# ── _handle_server ──
def test_handle_server_get_info(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="myserver", config={"type": "stdio"})
    args = argparse.Namespace(json=False, raw=False, descriptions=False)
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.list_tools.return_value = [{"name": "tool1"}]
        mock_create.return_value = mock_client
        rc = mcp_cli._handle_server("myserver", {"myserver": spec}, args)
        assert rc == 0


def test_handle_server_json_includes_config(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="myserver", config={"type": "stdio", "command": "test"})
    args = argparse.Namespace(json=True, raw=False, descriptions=True)
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.list_tools.return_value = []
        mock_create.return_value = mock_client
        rc = mcp_cli._handle_server("myserver", {"myserver": spec}, args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["server"] == "myserver"
        assert "config" in data


def test_handle_server_raw_format(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="myserver", config={"type": "stdio"})
    args = argparse.Namespace(json=False, raw=True, descriptions=False)
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.list_tools.return_value = [{"name": "tool1"}]
        mock_create.return_value = mock_client
        rc = mcp_cli._handle_server("myserver", {"myserver": spec}, args)
        out = capsys.readouterr().out
        assert "myserver/tool1" in out


# ── _handle_tool ──
def test_handle_tool_get_definition(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="server", config={"type": "stdio"})
    args = argparse.Namespace(payload=None, json=False, raw=False)
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.get_tool.return_value = {"name": "tool1", "description": "Test tool"}
        mock_create.return_value = mock_client
        rc = mcp_cli._handle_tool("server", "tool1", {"server": spec}, args)
        assert rc == 0


def test_handle_tool_call_with_json_payload(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="server", config={"type": "stdio"})
    args = argparse.Namespace(
        payload='{"arg1": "value1"}',
        json=False,
        raw=False,
    )
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.call_tool.return_value = {"result": "success"}
        mock_create.return_value = mock_client
        rc = mcp_cli._handle_tool("server", "tool1", {"server": spec}, args)
        assert rc == 0


def test_handle_tool_raw_text_extraction(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="server", config={"type": "stdio"})
    args = argparse.Namespace(payload=None, json=False, raw=True)
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.get_tool.return_value = "plain text output"
        mock_create.return_value = mock_client
        rc = mcp_cli._handle_tool("server", "tool1", {"server": spec}, args)
        out = capsys.readouterr().out
        assert "plain text output" in out


def test_handle_tool_json_payload_invalid(monkeypatch, capsys) -> None:
    spec = mcp_cli.MCPServerSpec(name="server", config={"type": "stdio"})
    args = argparse.Namespace(
        payload='{"invalid": json}',
        json=False,
        raw=False,
    )
    # Mock the client so we don't try to start a subprocess
    with patch("aura_cli.mcp_cli._create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_create.return_value = mock_client
        with pytest.raises(mcp_cli.MCPCLIError, match="Invalid JSON"):
            mcp_cli._handle_tool("server", "tool1", {"server": spec}, args)


# ── main ──
def test_main_with_no_args_lists_servers(monkeypatch, capsys, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    rc = mcp_cli.main([])
    assert rc == 0


def test_main_with_server_name(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"myserver": {"type": "stdio", "command": "echo"}}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    with patch("aura_cli.mcp_cli._handle_server") as mock_handle:
        mock_handle.return_value = 0
        rc = mcp_cli.main(["myserver"])
    assert rc == 0


def test_main_with_tool_name(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"myserver": {"type": "stdio", "command": "echo"}}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    with patch("aura_cli.mcp_cli._handle_tool") as mock_handle:
        mock_handle.return_value = 0
        rc = mcp_cli.main(["myserver/tool1"])
    assert rc == 0


def test_main_grep_with_pattern(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    with patch("aura_cli.mcp_cli._handle_grep") as mock_handle:
        mock_handle.return_value = 0
        rc = mcp_cli.main(["grep", "pattern"])
    assert rc == 0


def test_main_returns_1_on_error(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setattr(mcp_cli, "_resolve_config_path", lambda _: cfg)
    rc = mcp_cli.main(["nonexistent_server"])
    assert rc == 1
