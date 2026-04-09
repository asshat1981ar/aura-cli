from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

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
