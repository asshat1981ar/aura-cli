"""Dispatch handlers for MCP-related commands (B4)."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

from aura_cli.dispatch._helpers import _run_json_printing_callable_with_warnings


# ── MCP HTTP helpers ──────────────────────────────────────────────────────────

def _mcp_headers():
    """Return auth headers for MCP requests, if MCP_API_TOKEN is set."""
    token = os.getenv("MCP_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _mcp_base_url():
    """Base URL for MCP server (default http://localhost:8001)."""
    return os.getenv("MCP_SERVER_URL", "http://localhost:8001")


def _mcp_request(method: str, path: str, data: dict | None = None):
    """Small HTTP helper for MCP server; returns (status, json/dict)."""
    url = f"{_mcp_base_url()}{path}"
    headers = {"Content-Type": "application/json", **_mcp_headers()}
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 500, {"error": str(e)}


def cmd_mcp_tools():
    """List MCP tools via HTTP client."""
    status, data = _mcp_request("GET", "/tools")
    print(json.dumps({"status": status, "data": data}, indent=2))


def cmd_mcp_call(tool: str, args_json: str | None):
    """Call an MCP tool by name with JSON args."""
    try:
        args_obj = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as exc:
        print(f"Invalid args JSON: {exc}")
        return
    payload = {"tool_name": tool, "args": args_obj}
    status, data = _mcp_request("POST", "/call", payload)
    print(json.dumps({"status": status, "data": data}, indent=2))


def cmd_diag():
    """Fetch MCP health/metrics/limits/log tail and linter capabilities."""
    health = _mcp_request("GET", "/health")
    metrics = _mcp_request("GET", "/metrics")
    limits = _mcp_request("POST", "/call", {"tool_name": "limits", "args": {}})
    lcap = _mcp_request("POST", "/call", {"tool_name": "linter_capabilities", "args": {}})
    tail = _mcp_request("POST", "/call", {"tool_name": "tail_logs", "args": {"lines": 50}})
    print(json.dumps({
        "health": {"status": health[0], "data": health[1]},
        "metrics": {"status": metrics[0], "data": metrics[1]},
        "limits": {"status": limits[0], "data": limits[1]},
        "linter_capabilities": {"status": lcap[0], "data": lcap[1]},
        "tail_logs": {"status": tail[0], "data": tail[1]},
    }, indent=2))


def _run_local_script(script_name: str) -> dict:
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / script_name
    if not script.exists():
        return {"status": 1, "ok": False, "error": f"script not found: {script}"}

    proc = subprocess.run(
        [str(script)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=str(script.parent.parent),
        check=False,
    )
    return {
        "status": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "script": str(script),
    }


def cmd_mcp_check():
    """Run local MCP readiness checker script and print structured results."""
    print(json.dumps(_run_local_script("mcp_server_check.sh"), indent=2))


def cmd_mcp_setup():
    """Run local MCP setup script and print structured results."""
    print(json.dumps(_run_local_script("mcp_server_setup.sh"), indent=2))


# ── Dispatch handlers ─────────────────────────────────────────────────────────

def handle_mcp_tools(ctx) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_mcp_tools)
    return 0


def handle_mcp_call(ctx) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_mcp_call, ctx.args.mcp_call, ctx.args.mcp_args)
    return 0


def handle_diag(ctx) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_diag)
    return 0


def handle_mcp_check(ctx) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_mcp_check)
    return 0


def handle_mcp_setup(ctx) -> int:
    _run_json_printing_callable_with_warnings(ctx, cmd_mcp_setup)
    return 0
