from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _mcp_headers() -> dict[str, str]:
    """Return auth headers for MCP requests, if MCP_API_TOKEN is set."""
    token = os.getenv("MCP_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _mcp_base_url() -> str:
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
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:
            return exc.code, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def cmd_mcp_tools() -> None:
    """List MCP tools via HTTP client."""
    status, data = _mcp_request("GET", "/tools")
    print(json.dumps({"status": status, "data": data}, indent=2))


def cmd_mcp_call(tool: str, args_json: str | None) -> None:
    """Call an MCP tool by name with JSON args."""
    try:
        args_obj = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as exc:
        print(f"Invalid args JSON: {exc}")
        return
    payload = {"tool_name": tool, "args": args_obj}
    status, data = _mcp_request("POST", "/call", payload)
    print(json.dumps({"status": status, "data": data}, indent=2))


def cmd_diag() -> None:
    """Fetch MCP health/metrics/limits/log tail and linter capabilities."""
    health = _mcp_request("GET", "/health")
    metrics = _mcp_request("GET", "/metrics")
    limits = _mcp_request("POST", "/call", {"tool_name": "limits", "args": {}})
    lcap = _mcp_request("POST", "/call", {"tool_name": "linter_capabilities", "args": {}})
    tail = _mcp_request("POST", "/call", {"tool_name": "tail_logs", "args": {"lines": 50}})
    print(
        json.dumps(
            {
                "health": {"status": health[0], "data": health[1]},
                "metrics": {"status": metrics[0], "data": metrics[1]},
                "limits": {"status": limits[0], "data": limits[1]},
                "linter_capabilities": {"status": lcap[0], "data": lcap[1]},
                "tail_logs": {"status": tail[0], "data": tail[1]},
            },
            indent=2,
        )
    )
