from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from tools.mcp_auth import get_mcp_server_api_key
from tools.mcp_manifest import get_mcp_server_spec


def _get_mcp_server_api_key(server_name: str = "dev_tools") -> Optional[str]:
    """Get API key for an MCP server from config or environment.

    Priority:
    1. Canonical env var from the MCP manifest
    2. Legacy aliases supported by the shared auth helper
    3. Config manager: mcp_server_api_keys.<server_name>

    Args:
        server_name: Name of the MCP server (default: "dev_tools")

    Returns:
        API key string or None if not configured
    """
    return get_mcp_server_api_key(server_name)


def _mcp_headers(server_name: str = "dev_tools") -> dict[str, str]:
    """Return auth headers for MCP requests, including API key if configured.

    Supports both X-API-Key (preferred) and legacy Authorization: Bearer.

    Args:
        server_name: Name of the MCP server to get headers for

    Returns:
        Dictionary of HTTP headers
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}

    api_key = _get_mcp_server_api_key(server_name)
    if api_key:
        # Use X-API-Key header (preferred)
        headers["X-API-Key"] = api_key

    return headers


def _mcp_base_url(server_name: str = "dev_tools") -> str:
    """Base URL for MCP server (default http://localhost:8001).

    Args:
        server_name: Name of the MCP server to get URL for

    Returns:
        Base URL string
    """
    # Try server-specific env var first
    env_var = f"MCP_{server_name.upper()}_URL"
    env_url = os.getenv(env_var, "").strip()
    if env_url:
        return env_url

    # Fall back to config
    try:
        from core.config_manager import config as _cfg

        port = _cfg.get_mcp_server_port(server_name)
        return f"http://localhost:{port}"
    except Exception:
        pass

    # Default fallback
    try:
        spec = get_mcp_server_spec(server_name)
    except KeyError:
        port = 8001
    else:
        port = spec.default_port or 8001
    return f"http://localhost:{port}"


def _mcp_request(
    method: str,
    path: str,
    data: dict | None = None,
    server_name: str = "dev_tools",
) -> tuple[int, dict]:
    """Small HTTP helper for MCP server; returns (status, json/dict).

    Args:
        method: HTTP method (GET, POST, etc.)
        path: API path (e.g., /tools, /call)
        data: Optional request body data
        server_name: Name of the MCP server to call

    Returns:
        Tuple of (status_code, response_data)
    """
    url = f"{_mcp_base_url(server_name)}{path}"
    headers = _mcp_headers(server_name)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # Handle auth errors specifically
        if exc.code == 401:
            return exc.code, {"error": "Authentication required. Set the canonical server token env var or configure mcp_server_api_keys."}
        if exc.code == 403:
            return exc.code, {"error": "Invalid API key"}
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:
            return exc.code, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def cmd_mcp_tools(server_name: str = "dev_tools") -> int:
    """List servers and tools exposed by the repo-local MCP config.

    Args:
        server_name: Name of the MCP server to query
    """
    from aura_cli.mcp_cli import main as mcp_cli_main

    return mcp_cli_main([])


def cmd_mcp_call(tool: str, args_json: str | None, server_name: str = "dev_tools") -> int:
    """Call an MCP tool by name with JSON args.

    Args:
        tool: Name of the tool to call
        args_json: JSON string of arguments
        server_name: Name of the MCP server to call
    """
    from aura_cli.mcp_cli import main as mcp_cli_main

    argv = [tool]
    if args_json:
        argv.append(args_json)
    return mcp_cli_main(argv)


def cmd_diag(server_name: str = "dev_tools") -> None:
    """Fetch MCP health/metrics/limits/log tail and linter capabilities.

    Args:
        server_name: Name of the MCP server to diagnose
    """
    health = _mcp_request("GET", "/health", server_name=server_name)
    metrics = _mcp_request("GET", "/metrics", server_name=server_name)
    limits = _mcp_request("POST", "/call", {"tool_name": "limits", "args": {}}, server_name=server_name)
    lcap = _mcp_request("POST", "/call", {"tool_name": "linter_capabilities", "args": {}}, server_name=server_name)
    tail = _mcp_request("POST", "/call", {"tool_name": "tail_logs", "args": {"lines": 50}}, server_name=server_name)
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


# R8: New functions for multi-server support


def cmd_mcp_servers_list() -> None:
    """List all configured MCP servers and their status."""
    from core.config_manager import config as _cfg, DEFAULT_CONFIG

    servers = DEFAULT_CONFIG.get("mcp_servers", {})

    result = []
    for name, default_port in servers.items():
        try:
            port = _cfg.get_mcp_server_port(name)
            has_key = _get_mcp_server_api_key(name) is not None

            # Try to connect
            health_status, health_data = _mcp_request("GET", "/health", server_name=name)
            is_running = health_status == 200

            result.append(
                {
                    "name": name,
                    "port": port,
                    "configured": True,
                    "auth_enabled": has_key,
                    "running": is_running,
                    "status": "ok" if is_running else "unreachable",
                }
            )
        except Exception as e:
            result.append(
                {
                    "name": name,
                    "port": default_port,
                    "configured": False,
                    "auth_enabled": False,
                    "running": False,
                    "status": f"error: {e}",
                }
            )

    print(json.dumps({"servers": result}, indent=2))


def cmd_mcp_server_health(server_name: str) -> None:
    """Check health of a specific MCP server.

    Args:
        server_name: Name of the MCP server to check
    """
    status, data = _mcp_request("GET", "/health", server_name=server_name)
    print(
        json.dumps(
            {
                "server": server_name,
                "status": status,
                "healthy": status == 200,
                "data": data,
            },
            indent=2,
        )
    )
