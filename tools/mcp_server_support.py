"""Shared infrastructure helpers for AURA MCP HTTP servers.

This module intentionally covers only repeated bootstrap/auth/metrics concerns.
Each server keeps its own request models, tool schemas, and domain handlers.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Mapping

from tools.mcp_auth import get_api_key_validator, is_auth_enabled
from tools.mcp_manifest import get_mcp_server_spec


def auth_dependency(server_name: str):
    """Return the shared FastAPI auth dependency for a manifest-defined server."""
    return get_api_key_validator(server_name)


def resolve_server_port(server_name: str) -> int:
    """Resolve an MCP server port from env overrides, config, or manifest defaults."""
    spec = get_mcp_server_spec(server_name)
    if spec.default_port is None:
        raise ValueError(f"Server '{server_name}' does not expose an HTTP port")

    for env_name in spec.port_envs:
        value = os.getenv(env_name, "").strip()
        if value:
            return int(value)

    try:
        from core.config_manager import config as _cfg

        return int(_cfg.get_mcp_server_port(server_name))
    except Exception:
        return spec.default_port


def auth_mode_label(server_name: str) -> str:
    """Return a human-readable auth mode label for startup logs."""
    return "enabled" if is_auth_enabled(server_name) else "optional"


def uptime_seconds(server_start: float) -> float:
    """Return rounded uptime for metrics/health payloads."""
    return round(time.time() - server_start, 1)


def build_tool_metrics(
    tool_names: Iterable[str],
    call_counts: Mapping[str, int],
    call_errors: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Build a stable per-tool call/error summary."""
    return {
        name: {
            "calls": call_counts.get(name, 0),
            "errors": call_errors.get(name, 0),
        }
        for name in tool_names
    }


def build_basic_metrics_payload(
    *,
    server_start: float,
    tool_names: Iterable[str],
    call_counts: Mapping[str, int],
    call_errors: Mapping[str, int],
    **extra: Any,
) -> dict[str, Any]:
    """Build the common MCP metrics payload shared by multiple servers."""
    total_calls = sum(call_counts.values())
    total_errors = sum(call_errors.values())
    return {
        "uptime_seconds": uptime_seconds(server_start),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_calls, 1), 4),
        "tools": build_tool_metrics(tool_names, call_counts, call_errors),
        **extra,
    }
