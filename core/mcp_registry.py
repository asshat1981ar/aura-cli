"""Central registry metadata for AURA MCP-compatible HTTP services."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from core.config_manager import config as _config


@dataclass(frozen=True)
class MCPServiceSpec:
    config_name: str
    server_name: str
    title: str
    kind: str
    auth_env: str | None
    port_envs: tuple[str, ...]
    endpoints: tuple[str, ...]
    capabilities: tuple[str, ...]


_SERVICE_SPECS: tuple[MCPServiceSpec, ...] = (
    MCPServiceSpec(
        config_name="dev_tools",
        server_name="aura-dev-tools",
        title="AURA dev tools",
        kind="tooling",
        auth_env="AGENT_API_TOKEN",
        port_envs=("PORT", "MCP_SERVER_PORT"),
        endpoints=(
            "/health",
            "/metrics",
            "/tools",
            "/execute",
            "/discovery",
            "/environments",
            "/architecture",
            "/events/stream",
            "/events/publish",
            "/events/history",
        ),
        capabilities=("tool-execution", "goal-execution", "event-streaming", "a2a"),
    ),
    MCPServiceSpec(
        config_name="skills",
        server_name="aura-skills",
        title="AURA skills",
        kind="skills",
        auth_env="MCP_API_TOKEN",
        port_envs=("MCP_SKILLS_PORT",),
        endpoints=("/health", "/tools", "/discovery", "/skill/{name}", "/call"),
        capabilities=("skill-discovery", "skill-execution"),
    ),
    MCPServiceSpec(
        config_name="control",
        server_name="aura-control",
        title="AURA control",
        kind="control-plane",
        auth_env="MCP_CONTROL_TOKEN",
        port_envs=("MCP_CONTROL_PORT",),
        endpoints=("/health", "/tools", "/discovery", "/tool/{name}", "/call"),
        capabilities=("queue-control", "memory-control", "project-status"),
    ),
    MCPServiceSpec(
        config_name="agentic_loop",
        server_name="aura-agentic-loop",
        title="AURA agentic loop",
        kind="orchestration",
        auth_env="AGENTIC_LOOP_TOKEN",
        port_envs=("AGENTIC_LOOP_PORT",),
        endpoints=("/health", "/tools", "/discovery", "/tool/{name}", "/workflows", "/loops", "/call"),
        capabilities=("workflow-execution", "loop-control"),
    ),
    MCPServiceSpec(
        config_name="copilot",
        server_name="aura-copilot",
        title="AURA copilot",
        kind="copilot",
        auth_env="COPILOT_MCP_TOKEN",
        port_envs=("COPILOT_MCP_PORT",),
        endpoints=("/health", "/tools", "/discovery", "/tool/{name}", "/call"),
        capabilities=("github-analysis", "ai-review", "plan-generation"),
    ),
)


def _default_host() -> str:
    return os.getenv("AURA_COPILOT_MCP_HOST") or os.getenv("AURA_MCP_HOST") or "127.0.0.1"


def _resolve_port(spec: MCPServiceSpec) -> int:
    for env_name in spec.port_envs:
        value = os.getenv(env_name)
        if value:
            return int(value)
    return int(_config.get_mcp_server_port(spec.config_name))


def describe_service(spec: MCPServiceSpec, *, host: str | None = None) -> Dict[str, Any]:
    resolved_host = host or _default_host()
    port = _resolve_port(spec)
    return {
        "name": spec.server_name,
        "title": spec.title,
        "kind": spec.kind,
        "config_name": spec.config_name,
        "port": port,
        "url": f"http://{resolved_host}:{port}",
        "auth_env": spec.auth_env,
        "auth_configured": bool(spec.auth_env and os.getenv(spec.auth_env)),
        "endpoints": list(spec.endpoints),
        "capabilities": list(spec.capabilities),
    }


def list_registered_services(*, host: str | None = None) -> list[Dict[str, Any]]:
    return [describe_service(spec, host=host) for spec in _SERVICE_SPECS]


def get_registered_service(config_name: str, *, host: str | None = None) -> Dict[str, Any]:
    for spec in _SERVICE_SPECS:
        if spec.config_name == config_name:
            return describe_service(spec, host=host)
    raise KeyError(f"Unknown MCP service config name: {config_name}")


def iter_service_specs() -> Iterable[MCPServiceSpec]:
    return iter(_SERVICE_SPECS)
