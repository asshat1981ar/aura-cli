"""Canonical MCP server manifest for AURA-managed integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MCPServerSpec:
    """Stable metadata for an AURA-managed MCP server."""

    config_name: str
    config_key: str
    output_name: str
    transport: str
    default_port: int | None
    token_env: str | None = None
    port_envs: tuple[str, ...] = ()
    legacy_token_envs: tuple[str, ...] = ()
    include_in_generated_config: bool = True
    command: str | None = None
    args: tuple[str, ...] = ()


MCP_SERVER_SPECS: tuple[MCPServerSpec, ...] = (
    MCPServerSpec(
        config_name="dev_tools",
        config_key="aura-dev-tools",
        output_name="AURA dev tools",
        transport="http",
        default_port=8001,
        token_env="AGENT_API_TOKEN",
        port_envs=("PORT", "MCP_SERVER_PORT"),
        legacy_token_envs=("MCP_DEV_TOOLS_API_KEY", "MCP_API_TOKEN"),
    ),
    MCPServerSpec(
        config_name="skills",
        config_key="aura-skills",
        output_name="AURA skills",
        transport="http",
        default_port=8002,
        token_env="MCP_API_TOKEN",
        port_envs=("MCP_SKILLS_PORT",),
        legacy_token_envs=("MCP_SKILLS_API_KEY",),
    ),
    MCPServerSpec(
        config_name="control",
        config_key="aura-control",
        output_name="AURA control",
        transport="http",
        default_port=8003,
        token_env="MCP_CONTROL_TOKEN",
        port_envs=("MCP_CONTROL_PORT",),
        legacy_token_envs=("MCP_CONTROL_API_KEY",),
    ),
    MCPServerSpec(
        config_name="agentic_loop",
        config_key="aura-agentic-loop",
        output_name="AURA agentic loop",
        transport="http",
        default_port=8006,
        token_env="AGENTIC_LOOP_TOKEN",
        port_envs=("AGENTIC_LOOP_PORT",),
        legacy_token_envs=("MCP_AGENTIC_LOOP_API_KEY",),
    ),
    MCPServerSpec(
        config_name="copilot",
        config_key="aura-copilot",
        output_name="AURA copilot",
        transport="http",
        default_port=8007,
        token_env="COPILOT_MCP_TOKEN",
        port_envs=("COPILOT_MCP_PORT",),
        legacy_token_envs=("MCP_COPILOT_API_KEY",),
    ),
    MCPServerSpec(
        config_name="sadd",
        config_key="aura-sadd",
        output_name="AURA SADD",
        transport="http",
        default_port=8020,
        token_env="SADD_MCP_TOKEN",
        port_envs=("SADD_MCP_PORT",),
    ),
    MCPServerSpec(
        config_name="playwright",
        config_key="playwright",
        output_name="Playwright",
        transport="stdio",
        default_port=None,
        command="npx",
        args=("@playwright/mcp@latest",),
    ),
)

_SPECS_BY_CONFIG_NAME = {spec.config_name: spec for spec in MCP_SERVER_SPECS}
_SPECS_BY_CONFIG_KEY = {spec.config_key: spec for spec in MCP_SERVER_SPECS}


def iter_mcp_server_specs(*, generated_only: bool = False) -> Iterable[MCPServerSpec]:
    """Iterate over the canonical MCP manifest."""
    for spec in MCP_SERVER_SPECS:
        if generated_only and not spec.include_in_generated_config:
            continue
        yield spec


def get_mcp_server_spec(server_name: str) -> MCPServerSpec:
    """Resolve a spec by config name or generated config key."""
    spec = _SPECS_BY_CONFIG_NAME.get(server_name) or _SPECS_BY_CONFIG_KEY.get(server_name)
    if spec is None:
        raise KeyError(f"Unknown MCP server '{server_name}'")
    return spec
