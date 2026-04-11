"""Per-CLI bootstrap functions that generate environment-specific MCP configs.

Follows patterns from:
  - scripts/configure_gemini_mcp.sh (Gemini CLI)
  - scripts/write_copilot_mcp_config.py (Claude/Copilot)
  - scripts/mcp_server_setup.sh (general MCP setup)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from environments.config import EnvironmentConfig


# Default MCP server ports (internal HTTP servers)
_DEFAULT_MCP_PORTS: Dict[str, int] = {
    "dev_tools": 8001,
    "skills": 8002,
    "control": 8003,
    "thinking": 8004,
    "agentic_loop": 8006,
    "copilot": 8007,
    "hub": 8010,
    "docker": 8011,
    "kubernetes": 8012,
    "neo4j": 8013,
    "redis": 8014,
    "notification": 8015,
    "monitoring": 8016,
    "weaviate": 8017,
}


def _mcp_server_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def bootstrap_gemini(
    env: EnvironmentConfig,
    *,
    host: str = "127.0.0.1",
    mcp_ports: Optional[Dict[str, int]] = None,
) -> Path:
    """Generate Gemini CLI settings.json with MCP server entries.

    Follows the pattern in scripts/configure_gemini_mcp.sh.

    Returns:
        Path to the generated settings file.
    """
    ports = {**_DEFAULT_MCP_PORTS, **(mcp_ports or {})}

    # Gemini CLI uses mcpServers in settings.json
    settings = {
        "mcpServers": {
            f"aura-{name}": {
                "url": _mcp_server_url(host, port),
                "headers": {"Authorization": "${MCP_API_TOKEN}"},
            }
            for name, port in ports.items()
        }
    }

    env.config_dir.mkdir(parents=True, exist_ok=True)
    output_path = env.mcp_config_path
    output_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    # Write env vars template
    _write_env_template(env, ports)

    return output_path


def bootstrap_claude(
    env: EnvironmentConfig,
    *,
    host: str = "127.0.0.1",
    mcp_ports: Optional[Dict[str, int]] = None,
    project_root: Optional[Path] = None,
) -> Path:
    """Generate Claude Code .mcp.json with both stdio and HTTP servers.

    Returns:
        Path to the generated config file.
    """
    ports = {**_DEFAULT_MCP_PORTS, **(mcp_ports or {})}

    # Claude Code uses mcpServers in .mcp.json
    mcp_config: Dict = {"mcpServers": {}}

    # Add internal HTTP servers
    for name, port in ports.items():
        mcp_config["mcpServers"][f"aura-{name}"] = {
            "type": "http",
            "url": _mcp_server_url(host, port),
            "headers": {"Authorization": "Bearer ${MCP_API_TOKEN}"},
        }

    # Add standard stdio servers
    stdio_servers = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", str(project_root) if project_root else "."],
        },
        "git": {
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-git"],
        },
        "sequential-thinking": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        },
        "memory": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
        },
        "fetch": {
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-fetch"],
        },
    }

    for name, spec in stdio_servers.items():
        mcp_config["mcpServers"][name] = {"type": "stdio", **spec}

    env.config_dir.mkdir(parents=True, exist_ok=True)
    output_path = env.mcp_config_path
    output_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")

    _write_env_template(env, ports)

    return output_path


def bootstrap_codex(
    env: EnvironmentConfig,
    *,
    host: str = "127.0.0.1",
    mcp_ports: Optional[Dict[str, int]] = None,
) -> Path:
    """Generate Codex CLI MCP config.

    Returns:
        Path to the generated config file.
    """
    ports = {**_DEFAULT_MCP_PORTS, **(mcp_ports or {})}

    # Codex CLI uses a similar format to Claude
    codex_config = {
        "mcpServers": {
            f"aura-{name}": {
                "type": "http",
                "url": _mcp_server_url(host, port),
            }
            for name, port in ports.items()
        }
    }

    env.config_dir.mkdir(parents=True, exist_ok=True)
    output_path = env.mcp_config_path
    output_path.write_text(json.dumps(codex_config, indent=2), encoding="utf-8")

    _write_env_template(env, ports)

    return output_path


def _write_env_template(env: EnvironmentConfig, ports: Dict[str, int]) -> None:
    """Write a .env template file with placeholder values."""
    lines = [
        "# Environment variables for AURA MCP servers",
        f"# Environment: {env.name} ({env.cli_type})",
        "# Replace placeholders with real values before use.",
        "",
        "MCP_API_TOKEN=changeme",
        "MCP_CONTROL_TOKEN=changeme",
        "AGENTIC_LOOP_TOKEN=changeme",
        "COPILOT_MCP_TOKEN=changeme",
        "HUB_TOKEN=changeme",
        "",
        "# External service credentials",
        "GITHUB_PAT=changeme",
        "NEO4J_PASSWORD=changeme",
        "REDIS_PASSWORD=changeme",
        "SLACK_BOT_TOKEN=changeme",
        "BRAVE_API_KEY=changeme",
        "",
        "# MCP server ports",
    ]
    for name, port in sorted(ports.items()):
        env_name = f"MCP_{name.upper()}_PORT"
        lines.append(f"{env_name}={port}")

    env.secrets_dir.mkdir(parents=True, exist_ok=True)
    (env.secrets_dir / ".env.template").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bootstrap_environment(
    env: EnvironmentConfig,
    *,
    host: str = "127.0.0.1",
    mcp_ports: Optional[Dict[str, int]] = None,
    project_root: Optional[Path] = None,
) -> Path:
    """Bootstrap the correct config for an environment based on its cli_type.

    Returns:
        Path to the generated MCP config file.
    """
    dispatchers = {
        "gemini-cli": bootstrap_gemini,
        "claude-code": bootstrap_claude,
        "codex-cli": bootstrap_codex,
    }

    fn = dispatchers.get(env.cli_type)
    if fn is None:
        raise ValueError(f"Unknown CLI type: {env.cli_type}")

    kwargs = {"host": host, "mcp_ports": mcp_ports}
    if env.cli_type == "claude-code":
        kwargs["project_root"] = project_root

    return fn(env, **kwargs)
