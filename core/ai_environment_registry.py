"""Definitions for isolated AI CLI runtime environments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from environments.config import EnvironmentConfig


@dataclass(frozen=True)
class AIEnvironmentSpec:
    name: str
    cli_command: str
    workspace_subdir: str
    description: str
    config_files: tuple[str, ...]
    env_vars: tuple[str, ...]
    bootstrap_commands: tuple[str, ...]
    test_commands: tuple[str, ...]


_ENV_SPECS: tuple[AIEnvironmentSpec, ...] = (
    AIEnvironmentSpec(
        name="gemini-cli",
        cli_command="gemini",
        workspace_subdir="gemini",
        description="Isolated Gemini CLI workspace for MCP orchestration and code generation tasks.",
        config_files=("gemini_settings.json", "aura.config.json", ".env.template"),
        env_vars=("GEMINI_API_KEY", "AURA_API_KEY", "AURA_SKIP_CHDIR"),
        bootstrap_commands=("python3 -m venv .venv", ". .venv/bin/activate && pip install -r requirements.txt"),
        test_commands=("python3 -m pytest -q",),
    ),
    AIEnvironmentSpec(
        name="claude-code",
        cli_command="claude",
        workspace_subdir="claude",
        description="Isolated Claude Code workspace with its own config, logs, artifacts, and dependency state.",
        config_files=("mcp.json", "aura.config.json", ".env.template"),
        env_vars=("ANTHROPIC_API_KEY", "AURA_API_KEY", "AURA_SKIP_CHDIR"),
        bootstrap_commands=("python3 -m venv .venv", ". .venv/bin/activate && pip install -r requirements.txt"),
        test_commands=("python3 -m pytest -q",),
    ),
    AIEnvironmentSpec(
        name="codex-cli",
        cli_command="codex",
        workspace_subdir="codex",
        description="Isolated Codex CLI workspace for code editing, testing, and MCP server operations.",
        config_files=("codex.mcp.config.json", "aura.config.json", ".env.template"),
        env_vars=("OPENAI_API_KEY", "AURA_API_KEY", "AURA_SKIP_CHDIR"),
        bootstrap_commands=("python3 -m venv .venv", ". .venv/bin/activate && pip install -r requirements.txt"),
        test_commands=("python3 -m pytest -q",),
    ),
)


def list_ai_environments(project_root: Path) -> list[Dict[str, Any]]:
    runtime_root = project_root / "environments" / "workspaces"
    environments: list[Dict[str, Any]] = []
    for spec in _ENV_SPECS:
        env_config = EnvironmentConfig.from_name(
            spec.workspace_subdir,
            spec.name,
            runtime_root,
        )
        environments.append(
            {
                "name": spec.name,
                "cli_command": spec.cli_command,
                "description": spec.description,
                "workspace_root": str(env_config.workspace_root),
                "config_dir": str(env_config.config_dir),
                "logs_dir": str(env_config.log_dir),
                "temp_dir": str(env_config.temp_dir),
                "secrets_dir": str(env_config.secrets_dir),
                "deps_dir": str(env_config.deps_dir),
                "mcp_config_path": str(env_config.mcp_config_path),
                "config_files": list(spec.config_files),
                "env_vars": list(spec.env_vars),
                "bootstrap_commands": list(spec.bootstrap_commands),
                "test_commands": list(spec.test_commands),
            }
        )
    return environments
