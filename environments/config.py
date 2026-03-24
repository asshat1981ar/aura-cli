"""Environment configuration dataclass for isolated AI CLI workspaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class EnvironmentConfig:
    """Immutable configuration for a single AI CLI environment.

    Each environment gets its own workspace directory tree with isolated
    config, logs, temp files, secrets, and dependency installs.
    """

    name: str
    cli_type: str  # "gemini-cli", "claude-code", "codex-cli"
    workspace_root: Path
    config_dir: Path
    log_dir: Path
    temp_dir: Path
    secrets_dir: Path
    deps_dir: Path
    mcp_config_path: Path
    port_range: Tuple[int, int] = (8010, 8020)
    model_routing: Dict[str, str] = field(default_factory=dict)
    max_concurrent_tasks: int = 3
    cleanup_on_exit: bool = True
    env_vars: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_name(
        cls,
        name: str,
        cli_type: str,
        base_dir: Path,
        *,
        port_range: Tuple[int, int] = (8010, 8020),
        model_routing: Optional[Dict[str, str]] = None,
        max_concurrent_tasks: int = 3,
    ) -> "EnvironmentConfig":
        """Create an EnvironmentConfig with standard workspace layout.

        Args:
            name: Environment name (e.g., "gemini", "claude", "codex").
            cli_type: CLI identifier (e.g., "gemini-cli").
            base_dir: Parent directory for all workspaces.
            port_range: Inclusive (start, end) port range for this env.
            model_routing: Optional model routing overrides.
            max_concurrent_tasks: Max parallel tasks in this env.
        """
        workspace = base_dir / name
        config_dir = workspace / "config"

        # CLI-specific MCP config file names
        mcp_config_names = {
            "gemini-cli": "gemini_settings.json",
            "claude-code": "mcp.json",
            "codex-cli": "codex.mcp.config.json",
        }
        mcp_filename = mcp_config_names.get(cli_type, "mcp.json")

        return cls(
            name=name,
            cli_type=cli_type,
            workspace_root=workspace,
            config_dir=config_dir,
            log_dir=workspace / "logs",
            temp_dir=workspace / "temp",
            secrets_dir=workspace / "secrets",
            deps_dir=workspace / "deps",
            mcp_config_path=config_dir / mcp_filename,
            port_range=port_range,
            model_routing=model_routing or {},
            max_concurrent_tasks=max_concurrent_tasks,
        )

    def as_dict(self) -> Dict:
        """Serialize to a JSON-safe dict."""
        return {
            "name": self.name,
            "cli_type": self.cli_type,
            "workspace_root": str(self.workspace_root),
            "config_dir": str(self.config_dir),
            "log_dir": str(self.log_dir),
            "temp_dir": str(self.temp_dir),
            "secrets_dir": str(self.secrets_dir),
            "deps_dir": str(self.deps_dir),
            "mcp_config_path": str(self.mcp_config_path),
            "port_range": list(self.port_range),
            "model_routing": self.model_routing,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "cleanup_on_exit": self.cleanup_on_exit,
        }
