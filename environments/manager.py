"""Central environment lifecycle manager for multi-CLI workspace isolation."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from environments.config import EnvironmentConfig
from environments.isolation import (
    cleanup_stale,
    create_workspace_tree,
    teardown_workspace,
)


# Default workspace base directory (relative to project root)
_DEFAULT_WORKSPACES_DIR = "environments/workspaces"

# Default port ranges per CLI environment
_DEFAULT_PORT_RANGES: Dict[str, tuple] = {
    "gemini": (8020, 8029),
    "claude": (8030, 8039),
    "codex": (8040, 8049),
}

# CLI type mapping
_CLI_TYPES: Dict[str, str] = {
    "gemini": "gemini-cli",
    "claude": "claude-code",
    "codex": "codex-cli",
}


class EnvironmentManager:
    """Manages lifecycle of isolated AI CLI environments.

    Each environment gets its own workspace directory with config, logs,
    temp files, secrets, and dependency installs.
    """

    def __init__(self, project_root: Path, workspaces_dir: Optional[Path] = None):
        self.project_root = Path(project_root).resolve()
        self.workspaces_dir = (
            Path(workspaces_dir)
            if workspaces_dir
            else self.project_root / _DEFAULT_WORKSPACES_DIR
        )
        self._environments: Dict[str, EnvironmentConfig] = {}
        self._registry_path = self.workspaces_dir / ".registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load environment registry from disk."""
        if not self._registry_path.exists():
            return
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            for name, info in data.items():
                cli_type = info.get("cli_type", _CLI_TYPES.get(name, "unknown"))
                port_range = tuple(info.get("port_range", _DEFAULT_PORT_RANGES.get(name, (8010, 8019))))
                model_routing = info.get("model_routing", {})
                self._environments[name] = EnvironmentConfig.from_name(
                    name=name,
                    cli_type=cli_type,
                    base_dir=self.workspaces_dir,
                    port_range=port_range,
                    model_routing=model_routing,
                )
        except (json.JSONDecodeError, OSError):
            pass

    def _save_registry(self) -> None:
        """Persist environment registry to disk."""
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        data = {
            name: env.as_dict()
            for name, env in self._environments.items()
        }
        self._registry_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def bootstrap_environment(
        self,
        name: str,
        cli_type: Optional[str] = None,
        *,
        port_range: Optional[tuple] = None,
        model_routing: Optional[Dict[str, str]] = None,
        max_concurrent_tasks: int = 3,
    ) -> EnvironmentConfig:
        """Create and register a new isolated environment.

        Args:
            name: Environment name (e.g., "gemini", "claude", "codex").
            cli_type: CLI identifier. Auto-detected from name if omitted.
            port_range: Port range override.
            model_routing: Model routing overrides.
            max_concurrent_tasks: Max parallel tasks.

        Returns:
            The created EnvironmentConfig.
        """
        if cli_type is None:
            cli_type = _CLI_TYPES.get(name, f"{name}-cli")

        if port_range is None:
            port_range = _DEFAULT_PORT_RANGES.get(name, (8010, 8019))

        env = EnvironmentConfig.from_name(
            name=name,
            cli_type=cli_type,
            base_dir=self.workspaces_dir,
            port_range=port_range,
            model_routing=model_routing or {},
            max_concurrent_tasks=max_concurrent_tasks,
        )

        # Create workspace directory tree
        create_workspace_tree(env.workspace_root)

        # Write environment-specific aura.config.json
        self._write_env_config(env)

        # Register
        self._environments[name] = env
        self._save_registry()

        return env

    def _write_env_config(self, env: EnvironmentConfig) -> None:
        """Write an environment-specific aura.config.json."""
        base_config_path = self.project_root / "aura.config.json"
        if base_config_path.exists():
            try:
                config = json.loads(base_config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                config = {}
        else:
            config = {}

        # Apply environment-specific overrides
        config["environment_name"] = env.name
        config["cli_type"] = env.cli_type
        if env.model_routing:
            config["model_routing"] = {
                **config.get("model_routing", {}),
                **env.model_routing,
            }

        env.config_dir.mkdir(parents=True, exist_ok=True)
        (env.config_dir / "aura.config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    def teardown_environment(
        self, name: str, *, preserve_secrets: bool = True
    ) -> dict:
        """Tear down an environment workspace.

        Args:
            name: Environment name.
            preserve_secrets: If True, keep the secrets directory.

        Returns:
            Status dict from teardown_workspace.
        """
        env = self._environments.get(name)
        if env is None:
            return {"status": "not_found", "name": name}

        result = teardown_workspace(env.workspace_root, preserve_secrets=preserve_secrets)
        del self._environments[name]
        self._save_registry()
        return result

    def get_environment(self, name: str) -> Optional[EnvironmentConfig]:
        """Get environment config by name."""
        return self._environments.get(name)

    def list_environments(self) -> List[EnvironmentConfig]:
        """List all registered environments."""
        return list(self._environments.values())

    def environment_health(self, name: str) -> dict:
        """Check workspace integrity and resource usage for an environment.

        Returns:
            Dict with health status, disk usage, and directory checks.
        """
        env = self._environments.get(name)
        if env is None:
            return {"status": "not_found", "name": name}

        checks = {}
        total_size = 0

        for subdir in ("config", "logs", "temp", "secrets", "deps"):
            d = env.workspace_root / subdir
            exists = d.is_dir()
            size = 0
            file_count = 0
            if exists:
                for f in d.rglob("*"):
                    if f.is_file():
                        file_count += 1
                        try:
                            size += f.stat().st_size
                        except OSError:
                            pass
            total_size += size
            checks[subdir] = {
                "exists": exists,
                "file_count": file_count,
                "size_bytes": size,
            }

        return {
            "status": "healthy" if all(c["exists"] for c in checks.values()) else "degraded",
            "name": name,
            "cli_type": env.cli_type,
            "total_size_bytes": total_size,
            "directories": checks,
        }

    def cleanup_environment(self, name: str, max_age_hours: int = 24) -> dict:
        """Clean up stale temp and log files in an environment."""
        env = self._environments.get(name)
        if env is None:
            return {"status": "not_found", "name": name}
        return cleanup_stale(env.workspace_root, max_age_hours=max_age_hours)
