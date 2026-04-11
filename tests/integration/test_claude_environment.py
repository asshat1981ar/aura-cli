import json
from pathlib import Path
import shutil

import pytest

from environments.manager import EnvironmentManager
from environments.config import EnvironmentConfig


@pytest.fixture
def temp_workspaces(tmp_path: Path):
    """Fixture providing a temporary workspaces directory."""
    workspaces_dir = tmp_path / "workspaces"
    workspaces_dir.mkdir()
    yield workspaces_dir
    if workspaces_dir.exists():
        shutil.rmtree(workspaces_dir)


def test_claude_environment_isolation_and_bootstrap(temp_workspaces: Path):
    """
    Validates that a Claude Code environment can be bootstrapped successfully
    and isolated from other environments, creating its own config, logs, deps,
    and a specific .mcp.json file.
    """
    manager = EnvironmentManager(project_root=temp_workspaces.parent, workspaces_dir=temp_workspaces)

    # Bootstrap the Claude environment
    env = manager.bootstrap_environment("claude", cli_type="claude-code")

    # Verify directory structure
    assert env.workspace_root.exists()
    assert (env.workspace_root / "config").exists()
    assert (env.workspace_root / "logs").exists()
    assert (env.workspace_root / "temp").exists()
    assert (env.workspace_root / "deps").exists()
    assert (env.workspace_root / "secrets").exists()

    # Verify environment-specific aura config
    aura_config_path = env.config_dir / "aura.config.json"
    assert aura_config_path.exists()
    config_data = json.loads(aura_config_path.read_text())
    assert config_data.get("environment_name") == "claude"
    assert config_data.get("cli_type") == "claude-code"

    # Verify the specific MCP config file for claude is .mcp.json
    from environments.bootstrap import bootstrap_claude

    mcp_config_path = bootstrap_claude(env, host="127.0.0.1", project_root=temp_workspaces.parent)
    assert mcp_config_path.exists()
    assert mcp_config_path.name == "mcp.json"

    mcp_config_data = json.loads(mcp_config_path.read_text())
    assert "mcpServers" in mcp_config_data
    # Verify both HTTP and stdio servers exist
    assert "aura-hub" in mcp_config_data["mcpServers"]
    assert "filesystem" in mcp_config_data["mcpServers"]

    # Verify environment health
    health = manager.environment_health("claude")
    assert health["status"] == "healthy"
    for d in ("config", "logs", "temp", "secrets", "deps"):
        assert health["directories"][d]["exists"] is True

    # Test teardown
    result = manager.teardown_environment("claude", preserve_secrets=False)
    assert result["status"] in ("success", "removed")
    assert not env.workspace_root.exists()
