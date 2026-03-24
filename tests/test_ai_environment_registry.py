from pathlib import Path

from core.ai_environment_registry import list_ai_environments


def test_list_ai_environments_matches_workspace_layout():
    project_root = Path("/tmp/aura-project")

    environments = {env["name"]: env for env in list_ai_environments(project_root)}

    assert environments["gemini-cli"]["workspace_root"].endswith("environments/workspaces/gemini")
    assert environments["claude-code"]["workspace_root"].endswith("environments/workspaces/claude")
    assert environments["codex-cli"]["workspace_root"].endswith("environments/workspaces/codex")


def test_list_ai_environments_exposes_real_config_filenames():
    project_root = Path("/tmp/aura-project")

    environments = {env["name"]: env for env in list_ai_environments(project_root)}

    assert environments["gemini-cli"]["mcp_config_path"].endswith("config/gemini_settings.json")
    assert environments["claude-code"]["mcp_config_path"].endswith("config/mcp.json")
    assert environments["codex-cli"]["mcp_config_path"].endswith("config/codex.mcp.config.json")
