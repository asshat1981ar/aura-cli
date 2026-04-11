from tools.mcp_auth import get_mcp_server_api_key
from tools.mcp_manifest import get_mcp_server_spec


def test_manifest_exposes_canonical_auth_envs():
    assert get_mcp_server_spec("dev_tools").token_env == "AGENT_API_TOKEN"
    assert get_mcp_server_spec("skills").token_env == "MCP_API_TOKEN"
    assert get_mcp_server_spec("sadd").token_env == "SADD_MCP_TOKEN"


def test_manifest_exposes_generated_server_keys():
    assert get_mcp_server_spec("aura-dev-tools").config_name == "dev_tools"
    assert get_mcp_server_spec("aura-sadd").default_port == 8020
    assert get_mcp_server_spec("playwright").transport == "stdio"


def test_auth_uses_canonical_env_before_legacy_alias(monkeypatch):
    monkeypatch.setenv("MCP_DEV_TOOLS_API_KEY", "legacy-token")
    monkeypatch.setenv("AGENT_API_TOKEN", "canonical-token")

    assert get_mcp_server_api_key("dev_tools") == "canonical-token"


def test_auth_uses_sadd_canonical_env(monkeypatch):
    monkeypatch.setenv("SADD_MCP_TOKEN", "sadd-token")

    assert get_mcp_server_api_key("sadd") == "sadd-token"
