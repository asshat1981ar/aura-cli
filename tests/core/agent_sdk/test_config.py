"""Tests for core/agent_sdk/config.py — AgentSDKConfig."""

import os
import pytest
from pathlib import Path

from core.agent_sdk.config import AgentSDKConfig, _DEFAULT_ALLOWED_TOOLS, _DEFAULT_MCP_PORTS


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestAgentSDKConfigDefaults:
    def test_default_model(self):
        cfg = AgentSDKConfig()
        assert cfg.model == "claude-sonnet-4-6"

    def test_default_max_turns(self):
        assert AgentSDKConfig().max_turns == 30

    def test_default_max_budget(self):
        assert AgentSDKConfig().max_budget_usd == 2.0

    def test_default_permission_mode(self):
        assert AgentSDKConfig().permission_mode == "acceptEdits"

    def test_default_allowed_tools_contains_core(self):
        cfg = AgentSDKConfig()
        for tool in ("Read", "Write", "Edit", "Bash"):
            assert tool in cfg.allowed_tools

    def test_default_mcp_ports_contains_expected(self):
        cfg = AgentSDKConfig()
        assert "dev_tools" in cfg.mcp_ports
        assert "sadd" in cfg.mcp_ports
        assert cfg.mcp_ports["dev_tools"] == 8001

    def test_default_enable_thinking(self):
        assert AgentSDKConfig().enable_thinking is True

    def test_default_enable_subagents(self):
        assert AgentSDKConfig().enable_subagents is True

    def test_default_escalation_threshold(self):
        assert AgentSDKConfig().escalation_threshold == 2

    def test_default_skill_weight_cap(self):
        assert AgentSDKConfig().skill_weight_cap == 1.0

    def test_default_skill_weight_floor(self):
        assert AgentSDKConfig().skill_weight_floor == 0.1


# ---------------------------------------------------------------------------
# from_aura_config
# ---------------------------------------------------------------------------

class TestFromAuraConfig:
    def test_empty_config_uses_defaults(self):
        cfg = AgentSDKConfig.from_aura_config({})
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.max_turns == 30

    def test_agent_sdk_section_overrides_model(self):
        cfg = AgentSDKConfig.from_aura_config({"agent_sdk": {"model": "claude-opus-4-6"}})
        assert cfg.model == "claude-opus-4-6"

    def test_agent_sdk_section_overrides_max_turns(self):
        cfg = AgentSDKConfig.from_aura_config({"agent_sdk": {"max_turns": 50}})
        assert cfg.max_turns == 50

    def test_mcp_servers_section_overrides_ports(self):
        custom_ports = {"dev_tools": 9001, "skills": 9002}
        cfg = AgentSDKConfig.from_aura_config({"mcp_servers": custom_ports})
        assert cfg.mcp_ports["dev_tools"] == 9001

    def test_project_root_from_config(self):
        cfg = AgentSDKConfig.from_aura_config({"agent_sdk": {"project_root": "/my/project"}})
        assert cfg.project_root == "/my/project"

    def test_model_stats_path_is_path(self):
        cfg = AgentSDKConfig.from_aura_config({})
        assert isinstance(cfg.model_stats_path, Path)

    def test_scan_exclude_patterns_from_config(self):
        cfg = AgentSDKConfig.from_aura_config({
            "agent_sdk": {"scan_exclude_patterns": ["build", "dist"]}
        })
        assert cfg.scan_exclude_patterns == ["build", "dist"]


# ---------------------------------------------------------------------------
# apply_env_overrides
# ---------------------------------------------------------------------------

class TestApplyEnvOverrides:
    def test_model_override_from_env(self, monkeypatch):
        monkeypatch.setenv("AURA_AGENT_SDK_MODEL", "claude-haiku-4-5")
        cfg = AgentSDKConfig()
        cfg.apply_env_overrides()
        assert cfg.model == "claude-haiku-4-5"

    def test_max_turns_override_from_env(self, monkeypatch):
        monkeypatch.setenv("AURA_AGENT_SDK_MAX_TURNS", "10")
        cfg = AgentSDKConfig()
        cfg.apply_env_overrides()
        assert cfg.max_turns == 10

    def test_budget_override_from_env(self, monkeypatch):
        monkeypatch.setenv("AURA_AGENT_SDK_MAX_BUDGET", "5.0")
        cfg = AgentSDKConfig()
        cfg.apply_env_overrides()
        assert cfg.max_budget_usd == 5.0

    def test_permission_mode_override_from_env(self, monkeypatch):
        monkeypatch.setenv("AURA_AGENT_SDK_PERMISSION_MODE", "bypassPermissions")
        cfg = AgentSDKConfig()
        cfg.apply_env_overrides()
        assert cfg.permission_mode == "bypassPermissions"

    def test_no_env_vars_no_change(self, monkeypatch):
        for var in ("AURA_AGENT_SDK_MODEL", "AURA_AGENT_SDK_MAX_TURNS",
                    "AURA_AGENT_SDK_MAX_BUDGET", "AURA_AGENT_SDK_PERMISSION_MODE"):
            monkeypatch.delenv(var, raising=False)
        cfg = AgentSDKConfig()
        cfg.apply_env_overrides()
        assert cfg.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestAgentSDKConfigProperties:
    def test_mcp_server_endpoints_returns_urls(self):
        cfg = AgentSDKConfig()
        endpoints = cfg.mcp_server_endpoints
        assert "dev_tools" in endpoints
        assert endpoints["dev_tools"] == "http://localhost:8001"

    def test_mcp_server_endpoints_all_keys(self):
        cfg = AgentSDKConfig()
        endpoints = cfg.mcp_server_endpoints
        for name in cfg.mcp_ports:
            assert name in endpoints

    def test_thinking_config_enabled(self):
        cfg = AgentSDKConfig(enable_thinking=True)
        assert cfg.thinking_config == {"type": "adaptive"}

    def test_thinking_config_disabled(self):
        cfg = AgentSDKConfig(enable_thinking=False)
        assert cfg.thinking_config is None


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_default_allowed_tools_has_required(self):
        for tool in ("Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"):
            assert tool in _DEFAULT_ALLOWED_TOOLS

    def test_default_mcp_ports_has_required(self):
        assert "dev_tools" in _DEFAULT_MCP_PORTS
        assert "sadd" in _DEFAULT_MCP_PORTS
        assert _DEFAULT_MCP_PORTS["sadd"] == 8020
