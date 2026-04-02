# tests/test_agent_sdk_config.py
"""Tests for Agent SDK configuration."""
import unittest
from unittest.mock import patch
import os


class TestAgentSDKConfig(unittest.TestCase):
    """Test Agent SDK configuration loading and defaults."""

    def test_default_config_has_required_fields(self):
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        self.assertEqual(config.model, "claude-sonnet-4-6")
        self.assertIsInstance(config.max_turns, int)
        self.assertGreater(config.max_turns, 0)
        self.assertIsInstance(config.max_budget_usd, float)
        self.assertIn(config.permission_mode, ("default", "acceptEdits", "bypassPermissions", "plan"))

    def test_config_from_aura_config(self):
        from core.agent_sdk.config import AgentSDKConfig

        aura_config = {
            "agent_sdk": {
                "model": "claude-opus-4-6",
                "max_turns": 50,
                "max_budget_usd": 5.0,
                "permission_mode": "acceptEdits",
            }
        }
        config = AgentSDKConfig.from_aura_config(aura_config)
        self.assertEqual(config.model, "claude-opus-4-6")
        self.assertEqual(config.max_turns, 50)
        self.assertEqual(config.max_budget_usd, 5.0)

    def test_config_tool_allowlist_defaults(self):
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        self.assertIn("Read", config.allowed_tools)
        self.assertIn("Edit", config.allowed_tools)
        self.assertIn("Bash", config.allowed_tools)
        self.assertIn("Glob", config.allowed_tools)
        self.assertIn("Grep", config.allowed_tools)
        self.assertIn("Agent", config.allowed_tools)

    def test_config_env_override(self):
        from core.agent_sdk.config import AgentSDKConfig

        with patch.dict(os.environ, {"AURA_AGENT_SDK_MODEL": "claude-opus-4-6"}):
            config = AgentSDKConfig()
            config.apply_env_overrides()
            self.assertEqual(config.model, "claude-opus-4-6")

    def test_config_mcp_server_endpoints(self):
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        endpoints = config.mcp_server_endpoints
        self.assertIsInstance(endpoints, dict)
        self.assertIn("dev_tools", endpoints)
        self.assertIn("skills", endpoints)
        self.assertIn("control", endpoints)


if __name__ == "__main__":
    unittest.main()
