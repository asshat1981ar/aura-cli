# tests/test_agent_sdk_controller.py
"""Tests for Agent SDK meta-controller."""
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


class TestControllerInit(unittest.TestCase):
    """Test controller initialization."""

    def test_create_controller(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        controller = AuraController(
            config=config,
            project_root=Path("/tmp/test"),
        )
        self.assertIsNotNone(controller)
        self.assertEqual(controller.project_root, Path("/tmp/test"))

    def test_controller_builds_options(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig(model="claude-sonnet-4-6", max_turns=20)
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        options = controller._build_options(goal="Test goal")
        self.assertEqual(options.model, "claude-sonnet-4-6")
        self.assertEqual(options.max_turns, 20)

    def test_controller_builds_system_prompt(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        prompt = controller._build_prompt(goal="Fix login bug")
        self.assertIn("Fix login bug", prompt)
        self.assertIn("AURA", prompt)

    def test_controller_registers_subagents(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig(enable_subagents=True)
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        agents = controller._build_subagent_defs()
        self.assertIn("planning-agent", agents)
        self.assertIn("implementation-agent", agents)

    def test_controller_no_subagents_when_disabled(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig(enable_subagents=False)
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        agents = controller._build_subagent_defs()
        self.assertEqual(agents, {})


class TestControllerMCPServer(unittest.TestCase):
    """Test that the controller creates an MCP server with AURA tools."""

    def test_mcp_server_created(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        server = controller._build_mcp_server()
        self.assertIsNotNone(server)


if __name__ == "__main__":
    unittest.main()
