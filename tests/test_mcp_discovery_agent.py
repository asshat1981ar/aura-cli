"""Unit tests for agents/mcp_discovery_agent.py."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestMCPDiscoveryAgent(unittest.TestCase):
    """Tests for MCPDiscoveryAgent.run() and _register_mcp_agents_for_discovered()."""

    def _make_agent(self, config_path=".mcp.json"):
        from agents.mcp_discovery_agent import MCPDiscoveryAgent

        return MCPDiscoveryAgent(config_path=config_path)

    def _write_config(self, tmpdir, data):
        config_file = Path(tmpdir) / ".mcp.json"
        config_file.write_text(json.dumps(data))
        return tmpdir

    def test_agent_name(self):
        from agents.mcp_discovery_agent import MCPDiscoveryAgent

        self.assertEqual(MCPDiscoveryAgent.name, "mcp_discovery")

    def test_default_config_path(self):
        agent = self._make_agent()
        self.assertEqual(agent.config_path, ".mcp.json")

    def test_missing_config_returns_success_with_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent()
            result = agent.run({"project_root": tmpdir})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["discovered"], [])
        self.assertIn("No", result["message"])

    def test_valid_config_discovers_servers_with_command(self):
        config = {
            "mcpServers": {
                "my-server": {"command": "python", "args": ["-m", "my_server"]},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_config(tmpdir, config)
            agent = self._make_agent()
            with patch.object(agent, "_register_mcp_agents_for_discovered"):
                result = agent.run({"project_root": tmpdir})

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["discovered"]), 1)
        self.assertEqual(result["discovered"][0]["name"], "my-server")
        self.assertEqual(result["discovered"][0]["status"], "configured")

    def test_server_without_command_is_skipped(self):
        config = {
            "mcpServers": {
                "bad-server": {"url": "http://localhost:9000"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_config(tmpdir, config)
            agent = self._make_agent()
            with patch("agents.mcp_discovery_agent.log_json") as mock_log, patch.object(agent, "_register_mcp_agents_for_discovered"):
                result = agent.run({"project_root": tmpdir})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["discovered"], [])
        mock_log.assert_called_once()

    def test_invalid_json_returns_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / ".mcp.json"
            config_file.write_text("{ not valid json }")
            agent = self._make_agent()
            result = agent.run({"project_root": tmpdir})

        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid", result["error"])
        self.assertEqual(result["discovered"], [])

    def test_multiple_servers_all_discovered(self):
        config = {
            "mcpServers": {
                "server-a": {"command": "node", "args": ["a.js"]},
                "server-b": {"command": "python", "args": ["b.py"]},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_config(tmpdir, config)
            agent = self._make_agent()
            with patch.object(agent, "_register_mcp_agents_for_discovered"):
                result = agent.run({"project_root": tmpdir})

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["discovered"]), 2)
        names = {s["name"] for s in result["discovered"]}
        self.assertEqual(names, {"server-a", "server-b"})

    def test_register_called_after_discovery(self):
        config = {"mcpServers": {"s": {"command": "run"}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_config(tmpdir, config)
            agent = self._make_agent()
            with patch.object(agent, "_register_mcp_agents_for_discovered") as mock_reg:
                agent.run({"project_root": tmpdir})

        mock_reg.assert_called_once()


if __name__ == "__main__":
    unittest.main()
