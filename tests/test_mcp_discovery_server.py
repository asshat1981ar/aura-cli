"""Unit tests for tools/mcp_discovery_server.py."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from tools import mcp_discovery_server as discovery


class TestMCPDiscoveryServer(unittest.TestCase):
    def test_extract_tools_payload_supports_plain_list(self):
        payload = [{"name": "tool_a"}, {"name": "tool_b"}]
        tools = discovery._extract_tools_payload(payload)
        self.assertEqual([tool["name"] for tool in tools], ["tool_a", "tool_b"])

    def test_extract_tools_payload_supports_wrapped_payload(self):
        payload = {"data": {"tools": [{"name": "tool_a"}]}}
        tools = discovery._extract_tools_payload(payload)
        self.assertEqual([tool["name"] for tool in tools], ["tool_a"])

    @patch("tools.mcp_discovery_server.requests.get")
    @patch("tools.mcp_discovery_server.get_mcp_config")
    def test_list_all_tools_normalizes_multiple_response_shapes(self, mock_config, mock_get):
        mock_config.return_value = {"sadd": 8020, "skills": 8010, "discovery": discovery.DISCOVERY_PORT}

        sadd_response = MagicMock()
        sadd_response.status_code = 200
        sadd_response.json.return_value = [
            {"name": "sadd_parse_spec", "description": "parse specs"},
        ]

        skills_response = MagicMock()
        skills_response.status_code = 200
        skills_response.json.return_value = {
            "data": {
                "tools": [
                    {"name": "security_scanner", "description": "scan security issues"},
                ]
            }
        }
        mock_get.side_effect = [sadd_response, skills_response]

        tools = discovery.list_all_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["server"], "sadd")
        self.assertEqual(tools[0]["port"], 8020)
        self.assertEqual(tools[1]["server"], "skills")
        self.assertEqual(tools[1]["name"], "security_scanner")

    @patch("tools.mcp_discovery_server.requests.get")
    @patch("tools.mcp_discovery_server.get_mcp_config")
    def test_list_all_tools_skips_offline_servers(self, mock_config, mock_get):
        mock_config.return_value = {"sadd": 8020}
        mock_get.side_effect = RuntimeError("offline")
        self.assertEqual(discovery.list_all_tools(), [])

    @patch("tools.mcp_discovery_server.list_all_tools")
    def test_search_tools_semantically_orders_best_match_first(self, mock_list_all_tools):
        mock_list_all_tools.return_value = [
            {"name": "lint_files", "description": "lint project files", "server": "skills", "port": 8010},
            {"name": "security_scanner", "description": "audit security vulnerabilities", "server": "dev_tools", "port": 8001},
        ]

        response = asyncio.run(
            discovery.call_tool(
                discovery.CallRequest(
                    tool_name="search_tools_semantically",
                    args={"query": "audit security vulnerabilities", "top_k": 2},
                )
            )
        )

        data = response["data"]
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["name"], "security_scanner")
        self.assertGreater(data["results"][0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
