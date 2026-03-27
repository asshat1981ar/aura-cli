import unittest
from unittest.mock import patch, MagicMock
import os
import json
from core.sadd.mcp_tool_bridge import MCPToolBridge

class TestSADDMCPDiscovery(unittest.TestCase):
    def setUp(self):
        self.bridge = MCPToolBridge()

    @patch('requests.post')
    def test_semantic_match_success(self, mock_post):
        # Mock a successful response from the discovery server
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "results": [
                    {"name": "security_scanner", "server": "dev_tools"},
                    {"name": "vulnerability_check", "server": "security_mcp"}
                ]
            }
        }
        mock_post.return_value = mock_response
        
        results = self.bridge.match_tools_for_goal("I need to find security vulnerabilities")
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["matched_semantic"])
        self.assertEqual(results[0]["name"], "security_scanner")

    @patch('requests.post')
    def test_semantic_match_fallback(self, mock_post):
        # Mock a failed connection or empty response to trigger fallback
        mock_post.side_effect = Exception("Connection refused")
        
        results = self.bridge.match_tools_for_goal("lint some files")
        
        # Should fallback to the static _GOAL_TOOL_MAP for 'lint'
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["matched_keyword"], "lint")
        self.assertEqual(results[0]["name"], "linter_enforcer")

    @patch('requests.post')
    def test_discover_available_tools_discovery_server(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "tools": [
                    {"name": "tool1", "server": "server1"},
                    {"name": "tool2", "server": "server2"}
                ]
            }
        }
        mock_post.return_value = mock_response
        
        tools = self.bridge.discover_available_tools()
        
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "tool1")

if __name__ == '__main__':
    unittest.main()
