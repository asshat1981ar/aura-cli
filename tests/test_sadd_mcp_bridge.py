import unittest
from unittest.mock import patch, MagicMock
from core.sadd.mcp_tool_bridge import MCPToolBridge


class TestMCPToolBridge(unittest.TestCase):
    def test_discover_tools_returns_list(self):
        bridge = MCPToolBridge()
        tools = bridge.discover_available_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_match_security_goal(self):
        bridge = MCPToolBridge()
        matched = bridge.match_tools_for_goal("Run security scan on auth module")
        names = [t["name"] for t in matched]
        self.assertIn("security_scanner", names)

    def test_match_test_goal(self):
        bridge = MCPToolBridge()
        matched = bridge.match_tools_for_goal("Add unit tests for parser")
        names = [t["name"] for t in matched]
        self.assertIn("test_coverage_analyzer", names)

    def test_match_no_keywords(self):
        bridge = MCPToolBridge()
        matched = bridge.match_tools_for_goal("Do something generic")
        self.assertEqual(len(matched), 0)

    def test_build_tool_context(self):
        bridge = MCPToolBridge()
        ctx = bridge.build_tool_context(["security_scanner", "linter_enforcer"])
        self.assertIn("available_tools", ctx)
        self.assertEqual(len(ctx["available_tools"]), 2)
        self.assertEqual(ctx["available_tools"][0]["name"], "security_scanner")

    @patch("core.sadd.mcp_tool_bridge.requests.post")
    def test_build_tool_context_preserves_match_metadata(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "results": [
                    {
                        "name": "security_scanner",
                        "server": "dev_tools",
                        "description": "scan for vulnerabilities",
                        "score": 0.91,
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        bridge = MCPToolBridge()
        matched = bridge.match_tools_for_goal("scan for vulnerabilities")
        ctx = bridge.build_tool_context(matched)

        self.assertEqual(ctx["available_tools"][0]["match_source"], "semantic_discovery")
        self.assertTrue(ctx["available_tools"][0]["matched_semantic"])
        self.assertEqual(ctx["available_tools"][0]["server"], "dev_tools")

    def test_no_duplicate_matches(self):
        bridge = MCPToolBridge()
        matched = bridge.match_tools_for_goal("lint and format the code with linting tools")
        names = [t["name"] for t in matched]
        self.assertEqual(len(names), len(set(names)))
