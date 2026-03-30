"""Integration tests for core.sadd.mcp_tool_bridge — MCPToolBridge with mocked clients."""

import unittest
from unittest.mock import MagicMock, patch

from core.sadd.mcp_tool_bridge import MCPToolBridge, _GOAL_TOOL_MAP


class TestMCPToolBridgeDiscovery(unittest.TestCase):
    """Tests for MCPToolBridge.discover_available_tools()."""

    def test_static_fallback_when_no_server(self):
        """When no discovery server is reachable, returns static tool list."""
        bridge = MCPToolBridge()
        tools = bridge.discover_available_tools()
        self.assertIsInstance(tools, list)
        self.assertTrue(len(tools) > 0)
        # Each entry must have a 'name' key
        for tool in tools:
            self.assertIn("name", tool)

    def test_static_fallback_server_source(self):
        """Static fallback tools are tagged with server='static'."""
        bridge = MCPToolBridge()
        # Ensure requests always fail so we use the static fallback
        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            tools = bridge.discover_available_tools()
        for tool in tools:
            self.assertEqual(tool.get("server"), "static")

    def test_uses_discovery_server_when_available(self):
        """When discovery server responds with tools, they are returned."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "tools": [
                    {"name": "my_tool", "server": "discovery"},
                    {"name": "other_tool", "server": "discovery"},
                ]
            }
        }

        with patch("core.sadd.mcp_tool_bridge.requests.post", return_value=mock_resp):
            bridge = MCPToolBridge()
            tools = bridge.discover_available_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "my_tool")
        self.assertEqual(tools[1]["name"], "other_tool")

    def test_falls_back_to_static_when_discovery_returns_empty(self):
        """Discovery server returning empty tools list triggers static fallback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"tools": []}}

        with patch("core.sadd.mcp_tool_bridge.requests.post", return_value=mock_resp):
            bridge = MCPToolBridge()
            tools = bridge.discover_available_tools()

        # Should fall through to static list
        self.assertTrue(len(tools) > 0)

    def test_falls_back_when_discovery_returns_non_200(self):
        """Non-200 response from discovery server triggers static fallback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("core.sadd.mcp_tool_bridge.requests.post", return_value=mock_resp):
            bridge = MCPToolBridge()
            tools = bridge.discover_available_tools()

        self.assertTrue(len(tools) > 0)

    def test_falls_back_to_static_when_registry_import_fails(self):
        """When registry is provided but import fails, static fallback is returned."""
        mock_registry = MagicMock()

        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            bridge = MCPToolBridge(mcp_registry=mock_registry)
            # core.mcp_registry doesn't exist yet, so ImportError is caught internally
            tools = bridge.discover_available_tools()

        # Should still get tools from the static fallback
        self.assertTrue(len(tools) > 0)
        for tool in tools:
            self.assertIn("name", tool)


class TestMCPToolBridgeMatchTools(unittest.TestCase):
    """Tests for MCPToolBridge.match_tools_for_goal()."""

    def setUp(self):
        self.bridge = MCPToolBridge()

    def test_keyword_match_test(self):
        """Goal text containing 'test' returns test-related tools."""
        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            tools = self.bridge.match_tools_for_goal("Write unit test coverage for auth module")

        names = [t["name"] for t in tools]
        self.assertTrue(len(tools) > 0)
        # All matched_keyword entries should reference 'test'
        for t in tools:
            if "matched_keyword" in t:
                self.assertIn(t["matched_keyword"], _GOAL_TOOL_MAP)

    def test_keyword_match_security(self):
        """Goal text containing 'security' returns security scanner tools."""
        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            tools = self.bridge.match_tools_for_goal("Run a security audit on the codebase")

        names = [t["name"] for t in tools]
        self.assertIn("security_scanner", names)

    def test_keyword_match_lint(self):
        """Goal text containing 'lint' returns linter tools."""
        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            tools = self.bridge.match_tools_for_goal("lint the Python source files")

        names = [t["name"] for t in tools]
        self.assertTrue(any("lint" in n for n in names))

    def test_no_keyword_match_returns_empty(self):
        """Goal text with no matching keywords returns empty list."""
        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            tools = self.bridge.match_tools_for_goal("Do something completely unrelated to known keywords xyz123")

        self.assertEqual(tools, [])

    def test_no_duplicate_tools_returned(self):
        """Even if multiple keywords match, each tool is returned at most once."""
        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            tools = self.bridge.match_tools_for_goal("lint test lint test lint")

        names = [t["name"] for t in tools]
        self.assertEqual(len(names), len(set(names)))

    def test_uses_semantic_discovery_when_available(self):
        """When semantic discovery returns results, they take priority."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "results": [
                    {"name": "semantic_tool_1", "server": "discovery"},
                    {"name": "semantic_tool_2", "server": "discovery"},
                ]
            }
        }

        with patch("core.sadd.mcp_tool_bridge.requests.post", return_value=mock_resp):
            tools = self.bridge.match_tools_for_goal("lint the code")

        names = [t["name"] for t in tools]
        self.assertIn("semantic_tool_1", names)
        self.assertIn("semantic_tool_2", names)
        # Semantic results have matched_semantic=True
        for t in tools:
            self.assertTrue(t.get("matched_semantic"))

    def test_falls_back_to_keyword_when_semantic_returns_empty(self):
        """Empty semantic results fall back to keyword matching."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"results": []}}

        with patch("core.sadd.mcp_tool_bridge.requests.post", return_value=mock_resp):
            tools = self.bridge.match_tools_for_goal("lint the source code")

        names = [t["name"] for t in tools]
        self.assertTrue(any("lint" in n for n in names))


class TestMCPToolBridgeBuildContext(unittest.TestCase):
    """Tests for MCPToolBridge.build_tool_context()."""

    def setUp(self):
        self.bridge = MCPToolBridge()

    def test_returns_context_dict(self):
        tool_names = ["linter_enforcer", "type_checker"]
        ctx = self.bridge.build_tool_context(tool_names)
        self.assertIsInstance(ctx, dict)

    def test_available_tools_key_present(self):
        tool_names = ["security_scanner", "lint_files"]
        ctx = self.bridge.build_tool_context(tool_names)
        self.assertIn("available_tools", ctx)
        self.assertEqual(len(ctx["available_tools"]), 2)

    def test_tool_entries_have_name_and_type(self):
        ctx = self.bridge.build_tool_context(["my_tool"])
        entry = ctx["available_tools"][0]
        self.assertEqual(entry["name"], "my_tool")
        self.assertEqual(entry["type"], "mcp_tool")

    def test_discovery_source_label(self):
        ctx = self.bridge.build_tool_context(["foo"])
        self.assertEqual(ctx["tool_discovery_source"], "sadd_mcp_bridge")

    def test_empty_tool_list(self):
        ctx = self.bridge.build_tool_context([])
        self.assertEqual(ctx["available_tools"], [])

    def test_error_handling_for_missing_tool(self):
        """build_tool_context with unknown tool names should still succeed (no lookup)."""
        ctx = self.bridge.build_tool_context(["completely_nonexistent_tool"])
        self.assertEqual(len(ctx["available_tools"]), 1)
        self.assertEqual(ctx["available_tools"][0]["name"], "completely_nonexistent_tool")


class TestMCPToolBridgeEndToEnd(unittest.TestCase):
    """Higher-level integration: discover -> match -> build context pipeline."""

    def test_full_pipeline_with_mocked_discovery(self):
        """Full pipeline: discover tools, match for goal, build context."""
        mock_discover_resp = MagicMock()
        mock_discover_resp.status_code = 200
        mock_discover_resp.json.return_value = {
            "data": {
                "tools": [
                    {"name": "linter_enforcer", "server": "mcp"},
                    {"name": "type_checker", "server": "mcp"},
                    {"name": "security_scanner", "server": "mcp"},
                ]
            }
        }

        with patch("core.sadd.mcp_tool_bridge.requests.post", return_value=mock_discover_resp):
            bridge = MCPToolBridge()
            available = bridge.discover_available_tools()

        self.assertEqual(len(available), 3)

        # Now match for a linting goal (semantic call will also be mocked to fail)
        with patch("core.sadd.mcp_tool_bridge.requests.post", side_effect=ConnectionError()):
            matched = bridge.match_tools_for_goal("lint and type-check the codebase")

        matched_names = [t["name"] for t in matched]
        self.assertTrue(len(matched_names) > 0)

        ctx = bridge.build_tool_context(matched_names)
        self.assertEqual(len(ctx["available_tools"]), len(matched_names))


if __name__ == "__main__":
    unittest.main()
