"""Unit tests for tools/sadd_mcp_server.py.

These tests intentionally avoid FastAPI's TestClient. In this environment,
TestClient request execution hangs even for trivial apps, so we validate the
server handlers directly instead.
"""

from __future__ import annotations

import asyncio
import unittest

from tools.sadd_mcp_server import (
    ToolCallRequest,
    _build_descriptor,
    call_tool,
    get_tool,
    health,
    list_tools,
)


class TestSADDMCPServer(unittest.TestCase):
    def test_health_endpoint(self):
        data = asyncio.run(health())
        self.assertEqual(data["server"], "sadd_mcp")
        self.assertIn("tool_count", data)

    def test_tools_endpoint(self):
        data = asyncio.run(list_tools())
        tool_names = [t["name"] for t in data]
        self.assertIn("sadd_parse_spec", tool_names)
        self.assertIn("sadd_session_status", tool_names)
        self.assertIn("sadd_list_sessions", tool_names)

    def test_get_tool_descriptor(self):
        data = asyncio.run(get_tool("sadd_parse_spec"))
        self.assertEqual(data["name"], "sadd_parse_spec")
        self.assertIn("inputSchema", data)

    def test_build_descriptor_required_fields(self):
        descriptor = _build_descriptor("sadd_session_status")
        self.assertEqual(descriptor["name"], "sadd_session_status")
        self.assertIn("session_id", descriptor["inputSchema"]["required"])

    def test_call_parse_spec(self):
        spec_md = "# Test Spec\n## Task: Do thing\n- Build it\n- Test it"
        resp = asyncio.run(
            call_tool(
                ToolCallRequest(
                    tool_name="sadd_parse_spec",
                    args={"spec_text": spec_md},
                )
            )
        )
        data = resp.result
        self.assertEqual(data["title"], "Test Spec")
        self.assertGreaterEqual(data["workstream_count"], 1)

    def test_call_list_sessions(self):
        resp = asyncio.run(
            call_tool(
                ToolCallRequest(
                    tool_name="sadd_list_sessions",
                    args={},
                )
            )
        )
        self.assertEqual(resp.tool_name, "sadd_list_sessions")
        self.assertIn("sessions", resp.result)

    def test_call_unknown_tool_raises(self):
        with self.assertRaises(Exception) as ctx:
            asyncio.run(
                call_tool(
                    ToolCallRequest(
                        tool_name="nonexistent_tool",
                        args={},
                    )
                )
            )
        self.assertIn("not found", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
