import unittest
from fastapi.testclient import TestClient

class TestSADDMCPServer(unittest.TestCase):
    def setUp(self):
        from tools.sadd_mcp_server import create_app
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["server"], "aura-sadd")
        self.assertIn("tools", data)

    def test_tools_endpoint(self):
        resp = self.client.get("/tools")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        tool_names = [t["name"] for t in data]
        self.assertIn("sadd_parse_spec", tool_names)
        self.assertIn("sadd_session_status", tool_names)
        self.assertIn("sadd_list_sessions", tool_names)

    def test_call_parse_spec(self):
        spec_md = "# Test Spec\n## Task: Do thing\n- Build it\n- Test it"
        resp = self.client.post("/call", json={
            "tool_name": "sadd_parse_spec",
            "args": {"markdown": spec_md}
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["title"], "Test Spec")
        self.assertGreaterEqual(data["workstreams"], 1)

    def test_call_list_sessions(self):
        resp = self.client.post("/call", json={
            "tool_name": "sadd_list_sessions",
            "args": {}
        })
        self.assertEqual(resp.status_code, 200)

    def test_call_unknown_tool(self):
        resp = self.client.post("/call", json={
            "tool_name": "nonexistent_tool",
            "args": {}
        })
        self.assertEqual(resp.status_code, 404)
