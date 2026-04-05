"""Tests for the MCP health dashboard and BEADS schemas CLI commands.

Covers:
  aura mcp status   → _handle_mcp_status_dispatch
  aura mcp restart  → _handle_mcp_restart_dispatch
  aura beads schemas → _handle_beads_schemas_dispatch
"""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aura_cli.cli_options import parse_cli_args
import aura_cli.cli_main as cli_main


class TestMCPStatusDispatch(unittest.TestCase):
    """aura mcp status — renders health dashboard for all MCP servers."""

    def _dispatch(self, argv):
        parsed = parse_cli_args(argv)
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=MagicMock())
        return code, out.getvalue(), err.getvalue()

    def test_mcp_status_all_healthy_returns_zero(self):
        healthy_results = [
            {"name": "dev_tools", "status": "healthy", "port": 8001, "health_data": {"timestamp": "now", "tool_count": 5}},
            {"name": "skills", "status": "healthy", "port": 8002, "health_data": {}},
        ]
        with patch("core.mcp_health.check_all_mcp_health", new=AsyncMock(return_value=healthy_results)), \
             patch("core.mcp_health.get_health_summary", return_value={"total_servers": 2, "healthy_count": 2, "unhealthy_count": 0, "all_healthy": True}), \
             patch("core.mcp_registry.list_registered_services", return_value=[{"config_name": "dev_tools", "url": "http://127.0.0.1:8001"}, {"config_name": "skills", "url": "http://127.0.0.1:8002"}]):
            code, out, err = self._dispatch(["mcp", "status"])
        self.assertEqual(code, 0)

    def test_mcp_status_unhealthy_returns_nonzero(self):
        results = [
            {"name": "dev_tools", "status": "unhealthy", "error": "refused", "health_data": None},
        ]
        with patch("core.mcp_health.check_all_mcp_health", new=AsyncMock(return_value=results)), \
             patch("core.mcp_health.get_health_summary", return_value={"total_servers": 1, "healthy_count": 0, "unhealthy_count": 1, "all_healthy": False}), \
             patch("core.mcp_registry.list_registered_services", return_value=[{"config_name": "dev_tools", "url": "http://127.0.0.1:8001"}]):
            code, out, err = self._dispatch(["mcp", "status"])
        self.assertEqual(code, 1)

    def test_mcp_status_json_output(self):
        results = [{"name": "dev_tools", "status": "healthy", "health_data": {}}]
        with patch("core.mcp_health.check_all_mcp_health", new=AsyncMock(return_value=results)), \
             patch("core.mcp_health.get_health_summary", return_value={"total_servers": 1, "healthy_count": 1, "unhealthy_count": 0, "all_healthy": True}), \
             patch("core.mcp_registry.list_registered_services", return_value=[]):
            code, out, err = self._dispatch(["mcp", "status", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("servers", payload)
        self.assertIn("summary", payload)


class TestMCPRestartDispatch(unittest.TestCase):
    """aura mcp restart <server> — validate a named MCP server."""

    def _dispatch(self, argv):
        parsed = parse_cli_args(argv)
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=MagicMock())
        return code, out.getvalue(), err.getvalue()

    def test_restart_healthy_server_returns_zero(self):
        health_result = {"name": "dev_tools", "status": "healthy", "port": 8001, "health_data": {}}
        svc = {"name": "aura-dev-tools", "url": "http://127.0.0.1:8001", "config_name": "dev_tools"}
        with patch("core.mcp_health.check_mcp_health", new=AsyncMock(return_value=health_result)), \
             patch("core.mcp_registry.get_registered_service", return_value=svc):
            code, out, err = self._dispatch(["mcp", "restart", "dev_tools"])
        self.assertEqual(code, 0)

    def test_restart_unhealthy_server_returns_one(self):
        health_result = {"name": "dev_tools", "status": "unhealthy", "error": "connection refused"}
        svc = {"name": "aura-dev-tools", "url": "http://127.0.0.1:8001", "config_name": "dev_tools"}
        with patch("core.mcp_health.check_mcp_health", new=AsyncMock(return_value=health_result)), \
             patch("core.mcp_registry.get_registered_service", return_value=svc):
            code, out, err = self._dispatch(["mcp", "restart", "dev_tools"])
        self.assertEqual(code, 1)

    def test_restart_unknown_server_returns_one(self):
        with patch("core.mcp_registry.get_registered_service", side_effect=KeyError("unknown")):
            code, out, err = self._dispatch(["mcp", "restart", "unknown_server"])
        self.assertEqual(code, 1)

    def test_restart_json_output(self):
        health_result = {"name": "dev_tools", "status": "healthy", "health_data": {}}
        svc = {"name": "aura-dev-tools", "url": "http://127.0.0.1:8001", "config_name": "dev_tools"}
        with patch("core.mcp_health.check_mcp_health", new=AsyncMock(return_value=health_result)), \
             patch("core.mcp_registry.get_registered_service", return_value=svc):
            code, out, err = self._dispatch(["mcp", "restart", "dev_tools", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("server", payload)
        self.assertIn("result", payload)


class TestBeadsSchemasDispatch(unittest.TestCase):
    """aura beads schemas — list registered BEADS schema contracts."""

    def _dispatch(self, argv):
        parsed = parse_cli_args(argv)
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=MagicMock())
        return code, out.getvalue(), err.getvalue()

    def test_beads_schemas_returns_zero(self):
        code, out, err = self._dispatch(["beads", "schemas"])
        self.assertEqual(code, 0)

    def test_beads_schemas_json_output_has_schemas(self):
        code, out, err = self._dispatch(["beads", "schemas", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("schemas", payload)
        self.assertIn("schema_version", payload)
        names = [s["name"] for s in payload["schemas"]]
        self.assertIn("BeadsInput", names)
        self.assertIn("BeadsDecision", names)
        self.assertIn("BeadsResult", names)

    def test_beads_schemas_json_has_interaction_count(self):
        code, out, err = self._dispatch(["beads", "schemas", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("interaction_count", payload)
        self.assertIsInstance(payload["interaction_count"], int)

    def test_beads_schemas_json_schema_has_fields(self):
        code, out, err = self._dispatch(["beads", "schemas", "--json"])
        payload = json.loads(out)
        for schema in payload["schemas"]:
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertIn("fields", schema)
            self.assertIsInstance(schema["fields"], list)
