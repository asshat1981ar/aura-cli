"""Unit tests for agents/mcp_health_agent.py."""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock, AsyncMock


class TestUpdateRegistryHealth(unittest.TestCase):
    """Tests for the module-level _update_registry_health() helper."""

    def _make_registry(self, agents=None):
        registry = MagicMock()
        registry.list_agents.return_value = agents or []
        return registry

    def _make_spec(self, name, source, mcp_server):
        spec = MagicMock()
        spec.name = name
        spec.source = source
        spec.mcp_server = mcp_server
        return spec

    @patch("agents.mcp_health_agent.agent_registry", create=True)
    def test_healthy_server_marks_agent_healthy(self, mock_registry):
        from agents.mcp_health_agent import _update_registry_health
        import agents.mcp_health_agent as mod

        spec = self._make_spec("my-agent", "mcp", "server-a")
        mock_registry.list_agents.return_value = [spec]

        with patch.object(mod, "_update_registry_health", wraps=lambda r: None):
            pass

        # Directly patch core.mcp_agent_registry inside the function's import
        with patch.dict("sys.modules", {"core.mcp_agent_registry": MagicMock(agent_registry=mock_registry)}):
            _update_registry_health({"server-a": {"status": "healthy"}})

        mock_registry.mark_healthy.assert_called_once_with("my-agent")
        mock_registry.mark_unhealthy.assert_not_called()

    @patch("agents.mcp_health_agent.agent_registry", create=True)
    def test_unhealthy_server_marks_agent_unhealthy(self, mock_registry):
        from agents.mcp_health_agent import _update_registry_health

        spec = self._make_spec("my-agent", "mcp", "server-b")
        mock_registry.list_agents.return_value = [spec]

        with patch.dict("sys.modules", {"core.mcp_agent_registry": MagicMock(agent_registry=mock_registry)}):
            _update_registry_health({"server-b": {"status": "unhealthy"}})

        mock_registry.mark_unhealthy.assert_called_once_with("my-agent")

    @patch("agents.mcp_health_agent.agent_registry", create=True)
    def test_non_mcp_agent_skipped(self, mock_registry):
        from agents.mcp_health_agent import _update_registry_health

        spec = self._make_spec("local-agent", "local", "server-a")
        mock_registry.list_agents.return_value = [spec]

        with patch.dict("sys.modules", {"core.mcp_agent_registry": MagicMock(agent_registry=mock_registry)}):
            _update_registry_health({"server-a": {"status": "healthy"}})

        mock_registry.mark_healthy.assert_not_called()
        mock_registry.mark_unhealthy.assert_not_called()


class TestMCPHealthAgent(unittest.TestCase):
    """Tests for MCPHealthAgent.run()."""

    def _make_agent(self):
        from agents.mcp_health_agent import MCPHealthAgent

        return MCPHealthAgent()

    def test_agent_metadata(self):
        from agents.mcp_health_agent import MCPHealthAgent

        self.assertEqual(MCPHealthAgent.name, "mcp_health")
        self.assertIn("health", MCPHealthAgent.description.lower())

    def test_run_returns_success_on_good_check(self):
        import sys

        expected = {"status": "success", "results": [], "summary": {"healthy": 2}}
        mock_anyio = MagicMock()
        mock_anyio.run.return_value = expected
        mock_asyncio = MagicMock()
        mock_asyncio.get_running_loop.side_effect = RuntimeError

        with patch.dict(sys.modules, {"anyio": mock_anyio, "asyncio": mock_asyncio}):
            # Re-import to pick up patched modules
            import importlib
            import agents.mcp_health_agent as mod

            importlib.reload(mod)
            agent = mod.MCPHealthAgent()
            result = agent.run({})

        self.assertEqual(result["status"], "success")

    def test_run_returns_error_dict_on_exception(self):
        import sys

        mock_anyio = MagicMock()
        mock_anyio.run.side_effect = Exception("connection refused")
        mock_asyncio = MagicMock()
        mock_asyncio.get_running_loop.side_effect = RuntimeError

        with patch.dict(sys.modules, {"anyio": mock_anyio, "asyncio": mock_asyncio}):
            import importlib
            import agents.mcp_health_agent as mod

            importlib.reload(mod)
            agent = mod.MCPHealthAgent()
            result = agent.run({})

        self.assertEqual(result["status"], "error")
        self.assertIn("connection refused", result["error"])


if __name__ == "__main__":
    unittest.main()
