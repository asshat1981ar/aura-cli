"""Unit tests for core/mcp_agent_registry.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.types import AgentSpec, MCPServerConfig
from core.mcp_agent_registry import TypedAgentRegistry, _resolve_mcp_capabilities, agent_registry


def _make_spec(name: str, caps: list[str], source: str = "local") -> AgentSpec:
    return AgentSpec(name=name, description=f"{name} agent", capabilities=caps, source=source)


class TestResolveMcpCapabilities:
    def test_own_name_always_first(self):
        caps = _resolve_mcp_capabilities("git_blame")
        assert caps[0] == "git_blame"

    def test_group_included_as_secondary(self):
        caps = _resolve_mcp_capabilities("git_blame")
        assert "git" in caps
        assert "git_analysis" in caps  # git_blame also belongs to git_analysis group

    def test_tool_in_no_group_returns_only_itself(self):
        caps = _resolve_mcp_capabilities("completely_unknown_tool_xyz")
        assert caps == ["completely_unknown_tool_xyz"]

    def test_no_duplicates_in_result(self):
        caps = _resolve_mcp_capabilities("git_blame")
        assert len(caps) == len(set(caps))

    def test_file_system_tool(self):
        caps = _resolve_mcp_capabilities("read_file")
        assert "read_file" in caps
        assert "file_system" in caps


class TestTypedAgentRegistryRegister:
    def setup_method(self):
        self.registry = TypedAgentRegistry()

    @patch("core.mcp_agent_registry.log_json")
    def test_register_agent(self, _):
        spec = _make_spec("coder", ["code_generation"])
        self.registry.register(spec)
        assert self.registry.get_agent("coder") is spec

    @patch("core.mcp_agent_registry.log_json")
    def test_register_duplicate_raises(self, _):
        spec = _make_spec("coder", ["code_generation"])
        self.registry.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(spec)

    @patch("core.mcp_agent_registry.log_json")
    def test_register_duplicate_with_overwrite(self, _):
        spec = _make_spec("coder", ["code_generation"])
        self.registry.register(spec)
        new_spec = _make_spec("coder", ["code_generation", "refactor"])
        self.registry.register(new_spec, overwrite=True)
        assert self.registry.get_agent("coder") is new_spec

    @patch("core.mcp_agent_registry.log_json")
    def test_capabilities_indexed(self, _):
        spec = _make_spec("linter", ["lint_check"])
        self.registry.register(spec)
        results = self.registry.resolve_by_capability("lint_check")
        assert any(s.name == "linter" for s in results)


class TestTypedAgentRegistryResolve:
    def setup_method(self):
        self.registry = TypedAgentRegistry()

    @patch("core.mcp_agent_registry.log_json")
    def test_resolve_returns_empty_for_unknown_capability(self, _):
        results = self.registry.resolve_by_capability("nonexistent_cap")
        assert results == []

    @patch("core.mcp_agent_registry.log_json")
    def test_primary_capability_ranks_first(self, _):
        primary = _make_spec("primary_agent", ["code_gen", "lint"])
        secondary = _make_spec("secondary_agent", ["lint", "code_gen"])
        self.registry.register(primary)
        self.registry.register(secondary)
        results = self.registry.resolve_by_capability("code_gen")
        assert results[0].name == "primary_agent"

    @patch("core.mcp_agent_registry.log_json")
    def test_local_ranks_above_mcp(self, _):
        local_spec = _make_spec("local_agent", ["search"], source="local")
        mcp_spec = _make_spec("mcp_agent", ["search"], source="mcp")
        self.registry.register(local_spec)
        self.registry.register(mcp_spec)
        results = self.registry.resolve_by_capability("search")
        assert results[0].source == "local"

    @patch("core.mcp_agent_registry.log_json")
    def test_unhealthy_excluded_by_default(self, _):
        spec = _make_spec("flaky_agent", ["run"])
        self.registry.register(spec)
        self.registry.mark_unhealthy("flaky_agent")
        results = self.registry.resolve_by_capability("run")
        assert all(s.name != "flaky_agent" for s in results)

    @patch("core.mcp_agent_registry.log_json")
    def test_unhealthy_included_when_flag_false(self, _):
        spec = _make_spec("flaky_agent", ["run"])
        self.registry.register(spec)
        self.registry.mark_unhealthy("flaky_agent")
        results = self.registry.resolve_by_capability("run", skip_unhealthy=False)
        assert any(s.name == "flaky_agent" for s in results)


class TestTypedAgentRegistryHealthMarking:
    def setup_method(self):
        self.registry = TypedAgentRegistry()

    @patch("core.mcp_agent_registry.log_json")
    def test_mark_unhealthy_then_healthy(self, _):
        spec = _make_spec("svc", ["cap"])
        self.registry.register(spec)
        self.registry.mark_unhealthy("svc")
        self.registry.mark_healthy("svc")
        results = self.registry.resolve_by_capability("cap")
        assert any(s.name == "svc" for s in results)

    @patch("core.mcp_agent_registry.log_json")
    def test_mark_healthy_nonexistent_does_not_raise(self, _):
        self.registry.mark_healthy("ghost_agent")  # should not raise


class TestTypedAgentRegistryListAndClear:
    def setup_method(self):
        self.registry = TypedAgentRegistry()

    @patch("core.mcp_agent_registry.log_json")
    def test_list_agents_empty(self, _):
        assert self.registry.list_agents() == []

    @patch("core.mcp_agent_registry.log_json")
    def test_list_agents_returns_all(self, _):
        self.registry.register(_make_spec("a1", ["x"]))
        self.registry.register(_make_spec("a2", ["y"]))
        names = {s.name for s in self.registry.list_agents()}
        assert names == {"a1", "a2"}

    @patch("core.mcp_agent_registry.log_json")
    def test_clear_resets_registry(self, _):
        self.registry.register(_make_spec("a1", ["x"]))
        self.registry.clear()
        assert self.registry.list_agents() == []
        assert self.registry.resolve_by_capability("x") == []


class TestRegisterMcpAgents:
    @patch("core.mcp_agent_registry.log_json")
    def test_register_mcp_agents_success(self, _):
        import asyncio
        registry = TypedAgentRegistry()
        server_config = MCPServerConfig(name="test-mcp", command="python", port=8099)

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[
            {"name": "git_blame", "description": "Blame a file"},
            {"name": "read_file", "description": "Read a file"},
        ])

        with patch("core.mcp_agent_registry.MCPAsyncClient", return_value=mock_client):
            asyncio.get_event_loop().run_until_complete(
                registry.register_mcp_agents(server_config)
            )

        agents = registry.list_agents()
        names = {a.name for a in agents}
        assert "git_blame" in names
        assert "read_file" in names

    @patch("core.mcp_agent_registry.log_json")
    def test_register_mcp_agents_handles_exception(self, mock_log):
        import asyncio
        registry = TypedAgentRegistry()
        server_config = MCPServerConfig(name="bad-mcp", command="python", port=9999)

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("core.mcp_agent_registry.MCPAsyncClient", return_value=mock_client):
            asyncio.get_event_loop().run_until_complete(
                registry.register_mcp_agents(server_config)
            )

        # Should have logged the error, not raised
        assert mock_log.called


class TestGlobalAgentRegistrySingleton:
    def test_singleton_is_typed_registry(self):
        assert isinstance(agent_registry, TypedAgentRegistry)
