"""
Unit tests for core/mcp_agent_registry.py

Tests for the TypedAgentRegistry and MCP capability resolution.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from core.mcp_agent_registry import TypedAgentRegistry, _resolve_mcp_capabilities, _MCP_CAPABILITY_GROUPS, agent_registry
from core.types import AgentSpec, MCPServerConfig


class TestResolveMcpCapabilities:
    """Tests for _resolve_mcp_capabilities function."""

    def test_tool_with_no_group(self):
        """Test capability resolution for tool not in any group."""
        result = _resolve_mcp_capabilities("unique_tool")
        assert result == ["unique_tool"]

    def test_tool_in_single_group(self):
        """Test capability resolution for tool in one group."""
        result = _resolve_mcp_capabilities("git_blame")
        assert "git_blame" in result
        assert "git" in result
        assert "git_analysis" in result  # Also in git_analysis group

    def test_tool_in_multiple_groups(self):
        """Test capability resolution for tool in multiple groups."""
        result = _resolve_mcp_capabilities("fetch_url")
        assert "fetch_url" in result
        assert "web" in result
        assert "search" in result

    def test_primary_capability_first(self):
        """Test that tool's own name is first in capabilities list."""
        result = _resolve_mcp_capabilities("docker_build")
        assert result[0] == "docker_build"

    def test_all_mcp_group_tools_resolve(self):
        """Test that all tools in MCP groups can be resolved."""
        for group, tools in _MCP_CAPABILITY_GROUPS.items():
            for tool in tools:
                result = _resolve_mcp_capabilities(tool)
                assert tool in result
                assert group in result


class TestTypedAgentRegistry:
    """Tests for TypedAgentRegistry class."""

    @pytest.fixture
    def registry(self):
        """Fixture providing a fresh TypedAgentRegistry."""
        reg = TypedAgentRegistry()
        return reg

    @pytest.fixture
    def sample_agent_spec(self):
        """Fixture providing a sample AgentSpec."""
        return AgentSpec(name="test_agent", description="A test agent", capabilities=["test_capability", "secondary_cap"], source="local")

    @pytest.fixture
    def mcp_agent_spec(self):
        """Fixture providing an MCP AgentSpec."""
        return AgentSpec(name="mcp_tool", description="An MCP tool", capabilities=["docker", "containerization"], source="mcp", mcp_server="test_server")

    def test_initialization(self, registry):
        """Test registry initializes with empty collections."""
        assert registry._agents == {}
        assert registry._capabilities == {}
        assert registry._unhealthy == set()

    def test_register_single_agent(self, registry, sample_agent_spec):
        """Test registering a single agent."""
        registry.register(sample_agent_spec)

        assert "test_agent" in registry._agents
        assert registry._agents["test_agent"] == sample_agent_spec

    def test_register_indexes_capabilities(self, registry, sample_agent_spec):
        """Test that registering indexes agent capabilities."""
        registry.register(sample_agent_spec)

        assert "test_capability" in registry._capabilities
        assert "test_agent" in registry._capabilities["test_capability"]
        assert "secondary_cap" in registry._capabilities
        assert "test_agent" in registry._capabilities["secondary_cap"]

    def test_register_duplicate_without_overwrite(self, registry, sample_agent_spec):
        """Test registering duplicate agent without overwrite raises error."""
        registry.register(sample_agent_spec)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(sample_agent_spec)

    def test_register_duplicate_with_overwrite(self, registry, sample_agent_spec):
        """Test registering duplicate agent with overwrite enabled."""
        registry.register(sample_agent_spec)

        # Modify the spec
        modified_spec = AgentSpec(name="test_agent", description="Modified description", capabilities=["new_capability"], source="local")

        # Should not raise with overwrite=True
        registry.register(modified_spec, overwrite=True)

        assert registry._agents["test_agent"].description == "Modified description"

    def test_get_agent_exists(self, registry, sample_agent_spec):
        """Test retrieving an existing agent."""
        registry.register(sample_agent_spec)

        result = registry.get_agent("test_agent")

        assert result == sample_agent_spec

    def test_get_agent_not_exists(self, registry):
        """Test retrieving a non-existent agent returns None."""
        result = registry.get_agent("nonexistent")

        assert result is None

    def test_resolve_by_capability_single_match(self, registry, sample_agent_spec):
        """Test resolving agents by capability with single match."""
        registry.register(sample_agent_spec)

        results = registry.resolve_by_capability("test_capability")

        assert len(results) == 1
        assert results[0].name == "test_agent"

    def test_resolve_by_capability_multiple_matches(self, registry):
        """Test resolving agents by capability with multiple matches."""
        agent1 = AgentSpec(name="agent1", description="First agent", capabilities=["shared_cap"], source="local")
        agent2 = AgentSpec(name="agent2", description="Second agent", capabilities=["shared_cap"], source="local")

        registry.register(agent1)
        registry.register(agent2)

        results = registry.resolve_by_capability("shared_cap")

        assert len(results) == 2
        assert any(r.name == "agent1" for r in results)
        assert any(r.name == "agent2" for r in results)

    def test_resolve_by_capability_primary_precedence(self, registry):
        """Test that primary capability matches rank higher."""
        # Agent with capability as primary (first in list)
        primary_agent = AgentSpec(name="primary", description="Primary match", capabilities=["target_cap", "other"], source="local")
        # Agent with capability as secondary
        secondary_agent = AgentSpec(name="secondary", description="Secondary match", capabilities=["other", "target_cap"], source="local")

        registry.register(secondary_agent)
        registry.register(primary_agent)

        results = registry.resolve_by_capability("target_cap")

        # Primary should be first
        assert results[0].name == "primary"
        assert results[1].name == "secondary"

    def test_resolve_by_capability_local_precedence(self, registry):
        """Test that local agents rank above MCP agents."""
        mcp_agent = AgentSpec(name="mcp", description="MCP agent", capabilities=["shared_cap"], source="mcp", mcp_server="server1")
        local_agent = AgentSpec(name="local", description="Local agent", capabilities=["shared_cap"], source="local")

        registry.register(mcp_agent)
        registry.register(local_agent)

        results = registry.resolve_by_capability("shared_cap")

        # Local should be first
        assert results[0].name == "local"
        assert results[1].name == "mcp"

    def test_resolve_by_capability_skip_unhealthy(self, registry, sample_agent_spec):
        """Test that unhealthy agents are skipped by default."""
        registry.register(sample_agent_spec)
        registry.mark_unhealthy("test_agent")

        results = registry.resolve_by_capability("test_capability")

        assert len(results) == 0

    def test_resolve_by_capability_include_unhealthy(self, registry, sample_agent_spec):
        """Test including unhealthy agents when skip_unhealthy=False."""
        registry.register(sample_agent_spec)
        registry.mark_unhealthy("test_agent")

        results = registry.resolve_by_capability("test_capability", skip_unhealthy=False)

        assert len(results) == 1
        assert results[0].name == "test_agent"

    def test_mark_unhealthy(self, registry, sample_agent_spec):
        """Test marking an agent as unhealthy."""
        registry.register(sample_agent_spec)
        registry.mark_unhealthy("test_agent")

        assert "test_agent" in registry._unhealthy

    def test_mark_healthy(self, registry, sample_agent_spec):
        """Test marking an agent as healthy."""
        registry.register(sample_agent_spec)
        registry.mark_unhealthy("test_agent")
        registry.mark_healthy("test_agent")

        assert "test_agent" not in registry._unhealthy

    def test_mark_healthy_idempotent(self, registry):
        """Test marking healthy is idempotent."""
        # Should not raise for non-existent agent
        registry.mark_healthy("never_unhealthy")

        assert "never_unhealthy" not in registry._unhealthy

    def test_list_agents_empty(self, registry):
        """Test listing agents when registry is empty."""
        results = registry.list_agents()

        assert results == []

    def test_list_agents_multiple(self, registry):
        """Test listing multiple registered agents."""
        agent1 = AgentSpec(name="agent1", description="First", capabilities=["c1"], source="local")
        agent2 = AgentSpec(name="agent2", description="Second", capabilities=["c2"], source="local")

        registry.register(agent1)
        registry.register(agent2)

        results = registry.list_agents()

        assert len(results) == 2
        assert any(a.name == "agent1" for a in results)
        assert any(a.name == "agent2" for a in results)

    def test_clear(self, registry, sample_agent_spec):
        """Test clearing the registry."""
        registry.register(sample_agent_spec)
        registry.mark_unhealthy("test_agent")

        registry.clear()

        assert registry._agents == {}
        assert registry._capabilities == {}
        assert registry._unhealthy == set()


class TestTypedAgentRegistryAsync:
    """Async tests for TypedAgentRegistry."""

    @pytest.fixture
    def registry(self):
        """Fixture providing a fresh TypedAgentRegistry."""
        return TypedAgentRegistry()

    @pytest.fixture
    def mock_server_config(self):
        """Fixture providing a mock MCPServerConfig."""
        return MCPServerConfig(name="test_server", port=8080, command="test", args=[])

    @pytest.mark.asyncio
    async def test_register_mcp_agents_success(self, registry, mock_server_config):
        """Test successful MCP agent registration."""
        mock_tools = [{"name": "docker_build", "description": "Build Docker image"}, {"name": "git_status", "description": "Check git status"}]

        with patch("core.mcp_client.MCPAsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools = AsyncMock(return_value=mock_tools)
            mock_client_class.return_value = mock_client

            await registry.register_mcp_agents(mock_server_config)

            # Verify agents were registered
            assert "docker_build" in registry._agents
            assert "git_status" in registry._agents

            # Verify capabilities were resolved
            docker_agent = registry.get_agent("docker_build")
            assert "docker_build" in docker_agent.capabilities
            assert "docker" in docker_agent.capabilities

            git_agent = registry.get_agent("git_status")
            assert "git_status" in git_agent.capabilities
            assert "git" in git_agent.capabilities

    @pytest.mark.asyncio
    async def test_register_mcp_agents_empty_tools(self, registry, mock_server_config):
        """Test MCP agent registration with empty tools list."""
        with patch("core.mcp_client.MCPAsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_class.return_value = mock_client

            await registry.register_mcp_agents(mock_server_config)

            assert len(registry._agents) == 0

    @pytest.mark.asyncio
    async def test_register_mcp_agents_connection_error(self, registry, mock_server_config):
        """Test MCP agent registration handles connection errors."""
        with patch("core.mcp_client.MCPAsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools = AsyncMock(side_effect=ConnectionError("Connection refused"))
            mock_client_class.return_value = mock_client

            # Should not raise exception
            await registry.register_mcp_agents(mock_server_config)

            assert len(registry._agents) == 0

    @pytest.mark.asyncio
    async def test_register_mcp_agents_overwrites_existing(self, registry, mock_server_config):
        """Test MCP agent registration overwrites existing agents."""
        # Register existing agent with same name
        existing = AgentSpec(name="docker_build", description="Old description", capabilities=["old"], source="local")
        registry.register(existing)

        mock_tools = [{"name": "docker_build", "description": "New description"}]

        with patch("core.mcp_client.MCPAsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools = AsyncMock(return_value=mock_tools)
            mock_client_class.return_value = mock_client

            await registry.register_mcp_agents(mock_server_config)

            # Should be overwritten
            agent = registry.get_agent("docker_build")
            assert agent.description == "New description"
            assert agent.source == "mcp"


class TestGlobalRegistry:
    """Tests for the global agent_registry singleton."""

    def test_global_registry_exists(self):
        """Test that global registry singleton exists."""
        assert agent_registry is not None
        assert isinstance(agent_registry, TypedAgentRegistry)

    def test_global_registry_is_singleton(self):
        """Test that global registry is a singleton."""
        from core.mcp_agent_registry import agent_registry as registry2

        assert agent_registry is registry2


class TestIntegration:
    """Integration tests for MCP agent registry."""

    def test_end_to_end_workflow(self):
        """Test complete workflow of registering and resolving agents."""
        registry = TypedAgentRegistry()

        # Register multiple agents with different capabilities
        agents = [
            AgentSpec(name="git_agent", description="Git operations", capabilities=["git", "version_control"], source="local"),
            AgentSpec(name="docker_agent", description="Docker operations", capabilities=["docker", "containerization"], source="mcp", mcp_server="docker_mcp"),
            AgentSpec(name="file_agent", description="File operations", capabilities=["file_system", "file_ops"], source="local"),
            AgentSpec(name="super_agent", description="Multiple capabilities", capabilities=["git", "docker", "file_system"], source="local"),
        ]

        for agent in agents:
            registry.register(agent)

        # Test resolution
        git_agents = registry.resolve_by_capability("git")
        assert len(git_agents) == 2  # git_agent and super_agent

        docker_agents = registry.resolve_by_capability("docker")
        assert len(docker_agents) == 2  # docker_agent and super_agent

        file_agents = registry.resolve_by_capability("file_system")
        assert len(file_agents) == 2  # file_agent and super_agent

        # Test primary capability precedence
        git_results = registry.resolve_by_capability("git")
        # git_agent has git as primary, super_agent has it secondary
        assert git_results[0].name == "git_agent"

        # Test marking unhealthy
        registry.mark_unhealthy("super_agent")
        git_agents_filtered = registry.resolve_by_capability("git")
        assert len(git_agents_filtered) == 1
        assert git_agents_filtered[0].name == "git_agent"

    def test_capability_index_consistency(self):
        """Test that capability index stays consistent through operations."""
        registry = TypedAgentRegistry()

        agent = AgentSpec(name="test_agent", description="Test", capabilities=["cap1", "cap2", "cap3"], source="local")

        registry.register(agent)

        # Verify all capabilities indexed
        for cap in ["cap1", "cap2", "cap3"]:
            assert cap in registry._capabilities
            assert "test_agent" in registry._capabilities[cap]

        # Clear and verify
        registry.clear()
        assert registry._capabilities == {}
