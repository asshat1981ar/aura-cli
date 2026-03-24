import pytest
from core.mcp_agent_registry import TypedAgentRegistry
from core.types import AgentSpec, MCPServerConfig
from tests.fixtures.mcp_fixtures import MockMCPServer

def test_registry_local_registration():
    registry = TypedAgentRegistry()
    spec = AgentSpec(name="local_agent", description="desc", capabilities=["coding"])
    registry.register(spec)
    
    assert registry.get_agent("local_agent") == spec
    assert len(registry.resolve_by_capability("coding")) == 1

def test_registry_precedence():
    registry = TypedAgentRegistry()
    local_spec = AgentSpec(name="local_coder", description="local", capabilities=["code"], source="local")
    mcp_spec = AgentSpec(name="mcp_coder", description="mcp", capabilities=["code"], source="mcp")
    
    registry.register(mcp_spec)
    registry.register(local_spec)
    
    resolved = registry.resolve_by_capability("code")
    assert len(resolved) == 2
    # Local should be first
    assert resolved[0].source == "local"
    assert resolved[1].source == "mcp"

def test_registry_conflict():
    registry = TypedAgentRegistry()
    spec = AgentSpec(name="agent1", description="desc")
    registry.register(spec)
    
    with pytest.raises(ValueError):
        registry.register(spec) # Duplicate name

@pytest.mark.anyio
async def test_registry_mcp_discovery():
    server = MockMCPServer(port=9006)
    server.start()
    
    registry = TypedAgentRegistry()
    config = MCPServerConfig(name="mock", command="none", port=9006)
    
    await registry.register_mcp_agents(config)
    
    agents = registry.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "echo"
    assert agents[0].source == "mcp"
    assert agents[0].mcp_server == "mock"
