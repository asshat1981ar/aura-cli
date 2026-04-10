import pytest
import uuid
from core.orchestrator import LoopOrchestrator
from core.types import TaskRequest, ExecutionContext, AgentSpec
from core.mcp_agent_registry import agent_registry
from tests.fixtures.mcp_fixtures import MockMCPServer


class MockAgent:
    def run(self, input_data):
        return {"status": "ok", "echo": input_data.get("val")}


@pytest.mark.anyio
async def test_orchestrator_dispatch_local():
    # Register a local agent
    agent_registry.clear()
    spec = AgentSpec(name="local_test", description="test", capabilities=["test"], source="local")
    agent_registry.register(spec)

    # Create orchestrator with mock agent
    agents = {"local_test": MockAgent()}
    orch = LoopOrchestrator(agents=agents)

    req = TaskRequest(task_id=str(uuid.uuid4()), agent_name="local_test", input_data={"val": 123})

    result = await orch._dispatch_task(req)
    assert result.status == "success"
    assert result.output["echo"] == 123


@pytest.mark.anyio
async def test_orchestrator_dispatch_mcp(monkeypatch):
    # Start mock MCP server
    server = MockMCPServer(port=9007)
    server.start()

    # Register MCP agent
    agent_registry.clear()
    spec = AgentSpec(name="echo", description="echo", capabilities=["echo"], source="mcp", mcp_server="mock_mcp")
    agent_registry.register(spec)

    # Mock registry to find the service
    from core.mcp_registry import _SERVICE_SPECS, MCPServiceSpec

    mock_spec = MCPServiceSpec(config_name="mock_mcp", server_name="mock-server", title="mock", kind="tooling", auth_env=None, port_envs=(), endpoints=("/call",), capabilities=())

    # Update global config to point to our mock port
    from core.config_manager import config
    from unittest.mock import patch

    monkeypatch.setattr(config, "get_mcp_server_port", lambda x: 9007)

    orch = LoopOrchestrator(agents={})

    req = TaskRequest(task_id=str(uuid.uuid4()), agent_name="echo", input_data={"text": "hello mcp"})

    with patch("core.mcp_registry._SERVICE_SPECS", (mock_spec,)):
        result = await orch._dispatch_task(req)
    assert result.status == "success"
    assert "Echo: hello mcp" in result.output["result"]


@pytest.mark.anyio
async def test_orchestrator_dispatch_fallback_to_legacy():
    # Agent NOT in typed registry
    agent_registry.clear()

    agents = {"legacy_agent": MockAgent()}
    orch = LoopOrchestrator(agents=agents)

    req = TaskRequest(task_id=str(uuid.uuid4()), agent_name="legacy_agent", input_data={"val": "legacy"})

    result = await orch._dispatch_task(req)
    assert result.status == "success"
    assert result.output["echo"] == "legacy"
