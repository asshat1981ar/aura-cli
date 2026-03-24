import pytest
import asyncio
from core.mcp_client import MCPAsyncClient
from core.mcp_health import check_mcp_health
from core.exceptions import MCPTimeoutError, MCPServerUnavailableError
from tests.fixtures.mcp_fixtures import MockMCPServer

@pytest.mark.anyio
async def test_mcp_client_health():
    # Start mock server
    server = MockMCPServer(port=9002)
    server.start()
    
    client = MCPAsyncClient("http://127.0.0.1:9002")
    health = await client.get_health()
    assert health["status"] == "ok"

@pytest.mark.anyio
async def test_mcp_client_get_tools():
    server = MockMCPServer(port=9003)
    server.start()
    
    client = MCPAsyncClient("http://127.0.0.1:9003")
    tools = await client.get_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"

@pytest.mark.anyio
async def test_mcp_client_call_tool():
    server = MockMCPServer(port=9004)
    server.start()
    
    client = MCPAsyncClient("http://127.0.0.1:9004")
    result = await client.call_tool("echo", {"text": "hello"})
    assert result["status"] == "success"
    assert "Echo: hello" in result["result"]

@pytest.mark.anyio
async def test_mcp_client_unavailable():
    # No server on port 9999
    client = MCPAsyncClient("http://127.0.0.1:9999", timeout=1)
    with pytest.raises(MCPServerUnavailableError):
        await client.get_health()

@pytest.mark.anyio
async def test_mcp_health_utility(monkeypatch):
    server = MockMCPServer(port=9005)
    server.start()
    
    # Mock config to point to our test port
    from core.config_manager import config
    monkeypatch.setattr(config, "get_mcp_server_port", lambda x: 9005)
    
    result = await check_mcp_health("test_server")
    assert result["status"] == "healthy"
    assert result["health_data"]["status"] == "ok"
