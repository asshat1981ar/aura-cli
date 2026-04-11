import pytest
import httpx

from core.mcp_client import MCPAsyncClient
from core.mcp_health import check_mcp_health
from core.exceptions import MCPServerUnavailableError


def _mock_client(handler, timeout=30):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, timeout=timeout)


def _patch_get_client(monkeypatch, client):
    async def _get_client(cls, timeout):
        return client

    monkeypatch.setattr(MCPAsyncClient, "get_client", classmethod(_get_client))


@pytest.mark.anyio
async def test_mcp_client_health(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "version": "1.0.0"})

    client = MCPAsyncClient("http://127.0.0.1:9002")
    mock_client = _mock_client(handler)
    _patch_get_client(monkeypatch, mock_client)

    health = await client.get_health()
    assert health["status"] == "ok"
    await mock_client.aclose()


@pytest.mark.anyio
async def test_mcp_client_get_tools(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tools"
        return httpx.Response(200, json={"tools": [{"name": "echo", "description": "echo back"}]})

    client = MCPAsyncClient("http://127.0.0.1:9003")
    mock_client = _mock_client(handler)
    _patch_get_client(monkeypatch, mock_client)

    tools = await client.get_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"
    await mock_client.aclose()


@pytest.mark.anyio
async def test_mcp_client_call_tool(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/call"
        body = request.read().decode("utf-8")
        assert "echo" in body
        return httpx.Response(200, json={"status": "success", "result": "Echo: hello"})

    client = MCPAsyncClient("http://127.0.0.1:9004")
    mock_client = _mock_client(handler)
    _patch_get_client(monkeypatch, mock_client)

    result = await client.call_tool("echo", {"text": "hello"})
    assert result["status"] == "success"
    assert "Echo: hello" in result["result"]
    await mock_client.aclose()


@pytest.mark.anyio
async def test_mcp_client_unavailable():
    client = MCPAsyncClient("http://127.0.0.1:9999", timeout=1)
    with pytest.raises(MCPServerUnavailableError):
        await client.get_health()


@pytest.mark.anyio
async def test_mcp_health_utility(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "version": "1.0.0"})

    mock_client = _mock_client(handler)
    _patch_get_client(monkeypatch, mock_client)

    from core.config_manager import config

    monkeypatch.setattr(config, "get_mcp_server_port", lambda x: 9005)
    result = await check_mcp_health("test_server")
    assert result["status"] == "healthy"
    assert result["health_data"]["status"] == "ok"
    await mock_client.aclose()
