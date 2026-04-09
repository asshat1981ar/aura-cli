import httpx
import asyncio
import time
from typing import Any, ClassVar, Dict, List
from core.exceptions import (
    MCPTimeoutError,
    MCPServerUnavailableError,
    MCPInvalidResponseError,
    MCPRetryExhaustedError,
)
from core.logging_utils import log_json
from core.security.http_client import attach_dpop_headers


class MCPAsyncClient:
    """Async client for interacting with MCP-compatible HTTP services."""

    _client_pool: ClassVar[Dict[str, httpx.AsyncClient]] = {}
    # Tracks the id() of the event loop that created each pooled client.
    # If the running loop changes (e.g. between pytest-anyio test cases) the
    # stale client is discarded and a fresh one is created for the new loop.
    _client_pool_loops: ClassVar[Dict[str, int]] = {}
    _pool_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    async def get_client(cls, timeout: int) -> httpx.AsyncClient:
        """Get or create a pooled httpx.AsyncClient.

        A new client is created whenever the current event loop differs from
        the loop that created the cached client, preventing "Event loop is
        closed" errors across test boundaries or server restarts.
        """
        key = f"timeout_{timeout}"
        current_loop_id = id(asyncio.get_running_loop())
        async with cls._pool_lock:
            client = cls._client_pool.get(key)
            stored_loop_id = cls._client_pool_loops.get(key)
            if client is None or client.is_closed or stored_loop_id != current_loop_id:
                client = httpx.AsyncClient(timeout=timeout)
                cls._client_pool[key] = client
                cls._client_pool_loops[key] = current_loop_id
        return client

    @classmethod
    async def close_all(cls):
        """Close all clients in the pool."""
        for client in cls._client_pool.values():
            await client.aclose()
        cls._client_pool.clear()
        cls._client_pool_loops.clear()

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        # Attach DPoP proof to outgoing requests
        headers = dict(kwargs.pop("headers", {}) or {})
        bearer = headers.get("Authorization", "")
        token = bearer.removeprefix("Bearer ").strip() if bearer.startswith("Bearer ") else None
        attach_dpop_headers(headers, method, url, access_token=token)
        kwargs["headers"] = headers

        max_retries = 3
        backoff = 1.0
        start_time = time.perf_counter()

        client = await self.get_client(self.timeout)
        for attempt in range(max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                duration = time.perf_counter() - start_time
                try:
                    res_json = response.json()
                    log_json("DEBUG", "mcp_client_request_success", details={"url": url, "method": method, "duration_ms": round(duration * 1000, 2), "attempts": attempt + 1})
                    return res_json
                except (OSError, IOError, ValueError):
                    raise MCPInvalidResponseError(f"MCP server at {url} returned invalid JSON")
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                is_timeout = isinstance(e, httpx.TimeoutException)
                err_type = "timeout" if is_timeout else "connection_error"
                log_json("WARN", f"mcp_client_{err_type}", details={"url": url, "attempt": attempt + 1})

                if attempt == max_retries - 1:
                    if is_timeout:
                        raise MCPTimeoutError(f"Request to {url} timed out after {max_retries} attempts")
                    else:
                        raise MCPServerUnavailableError(f"MCP server at {url} is unavailable")

                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except httpx.HTTPStatusError as e:
                log_json("ERROR", "mcp_client_http_error", details={"url": url, "status": e.response.status_code})
                raise MCPInvalidResponseError(f"MCP server at {url} returned status {e.response.status_code}")
            except Exception as e:
                log_json("ERROR", "mcp_client_unexpected_error", details={"url": url, "error": str(e)})
                raise MCPInvalidResponseError(f"Unexpected error calling MCP server at {url}: {e}")

        raise MCPRetryExhaustedError(f"All {max_retries} retries exhausted for {url}")

    async def get_health(self) -> Dict[str, Any]:
        """Check the health status of the MCP server."""
        return await self._request("GET", "/health")

    async def get_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server."""
        result = await self._request("GET", "/tools")
        if isinstance(result, list):
            return result
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a specific tool on the MCP server."""
        try:
            return await self._request("POST", "/call", json={"name": name, "arguments": arguments})
        except MCPInvalidResponseError:
            return await self._request("POST", f"/tool/{name}", json=arguments)

    async def get_discovery(self) -> Dict[str, Any]:
        """Retrieve service discovery metadata from the MCP server."""
        return await self._request("GET", "/discovery")
