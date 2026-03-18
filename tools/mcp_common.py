"""Shared MCP server infrastructure (E3).

Provides:
- ``CallRequest`` — standard tool-call request model.
- ``ToolResult`` — standard response wrapper.
- ``require_auth`` — configurable bearer token auth dependency.
- ``RateLimiter`` — sliding-window per-token rate limiter.
- ``MCPServerBase`` — base class for building namespaced MCP servers.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel

from core.logging_utils import log_json


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CallRequest(BaseModel):
    """Generic tool-invocation payload."""
    tool_name: str
    args: Dict[str, Any] = {}


class ToolResult(BaseModel):
    """Standard response wrapper for all MCP tool calls."""
    tool_name: str
    status: str = "success"
    result: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window per-token rate limiter."""

    def __init__(self, limit_per_min: int = 0):
        self.limit = limit_per_min
        self._state: Dict[str, deque] = {}

    def check(self, token: str) -> None:
        if self.limit <= 0:
            return
        now = time.time()
        window = 60.0
        timestamps = self._state.setdefault(token, deque())
        while timestamps and now - timestamps[0] > window:
            timestamps.popleft()
        if len(timestamps) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        timestamps.append(now)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def make_auth_dependency(
    token_env_var: str = "MCP_API_TOKEN",
    rate_limiter: RateLimiter | None = None,
) -> Callable:
    """Create a FastAPI auth dependency for bearer-token validation.

    When the env var is unset/empty, auth is skipped (returns ``"anon"``).
    """
    def _require_auth(authorization: Optional[str] = Header(default=None)) -> str:
        token = os.getenv(token_env_var, "")
        if not token:
            return "anon"
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
            raise HTTPException(status_code=403, detail="Invalid token")
        if rate_limiter:
            rate_limiter.check(parts[1])
        return parts[1]

    return _require_auth


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Manages tool descriptors and dispatch handlers for an MCP server namespace."""

    def __init__(self, namespace: str = ""):
        self.namespace = namespace
        self._descriptors: List[Dict[str, Any]] = []
        self._handlers: Dict[str, Callable] = {}

    def register(self, name: str, description: str, input_schema: Dict[str, Any],
                 handler: Callable) -> None:
        self._descriptors.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema,
        })
        self._handlers[name] = handler

    def register_batch(self, descriptors: List[Dict[str, Any]],
                       handlers: Dict[str, Callable]) -> None:
        """Bulk-register tools from a descriptor list and handler dict."""
        self._descriptors.extend(descriptors)
        self._handlers.update(handlers)

    @property
    def tool_names(self) -> List[str]:
        return list(self._handlers.keys())

    @property
    def descriptors(self) -> List[Dict[str, Any]]:
        return list(self._descriptors)

    def dispatch(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name!r}")

        start = time.time()
        try:
            result = handler(args)
            elapsed = (time.time() - start) * 1000
            log_json("INFO", "mcp_tool_call", details={
                "namespace": self.namespace,
                "tool": tool_name,
                "elapsed_ms": round(elapsed, 1),
            })
            return ToolResult(
                tool_name=tool_name,
                result=result,
                elapsed_ms=round(elapsed, 1),
            )
        except HTTPException:
            raise
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            log_json("ERROR", "mcp_tool_error", details={
                "namespace": self.namespace,
                "tool": tool_name,
                "error": str(exc),
                "elapsed_ms": round(elapsed, 1),
            })
            return ToolResult(
                tool_name=tool_name,
                status="error",
                error=str(exc),
                elapsed_ms=round(elapsed, 1),
            )
