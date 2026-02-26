"""R3: Canonical MCP Pydantic models shared across all MCP server modules.

Import from here instead of defining local ToolCallRequest classes:

    from tools.mcp_types import ToolCallRequest, ToolResult
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """Generic tool-invocation payload accepted by all AURA MCP servers."""

    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Standard response envelope returned by all AURA MCP servers."""

    tool_name: str
    result: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
