"""Skill for semantic MCP tool discovery.

Enables agents to find relevant tools across the entire AURA MCP cluster
by describing what they need to accomplish.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from core.sadd.mcp_tool_bridge import MCPToolBridge

logger = logging.getLogger(__name__)


def mcp_semantic_discovery(goal: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search for MCP tools that semantically match the provided goal.

    Returns a dictionary with 'tools' (list of matched tools) and 'discovery_method'.
    """
    bridge = MCPToolBridge()
    matched = bridge.match_tools_for_goal(goal)

    # Filter for top_k if needed (bridge already does some filtering)
    results = matched[:top_k]

    method = "semantic" if any(r.get("matched_semantic") for r in results) else "keyword_fallback"

    return {"tools": results, "discovery_method": method, "goal": goal}


def list_all_available_capabilities() -> List[Dict[str, Any]]:
    """Aggregate all capabilities from all active MCP servers."""
    bridge = MCPToolBridge()
    return bridge.discover_available_tools()
