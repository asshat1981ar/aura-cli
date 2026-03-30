"""Bridge between SADD workstreams and MCP tool servers.

Discovers available MCP tools, matches them to workstream goals,
and builds tool context dicts for injection into sub-agent runners.
"""
from __future__ import annotations

import logging
import os
import requests
from typing import Any

logger = logging.getLogger(__name__)

# Discovery server default port
DISCOVERY_PORT = 8025
API_TOKEN = os.getenv("MCP_API_TOKEN", "default_token")

# Keyword → MCP tool/skill mapping for goal-based matching
_GOAL_TOOL_MAP: dict[str, list[str]] = {
    "security": ["security_scanner", "secrets_scan", "semgrep_scan"],
    "test": ["test_coverage_analyzer", "lint_files"],
    "lint": ["linter_enforcer", "lint_files", "lint_all"],
    "refactor": ["refactoring_advisor", "complexity_scorer", "architecture_validator"],
    "doc": ["doc_generator", "docstring_fill"],
    "depend": ["dependency_analyzer", "dependency_audit"],
    "type": ["type_checker"],
    "perform": ["performance_profiler"],
    "schema": ["schema_validator"],
    "format": ["format"],
    "git": ["git_blame_snippet", "git_file_history"],
    "search": ["structured_search", "code_intel_xref"],
}


class MCPToolBridge:
    """Discovers and matches MCP tools to SADD workstream goals."""

    def __init__(self, mcp_registry: Any = None) -> None:
        self._registry = mcp_registry

    def discover_available_tools(self) -> list[dict[str, Any]]:
        """Discover tools from the MCP registry or Discovery server."""
        # 1. Try Discovery Server first
        try:
            url = f"http://localhost:{DISCOVERY_PORT}/call"
            resp = requests.post(
                url,
                json={"tool_name": "list_all_mcp_tools", "args": {}},
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=1.0
            )
            if resp.status_code == 200:
                data = resp.json()
                tools = data.get("data", {}).get("tools", [])
                if tools:
                    return tools
        except Exception as e:
            logger.debug(f"Discovery server unreachable: {e}")

        # 2. Fallback to Registry if available
        if self._registry:
            try:
                from core.mcp_registry import list_registered_services
                services = list_registered_services()
                tools = []
                for svc in services:
                    for cap in svc.get("capabilities", []):
                        tools.append({"name": cap, "server": svc.get("name", "unknown")})
                return tools
            except Exception:
                logger.debug("MCP registry not available, using static tool list")

        # Static fallback: known tools from AURA's MCP servers
        return [
            {"name": name, "server": "static"}
            for names in _GOAL_TOOL_MAP.values()
            for name in names
        ]

    def match_tools_for_goal(self, goal_text: str) -> list[dict[str, Any]]:
        """Match MCP tools relevant to a workstream goal."""
        # 1. Try Semantic Discovery Server
        try:
            url = f"http://localhost:{DISCOVERY_PORT}/call"
            resp = requests.post(
                url,
                json={"tool_name": "search_tools_semantically", "args": {"query": goal_text, "top_k": 3}},
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=1.0
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", {}).get("results", [])
                if results:
                    return [{"name": r["name"], "matched_semantic": True, "server": r.get("server")} for r in results]
        except Exception as e:
            logger.debug(f"Semantic discovery failed: {e}")

        # 2. Fallback to Static Keyword Map
        goal_lower = goal_text.lower()
        matched: list[dict[str, Any]] = []
        seen: set[str] = set()

        for keyword, tool_names in _GOAL_TOOL_MAP.items():
            if keyword in goal_lower:
                for name in tool_names:
                    if name not in seen:
                        matched.append({"name": name, "matched_keyword": keyword})
                        seen.add(name)

        return matched

    def build_tool_context(self, tool_names: list[str]) -> dict[str, Any]:
        """Build a context dict for injection into sub-agent runners."""
        return {
            "available_tools": [
                {"name": name, "type": "mcp_tool"} for name in tool_names
            ],
            "tool_discovery_source": "sadd_mcp_bridge",
        }
