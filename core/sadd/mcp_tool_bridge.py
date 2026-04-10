"""Bridge between SADD workstreams and MCP tool servers.

Discovers available MCP tools, matches them to workstream goals,
and builds tool context dicts for injection into sub-agent runners.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

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

    def _call_discovery(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Call the discovery server and return a decoded JSON object."""
        url = f"http://localhost:{DISCOVERY_PORT}/call"
        resp = requests.post(
            url,
            json={"tool_name": tool_name, "args": args},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=1.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None

    @staticmethod
    def _normalize_discovered_tools(tools: Any, *, source: str) -> list[dict[str, Any]]:
        """Normalize discovery payloads into a stable list of tool records."""
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        if not isinstance(tools, list):
            return normalized
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if not name:
                continue
            server = tool.get("server", "unknown")
            key = (str(server), str(name))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "name": str(name),
                    "server": str(server),
                    "description": tool.get("description", ""),
                    "score": tool.get("score"),
                    "match_source": source,
                }
            )
        return normalized

    def discover_available_tools(self) -> list[dict[str, Any]]:
        """Discover tools from the MCP registry or Discovery server."""
        # 1. Try Discovery Server first
        try:
            data = self._call_discovery("list_all_mcp_tools", {})
            if data:
                tools = self._normalize_discovered_tools(
                    data.get("data", {}).get("tools", []),
                    source="discovery_server",
                )
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
                        tools.append(
                            {
                                "name": cap,
                                "server": svc.get("name", "unknown"),
                                "description": "",
                                "score": None,
                                "match_source": "registry",
                            }
                        )
                return tools
            except (ImportError, AttributeError, ConnectionError) as e:
                logger.debug("MCP registry not available, using static tool list: %s", e)

        # Static fallback: known tools from AURA's MCP servers
        return [
            {
                "name": name,
                "server": "static",
                "description": "",
                "score": None,
                "match_source": "static_fallback",
            }
            for names in _GOAL_TOOL_MAP.values()
            for name in names
        ]

    def match_tools_for_goal(self, goal_text: str) -> list[dict[str, Any]]:
        """Match MCP tools relevant to a workstream goal."""
        # 1. Try Semantic Discovery Server
        try:
            data = self._call_discovery(
                "search_tools_semantically",
                {"query": goal_text, "top_k": 3},
            )
            if data:
                results = self._normalize_discovered_tools(
                    data.get("data", {}).get("results", []),
                    source="semantic_discovery",
                )
                if results:
                    return [
                        {
                            **result,
                            "matched_semantic": True,
                            "matched_keyword": None,
                        }
                        for result in results
                    ]
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
                        matched.append(
                            {
                                "name": name,
                                "server": "static",
                                "description": "",
                                "score": None,
                                "match_source": "keyword_fallback",
                                "matched_semantic": False,
                                "matched_keyword": keyword,
                            }
                        )
                        seen.add(name)

        return matched

    def build_tool_context(self, tools: list[str] | list[dict[str, Any]]) -> dict[str, Any]:
        """Build a context dict for injection into sub-agent runners."""
        available_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, str):
                available_tools.append({"name": tool, "type": "mcp_tool"})
            elif isinstance(tool, dict) and tool.get("name"):
                available_tools.append(
                    {
                        "name": tool["name"],
                        "type": "mcp_tool",
                        "server": tool.get("server"),
                        "description": tool.get("description", ""),
                        "match_source": tool.get("match_source"),
                        "score": tool.get("score"),
                        "matched_keyword": tool.get("matched_keyword"),
                        "matched_semantic": tool.get("matched_semantic", False),
                    }
                )
        return {
            "available_tools": available_tools,
            "tool_discovery_source": "sadd_mcp_bridge",
        }
