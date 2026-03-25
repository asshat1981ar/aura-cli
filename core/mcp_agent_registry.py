from __future__ import annotations
from typing import Dict, List, Optional
from core.types import AgentSpec, MCPServerConfig
from core.logging_utils import log_json


# ---------------------------------------------------------------------------
# MCP tool → capability group mapping
# ---------------------------------------------------------------------------
# Tools listed under a group key are registered under both their own name AND
# the group name, enabling broad queries like resolve_by_capability("git").
# The tool's own name is always the PRIMARY capability; the group is secondary.

_MCP_CAPABILITY_GROUPS: Dict[str, List[str]] = {
    "git":         ["git_blame", "git_log", "git_diff", "git_status", "git_commit", "git_push"],
    "file_system": ["read_file", "write_file", "list_dir", "search_files", "move_file", "delete_file"],
    "docker":      ["docker_build", "docker_run", "docker_logs", "docker_ps", "docker_stop"],
    "database":    ["execute_query", "list_tables", "describe_schema", "run_migration"],
    "web":         ["fetch_url", "http_get", "http_post", "web_search"],
    "github":      ["gh_issue", "gh_pr", "gh_repo", "gh_release", "gh_workflow"],
    # AURA skills exposed as MCP tools (via aura_mcp_skills_server)
    "code_analysis": ["ast_analyzer", "symbol_indexer", "complexity_scorer", "code_clone_detector"],
    "security":      ["security_scanner", "security_hardener", "semgrep_scan"],
    "lint_check":    ["linter_enforcer", "type_checker", "error_pattern_matcher"],
    "architecture":  ["architecture_validator", "dependency_analyzer", "structural_analyzer"],
    "test_coverage": ["test_coverage_analyzer", "incremental_differ"],
    "git_analysis":  ["git_history_analyzer", "git_blame", "git_log", "git_diff"],
    "documentation": ["doc_generator", "changelog_generator"],
    # MCP external servers by capability cluster
    "browser":       ["playwright_navigate", "playwright_click", "puppeteer_screenshot"],
    "search":        ["brave_search", "fetch_url", "web_search"],
    "memory_store":  ["memory_add", "memory_search", "memory_get"],
    "sadd":          ["sadd_parse_spec", "sadd_session_status", "sadd_list_sessions", "sadd_session_events", "sadd_session_artifacts"],
}


def _resolve_mcp_capabilities(tool_name: str) -> List[str]:
    """Return capability list for an MCP tool: its own name (primary) plus any
    logical group names it belongs to (secondary).
    """
    caps: List[str] = [tool_name]
    for group, members in _MCP_CAPABILITY_GROUPS.items():
        if tool_name in members and group not in caps:
            caps.append(group)
    return caps


class TypedAgentRegistry:
    """
    A modern, typed registry for managing both local and MCP-backed agents.
    Supports resolution by capability with deterministic precedence:
      1. Primary-capability match ranks above secondary-capability match.
      2. Local agents rank above MCP agents when both match.
    """

    def __init__(self):
        self._agents: Dict[str, AgentSpec] = {}
        self._capabilities: Dict[str, List[str]] = {}  # capability -> list of agent names
        self._unhealthy: set = set()  # agent names currently marked unavailable

    def register(self, spec: AgentSpec, overwrite: bool = False):
        """Register an agent specification."""
        if spec.name in self._agents and not overwrite:
            log_json("ERROR", "agent_registration_conflict", details={"name": spec.name})
            raise ValueError(f"Agent '{spec.name}' is already registered.")

        self._agents[spec.name] = spec

        # Index capabilities
        for cap in spec.capabilities:
            if cap not in self._capabilities:
                self._capabilities[cap] = []
            if spec.name not in self._capabilities[cap]:
                self._capabilities[cap].append(spec.name)

        log_json("INFO", "agent_registered", details={"name": spec.name, "source": spec.source, "capabilities": spec.capabilities})

    def get_agent(self, name: str) -> Optional[AgentSpec]:
        """Retrieve an agent by name."""
        return self._agents.get(name)

    def resolve_by_capability(self, capability: str, skip_unhealthy: bool = True) -> List[AgentSpec]:
        """Find agents that support a specific capability.

        Sorted by deterministic precedence:
          1. Agents whose *primary* capability (first entry) matches rank first.
          2. Local agents rank above MCP agents within the same tier.

        Args:
            capability:      The capability string to look up.
            skip_unhealthy:  When ``True`` (default), agents marked unhealthy via
                             :meth:`mark_unhealthy` are excluded from results.
        """
        names = self._capabilities.get(capability, [])
        results = [
            self._agents[name]
            for name in names
            if name in self._agents and not (skip_unhealthy and name in self._unhealthy)
        ]

        def _sort_key(spec: AgentSpec):
            primary_match = 0 if (spec.capabilities and spec.capabilities[0] == capability) else 1
            source_rank   = 0 if spec.source == "local" else 1
            return (primary_match, source_rank)

        results.sort(key=_sort_key)
        return results

    def mark_unhealthy(self, name: str) -> None:
        """Mark *name* as unavailable; it will be excluded from :meth:`resolve_by_capability`."""
        self._unhealthy.add(name)
        log_json("WARN", "agent_marked_unhealthy", details={"name": name})

    def mark_healthy(self, name: str) -> None:
        """Restore *name* to the available pool after a previous :meth:`mark_unhealthy`."""
        self._unhealthy.discard(name)
        log_json("INFO", "agent_marked_healthy", details={"name": name})

    def list_agents(self) -> List[AgentSpec]:
        """Return all registered agent specifications."""
        return list(self._agents.values())

    async def register_mcp_agents(self, server_config: MCPServerConfig):
        """Discover and register agents from an MCP server.

        Each tool is registered under its own name (primary capability) plus any
        logical group names defined in :data:`_MCP_CAPABILITY_GROUPS`.
        """
        from core.mcp_client import MCPAsyncClient

        url = f"http://127.0.0.1:{server_config.port}"
        client = MCPAsyncClient(url)
        try:
            tools = await client.get_tools()
            for tool in tools:
                spec = AgentSpec(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    capabilities=_resolve_mcp_capabilities(tool["name"]),
                    source="mcp",
                    mcp_server=server_config.name,
                )
                self.register(spec, overwrite=True)
        except Exception as e:
            log_json("ERROR", "mcp_agent_discovery_failed", details={"server": server_config.name, "error": str(e)})

    def clear(self):
        """Clear the registry (mostly for testing)."""
        self._agents.clear()
        self._capabilities.clear()
        self._unhealthy.clear()


# Global singleton for the typed registry
agent_registry = TypedAgentRegistry()
