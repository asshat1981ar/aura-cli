# Design Spec: Rich Multi-Capability Agent Registration

**Date:** 2026-03-25
**Status:** Approved
**Scope:** agents/base.py, agents/registry.py, core/mcp_agent_registry.py, tests/test_roadmap_m3.py

## Problem Statement

Current registration assigns each agent a single capability matching its own name:
    AgentSpec(name="act", capabilities=["act"], source="local")

resolve_by_capability("python") returns [] even though CoderAgent handles Python.

## Goals

1. Agents declare semantic capabilities natively (class attribute).
2. FALLBACK_CAPABILITIES dict covers legacy agents.
3. resolve_by_capability() sorts by primary-capability match, then local-before-mcp.
4. MCP tools grouped into logical clusters (git_log -> ["git_log","git"]).
5. No breaking changes to existing tests or CLI behavior.

## Files Changed

| File | Change |
|------|--------|
| agents/base.py | Add capabilities: list[str] = [] |
| agents/registry.py | Add FALLBACK_CAPABILITIES, _make_spec(), update default_agents() |
| core/mcp_agent_registry.py | Add _MCP_CAPABILITY_GROUPS, _resolve_mcp_capabilities(), update sort |
| tests/test_roadmap_m3.py | Add 7 new tests |

## Tests

- test_rich_capability_resolution
- test_primary_capability_sorting
- test_local_before_mcp_sorting
- test_fallback_dict_used_for_legacy_agent
- test_native_capabilities_override_fallback
- test_mcp_tool_grouping
- test_mcp_tool_no_group
