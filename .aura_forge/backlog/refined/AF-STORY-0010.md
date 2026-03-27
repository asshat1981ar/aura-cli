# Design Spec: Semantic MCP Tool Discovery for SADD

## Overview
This spec defines the implementation of a semantic search system for MCP tools, allowing AURA sub-agents to discover relevant tools dynamically based on their workstream goals, rather than relying on a static keyword map.

## Summary
The `MCPToolBridge` currently uses a hardcoded `_GOAL_TOOL_MAP`. We will replace/augment this with an MCP-driven discovery mechanism that uses semantic embeddings to match goal descriptions to tool definitions.

## Workstream: Create Discovery MCP Server
- Implement `tools/mcp_discovery_server.py`.
- This server will provide tools to:
  - `list_all_mcp_tools`: Aggregate tool lists from all registered MCP servers.
  - `search_tools_semantically`: Perform a similarity search on tool descriptions using `SentenceTransformer` (or a lightweight equivalent if already in repo).
- Acceptance:
  - [ ] Server starts on port 8030.
  - [ ] `list_all_mcp_tools` returns a unified JSON list.
  - [ ] `search_tools_semantically` returns top-K matching tools for a given query.

## Workstream: Update MCP Tool Bridge
- Refactor `core/sadd/mcp_tool_bridge.py`.
- Modify `discover_available_tools` to attempt calling the discovery MCP server.
- Modify `match_tools_for_goal` to use the semantic search tool if the discovery server is active.
- Acceptance:
  - [ ] `MCPToolBridge` remains backward compatible with the static map if the discovery server is down.
  - [ ] Sub-agents receive richer tool contexts when the discovery server is running.

## Workstream: Implement Discovery Skill
- Create `agents/skills/mcp_semantic_discovery.py`.
- This skill will wrap the discovery server calls for easier use by the `LoopOrchestrator`.
- Acceptance:
  - [ ] Skill correctly identifies the need for specific MCP tools based on plan context.

## Workstream: Integration Test
- Create `tests/integration/test_sadd_mcp_discovery.py`.
- Mock multiple MCP servers with different tool descriptions.
- Verify that a workstream goal "Audit security vulnerabilities" matches a tool named "vulnerability_scanner" even if "security" isn't in the tool name.
- Acceptance:
  - [ ] Integration test passes with mocked MCP registry.
