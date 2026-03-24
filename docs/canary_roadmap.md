# Aura-CLI Canary Rollout Strategy

This document outlines the planned rollout of the new async orchestrator and typed registry (Milestone 5).

## Rollout Waves

### Wave 1: Low-Risk Discovery & Health (Active)
- **Capabilities:** `mcp_discovery`, `mcp_health`
- **MCP Servers:** `aura-dev-tools` (8001), `aura-skills` (8002)
- **Feature Flag:** `AURA_ENABLE_MCP_REGISTRY=true`
- **Status:** Foundations landed. Discovery agent integrated.

### Wave 2: Core Tooling Canary (Planned)
- **Capabilities:** `code_search`, `investigation`
- **MCP Servers:** `aura-control` (8003)
- **Feature Flag:** `AURA_NEW_ORCHESTRATOR_SHADOW_MODE=true`
- **Goal:** Compare legacy vs. new execution paths for non-mutating tasks.

### Wave 3: Full Orchestration Canary (Planned)
- **Capabilities:** `plan`, `act`, `verify`
- **MCP Servers:** `aura-agentic-loop` (8006)
- **Feature Flag:** `AURA_ENABLE_NEW_ORCHESTRATOR=true` (Selective)
- **Goal:** Shift high-risk mutating tasks to the new async pipeline.

## Observability
- Monitor `orchestrator_shadow_mode_active` logs.
- Track `mcp_client_timeout` and `mcp_client_connection_error` events.
- Success Metric: Zero regressions in `AURA_NEW_ORCHESTRATOR_SHADOW_MODE`.

## Rollback Plan
1. Set `AURA_ENABLE_MCP_REGISTRY=false`.
2. Set `AURA_ENABLE_NEW_ORCHESTRATOR=false`.
3. Set `AURA_FORCE_LEGACY_ORCHESTRATOR=true`.
