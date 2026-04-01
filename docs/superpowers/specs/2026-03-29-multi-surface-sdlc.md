# SADD Design Spec: Multi-Surface Architecture + SDLC Scaffold

## Summary

Two parallel workstreams addressing the final open issues:
1. JSON-RPC stdio transport layer (#209) — enables TUI, IDE, and browser interfaces
2. SDLC Orchestrator scaffold review + wiring (#308) — integrates Botpress ADK scaffold

## Workstream 1: JSON-RPC Stdio Transport

**Goal:** Implement `core/transport.py` with a JSON-RPC 2.0 over stdio transport that bridges to the existing orchestrator.

**Acceptance Criteria:**
- StdioTransport class reads JSON-RPC from stdin, writes to stdout
- Supports methods: `goal/run`, `goal/status`, `goal/add`, `system/health`
- Routes requests to LoopOrchestrator methods
- Handles errors per JSON-RPC 2.0 spec (error codes, messages)
- Tests cover all 4 methods + error handling
- No HTTP dependency — pure stdio

## Workstream 2: SDLC Scaffold Integration

**Goal:** Review the generated `sdlc-orchestrator/` scaffold, add gitignore entries, and create a PR.
Depends on: nothing (independent)

**Acceptance Criteria:**
- Review sdlc-orchestrator/ directory structure
- Add .gitignore entries for ADK build artifacts (node_modules, dist, .botpress)
- Verify package.json is valid
- Create dedicated branch and PR
- Also evaluate swarm-mcp-server/ directory

## Workstream 3: n8n Bridge Enhancement

**Goal:** Improve the n8n MCP proxy to support bidirectional communication — allow n8n workflows to call back into AURA's orchestrator via the stdio transport.
Depends on: Workstream 1

**Acceptance Criteria:**
- Add a callback registration mechanism to the proxy
- n8n workflow can trigger AURA goals via the execute_workflow → stdio bridge
- Tests verify round-trip: n8n → proxy → orchestrator → result → n8n

## Workstream 4: Integration Loop

**Goal:** Wire everything together — SADD coordinator runs workstreams, n8n reviews each, results feed back into the next cycle.
Depends on: Workstream 1, Workstream 2, Workstream 3

**Acceptance Criteria:**
- n8n Aura Dev Suite reviews each completed workstream
- Issues created for any findings
- Final PR merges all workstreams
