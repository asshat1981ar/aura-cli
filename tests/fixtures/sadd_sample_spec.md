# AURA Server Test Suite

Expand the AURA FastAPI server with comprehensive tests and wire the A2A framework.

## Workstream: Server Unit Tests

Implement unit tests for all FastAPI endpoints in aura_cli/server.py.

- Create test fixtures for mock runtime
- Test /health, /metrics, /tools, /discovery endpoints
- Test /execute SSE streaming
- Acceptance: all endpoints covered with passing tests

## Workstream: A2A Framework Integration

Wire the core/a2a/ task framework into the orchestrator pipeline.

- Connect A2ATask state machine to LoopOrchestrator
- Add agent card registration
- Implement task lifecycle hooks
- Depends on: Server Unit Tests

## Workstream: Integration Test Suite

End-to-end integration tests for server + A2A + orchestrator.

- Test full request flow from HTTP to orchestrator cycle
- Verify SSE event streaming
- Test error handling and recovery paths
- Depends on: Server Unit Tests
- Depends on: A2A Framework Integration
