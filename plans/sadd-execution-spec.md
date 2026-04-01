# SADD Execution Spec: Parallel Goal Processing

Execute high-priority codebase improvement goals using Sub-Agent Driven Development with parallel workstreams and Agent Swarm orchestration.

## Workstream: Exception Handling Hardening

Replace bare `except Exception` patterns with specific exception handling across core modules.

- Audit all 364+ bare exception handlers in core/ and agents/
- Create specific exception types in core/exceptions.py
- Replace generic handlers with specific ones
- Add unit tests for each exception path
- Acceptance: <20 bare `except Exception` remain, all tests pass

## Workstream: Remove Hardcoded Secrets

Remove hardcoded passwords and implement secret scanning.

- Audit environments/bootstrap.py for secrets
- Replace with environment variable templates
- Add detect-secrets to pre-commit config
- Document in SECURITY.md
- Acceptance: Zero hardcoded secrets, scanning active

## Workstream: Agent Swarm Orchestration

Implement and configure Agent Swarm for multi-agent coordination in SADD workflows.

- Enable `AURA_ENABLE_SWARM=1` environment variable handling
- Create `core/swarm_supervisor.py` for agent coordination
- Implement agent registry with capability-based routing
- Add swarm health monitoring and auto-scaling
- Configure agent-to-agent communication protocol
- Create agent lifecycle management (spawn, monitor, terminate)
- Add swarm telemetry collection
- Acceptance: 5+ agents can run concurrently, auto-assigned to workstreams, telemetry visible

## Workstream: File Filtering Deduplication

Extract shared file filtering logic from 15+ skills into a centralized utility in core/path_utils.py.

- Create core/path_utils.py with should_skip_path() function
- Support patterns: .git, __pycache__, node_modules, .venv
- Update all skills to use shared utility
- Remove 100+ lines of duplicated code
- Acceptance: All skills use shared utility, tests pass

Depends on: Exception Handling Hardening

## Workstream: Implement Streaming for Large Files

Add streaming support for files larger than 1MB to prevent memory exhaustion in core/streaming_utils.py.

- Create core/streaming_utils.py with safe_read_text() helper
- Add file size checks before reading
- Implement streaming/chunking for large files
- Update 15+ skills to use streaming
- Acceptance: Memory stays below 100MB even with 100MB files

Depends on: Exception Handling Hardening

## Workstream: Test Coverage Enforcement

Increase test coverage from 40% to 70% for critical core modules.

- Focus on orchestrator.py, workflow_engine.py, file_tools.py
- Update fail_under in pyproject.toml
- Generate coverage gap report
- Add missing unit tests
- Acceptance: 70% coverage achieved, CI passes

Depends on: File Filtering Deduplication

## Workstream: Fix Skipped Tests

Audit and update or remove 5+ legacy skipped tests.

- Find all @pytest.mark.skip and @unittest.skip tests
- Update tests that can be fixed
- Remove obsolete legacy tests
- Document reasons for remaining skips
- Acceptance: Zero undocumented skipped tests

Depends on: Test Coverage Enforcement

## Workstream: Swarm Integration with SADD n8n

Integrate Agent Swarm with SADD workflows and n8n automation.

- Configure swarm agents to trigger n8n webhooks
- Implement agent-to-n8n communication protocol
- Create swarm-aware n8n nodes (WF-7-agent-dispatcher.json)
- Add agent telemetry to n8n workflows
- Implement agent swarm visualization dashboard
- Acceptance: Agents can trigger n8n workflows, telemetry flows to dashboard

Depends on: Agent Swarm Orchestration
