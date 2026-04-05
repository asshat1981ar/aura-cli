# Multi-Agent Prompt Library

Use these prompt sets with the shared collaboration files:

- `/home/westonaaron675/aura-cli/COLLAB_CONTEXT.md`
- `/home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md`

General rule:

- `Codex` is the default coordinator for this repository.
- `Gemini` is the default research and analysis worker.
- `Copilot coding agent` is the default GitHub-native and bounded test worker.
- `Kimi` is the default sidecar worker for isolated docs, helper, or bounded implementation tasks.

## Shared Coordinator Prompt

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

You are the coordinator for this repository.

Your job:
1. Read both files first.
2. Pick the current real task.
3. Break it into non-overlapping subtasks.
4. Assign exactly one owner per subtask.
5. Define for each subtask:
   - title
   - owner
   - status
   - depends on
   - exact write scope
   - read-only context
   - deliverable
   - verification
   - handoff target
6. Update COLLAB_CONTEXT.md before any worker starts.
7. Do not let two agents edit the same file at the same time.
8. Keep central runtime integration ownership with Codex unless explicitly reassigned.
9. When workers finish, integrate, verify, and decide the next assignment.

If the task is vague, clarify it in COLLAB_CONTEXT.md and do not dispatch workers until the scope is safe.
```

## Shared Worker Prompt

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

You own only the task assigned to you in COLLAB_CONTEXT.md.

Rules:
- Stay inside your task's write scope.
- Do not duplicate another agent's work.
- Do not expand scope without coordinator approval.
- When finished, update task status and add a handoff note with:
  - what changed
  - files touched
  - verification run
  - remaining risks
  - next recommended step
```

## Orchestrator And MCP Registry Coordination

### Human Kickoff

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md.

The task is:
Refactor the async orchestrator and MCP registry coordination path so dispatch ownership is clearer and the change is covered by targeted tests.

Codex should act as coordinator. Update the task board first, then assign:
- Gemini to read-only architecture analysis
- Codex to implementation
- Copilot to bounded tests or PR follow-through
- Kimi only if there is an isolated docs/helper slice

Do not allow overlapping file ownership.
```

### Gemini

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your assigned task:
Analyze the interaction points between:
- core/orchestrator.py
- core/mcp_agent_registry.py
- core/model_adapter.py
- aura_cli/cli_main.py
- docs/INTEGRATION_MAP.md

Goal:
Produce a read-only architecture handoff for Codex covering:
1. current control flow
2. dispatch boundaries
3. coupling points between orchestrator and registry
4. likely regression risks
5. recommended implementation order
6. exact file references for each finding

Constraints:
- Do not change runtime code.
- Only update docs/ or the COLLAB_CONTEXT.md handoff section if needed.
- Keep findings concise and implementation-oriented.
```

### Codex

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your assigned task:
Implement the orchestrator and MCP registry coordination refactor.

Primary write scope:
- core/orchestrator.py
- core/mcp_agent_registry.py
- core/agent_sdk/
- aura_cli/cli_main.py
- directly affected tests only if needed for implementation

Goals:
1. clarify dispatch ownership and integration boundaries
2. keep behavior stable outside the targeted path
3. avoid unrelated refactors
4. add or update only the tests required to verify the changed behavior
```

### Copilot Coding Agent

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your assigned task:
After Codex completes the orchestrator and registry refactor, own one bounded follow-up slice:

Preferred scope:
- add or repair targeted tests in tests/ or core/tests/
- or handle GitHub PR follow-through for the change set

Constraints:
- Do not edit:
  - core/orchestrator.py
  - core/mcp_agent_registry.py
  - aura_cli/cli_main.py
  unless the task board explicitly reassigns ownership
- Stay inside the test files or PR workflow assigned to you
- Do not expand scope into architecture changes
```

### Kimi

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your assigned task:
Take only an isolated sidecar task related to the orchestrator/registry refactor if the coordinator assigns one.

Allowed scope examples:
- docs updates under docs/
- helper cleanup in one non-central module
- script or notes support for verification

Not allowed:
- core/orchestrator.py
- core/mcp_agent_registry.py
- aura_cli/cli_main.py
- any file actively owned by Codex
```

## Test Coverage Expansion

### Coordinator

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

The task is:
Expand test coverage for a targeted subsystem in AURA CLI without broad refactors. Prefer missing-path coverage, regression coverage, and behavior-contract tests over snapshot churn.

Act as coordinator.

Routing:
- Gemini: analyze current coverage gaps, risky behaviors, and likely untested paths
- Codex: make any minimal production fixes required for testability and integrate final test plan
- Copilot coding agent: own bounded test-file additions and repairs
- Kimi: docs or isolated fixture/helper work only

Rules:
- One owner per test file.
- No overlapping edits on the same test module.
- Production code changes must stay minimal and justified by testability or bug exposure.
- Update COLLAB_CONTEXT.md before dispatching workers.
```

### Gemini

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Analyze the assigned subsystem for test coverage opportunities.

Produce:
1. highest-risk untested paths
2. edge cases worth covering
3. likely regression points
4. exact test modules to add or update
5. whether any production seams need minor improvement for testability
```

### Codex

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Own production-side integration for the coverage expansion.

Scope:
- review Gemini's findings
- make minimal production changes only if required
- keep behavioral scope tight
- integrate the final test plan
- run targeted verification
- update COLLAB_CONTEXT.md with risks and next steps
```

### Copilot Coding Agent

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Own bounded test additions for the assigned subsystem.

Constraints:
- Stay inside the test files assigned to you.
- Do not redesign production code.
- Prefer focused unit and integration coverage over broad snapshot rewrites.
- Report exactly which cases were added and which gaps remain.
```

### Kimi

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Take only isolated helper or fixture work related to the coverage effort.

Allowed scope:
- fixture files
- helper utilities for tests
- docs updates describing new test commands or coverage targets
```

## CLI Command Refactor

### Coordinator

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

The task is:
Refactor a CLI command path in AURA CLI while preserving command behavior, help text expectations, and documented output contracts except where intentionally changed.

Routing:
- Gemini: analyze current CLI flow, command parsing, help/docs/snapshot impact
- Codex: own command-path implementation across aura_cli/ and core integration points
- Copilot coding agent: own snapshot/help-doc updates and PR follow-through
- Kimi: own isolated docs or helper cleanup if non-overlapping

Rules:
- Preserve CLI contract unless explicitly changing it.
- Regenerate and verify CLI docs if command behavior changes.
- Update COLLAB_CONTEXT.md before work starts.
```

### Gemini

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Map the current CLI command flow for the assigned refactor.

Focus on:
- entrypoints in main.py and aura_cli/cli_main.py
- option parsing and dispatch
- help text and JSON output contracts
- tests and snapshot files that will be affected
- risks of breaking documented behavior
```

### Codex

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Implement the CLI command refactor.

Primary scope:
- aura_cli/
- main.py if needed
- directly related core wiring
- only the minimum affected tests/docs

Requirements:
- preserve intended command semantics
- update docs/snapshots if output changes intentionally
- run targeted CLI verification
- leave a handoff note with exact commands run
```

### Copilot Coding Agent

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Own bounded CLI follow-up work after implementation.

Preferred scope:
- help snapshot updates
- CLI docs regeneration checks
- docs-only updates
- PR workflow follow-through

Do not take ownership of core command dispatch files unless explicitly reassigned.
```

### Kimi

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Take only isolated side work for the CLI refactor, such as docs or helper cleanup in a non-overlapping scope.

Do not edit the same parser, dispatcher, or snapshot file currently owned by another worker.
```

## Agent SDK Controller Hardening

### Coordinator

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

The task is:
Harden the Agent SDK meta-controller path for reliability, fault tolerance, and clearer failure handling without broad architectural drift.

Routing:
- Gemini: analyze controller, tool registry, and resilience boundaries; identify failure modes and high-risk flows
- Codex: implement resilience and controller-path changes
- Copilot coding agent: add bounded tests or PR follow-through
- Kimi: docs or isolated helper support only

Priority areas:
- core/agent_sdk/controller.py
- core/agent_sdk/tool_registry.py
- core/agent_sdk/resilience.py
- related tests
```

### Gemini

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Perform read-only failure-mode analysis for the Agent SDK controller path.

Focus on:
- controller entry and exit points
- retry and circuit-breaker boundaries
- health-monitor interactions
- likely partial-failure paths
- missing observability or recovery seams
```

### Codex

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Implement the Agent SDK hardening changes.

Goals:
- improve resilience without widening scope
- preserve intended controller behavior
- add targeted tests where behavior changes
- document remaining failure modes in the handoff
```

### Copilot Coding Agent

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Own bounded follow-up after the Agent SDK hardening patch.

Preferred scope:
- targeted tests for new retry/circuit-breaker/controller behavior
- docs-only follow-up
- PR and review-thread handling

Do not expand into redesigning controller or resilience architecture.
```

### Kimi

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Take only isolated support work for the Agent SDK hardening effort, such as docs, helper scripts, or one bounded non-central module.

Do not edit controller or resilience core files owned by Codex.
```

## Goal Queue Cleanup

### Coordinator

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

The task is:
Clean up the goal queue path in AURA CLI, improving clarity and maintainability while preserving queue behavior and avoiding regressions in queue/archive interactions.

Routing:
- Gemini: analyze goal queue, archive, and loop touchpoints; identify stale assumptions and regression risks
- Codex: own implementation across core queue and loop integration
- Copilot coding agent: own bounded tests and PR follow-through
- Kimi: own isolated docs or helper cleanup only

Focus areas:
- core/goal_queue.py
- core/goal_archive.py
- orchestrator or loop call sites that depend on queue behavior
```

### Gemini

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Produce a read-only analysis of the goal queue path.

Focus on:
- current queue lifecycle
- archive interactions
- old vs current assumptions
- serialization or persistence risks
- tests that should exist or be updated
```

### Codex

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Implement the goal queue cleanup.

Requirements:
- keep queue behavior stable unless explicitly changing it
- avoid unrelated loop refactors
- update targeted tests
- leave clear notes on persistence and migration risk if any
```

### Copilot Coding Agent

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Own bounded test and GitHub follow-up work for the goal queue cleanup.

Scope:
- queue-related tests
- docs-only updates
- PR and review-thread handling

Do not change queue core logic unless the task board explicitly reassigns ownership.
```

### Kimi

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

Your task:
Take only isolated helper, script, or docs work related to the goal queue cleanup.

Do not overlap with queue core files or actively owned tests.
```
