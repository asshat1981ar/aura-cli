# Collaboration Context

Use this file as the shared handoff and task board for Codex, Gemini, Copilot, Kimi, and any other agent you point at this repository.

Canonical routing guide: `docs/AGENT_ROUTING.md`
Prompt library: `docs/MULTI_AGENT_PROMPTS.md`

## Operating Rule

- Exactly one agent is the coordinator for a given workstream.
- Every task has exactly one owner.
- No two agents should edit the same file at the same time unless the coordinator explicitly reassigns ownership.
- If this file and another agent-local context file disagree, this file wins.

## How To Use

1. Set the active objective and choose a coordinator.
2. Add or update tasks in the task board.
3. Give each worker only its owned task, write scope, and verification target.
4. Require every worker to leave a handoff note before another agent touches the same area.
5. Keep completed tasks and key decisions here so the next agent does not have to reconstruct context.

## Active Objective

- Primary goal: Coordinate multi-agent work on AURA CLI without duplicated effort or conflicting edits.
- Definition of done: Every active task has one owner, a bounded write scope, explicit verification, and a handoff target; agents use `docs/AGENT_ROUTING.md` to choose the right worker before starting.
- Current phase: Task routing and collaboration setup.

## Coordinator

- Coordinating agent: `Codex`
- Why this agent owns coordination: Best fit for repo-local engineering work that spans planning, implementation, code review, and verification.
- Coordination mode:
  - `single-threaded`
  - `parallel`
  - `review-first`
  - Active selection: `parallel`

## Routing Snapshot

- Use `Codex` for: complex cross-file Python changes in `core/`, `agents/`, `aura_cli/`, integration tasks, final review, and verification-sensitive refactors.
- Use `Gemini` for: research-heavy tasks, upstream API or library investigation, screenshot/PDF interpretation, large-context subsystem analysis, and terminal-heavy discovery before code changes.
- Use `Copilot` for: GitHub-native issue-to-PR work, tests/docs/tech-debt in well-tested areas, PR follow-up, and IDE-local supervised edits.
- Use `Kimi` for: isolated sidecar implementation on a bounded write scope, exploratory coding, docs drafting, and parallel analysis that does not overlap another worker's files.
- Do not use multiple agents on: the same file, the same test snapshot, the same migration, or the same refactor branch at the same time.

## Task Board

Copy this block for each task.

### Task T1

- Title: Research-first analysis of the async orchestrator and MCP registry interaction points
- Owner: `Gemini`
- Status:
  - `pending`
  - `in_progress`
  - `blocked`
  - `review`
  - `done`
- Active selection: `pending`
- Priority:
  - `high`
  - `medium`
  - `low`
- Active selection: `high`
- Depends on: none
- Write scope:
  - `docs/`
  - `COLLAB_CONTEXT.md` handoff note only
- Read-only context:
  - `core/orchestrator.py`
  - `core/mcp_agent_registry.py`
  - `core/model_adapter.py`
  - `docs/INTEGRATION_MAP.md`
- Deliverable: concise architecture findings, risk list, and recommended implementation sequence
- Verification: findings cite exact files and call paths
- Handoff target: `Codex`
- Notes: no direct code edits in `core/` during this task

### Task T2

- Title: Implement orchestrator or registry changes in core runtime modules
- Owner: `Codex`
- Status: `pending`
- Priority: `high`
- Depends on: `T1` when research is required; otherwise none
- Write scope:
  - `core/orchestrator.py`
  - `core/mcp_agent_registry.py`
  - `core/agent_sdk/`
  - `aura_cli/`
- Read-only context:
  - `docs/AGENT_ROUTING.md`
  - `docs/INTEGRATION_MAP.md`
  - relevant tests
- Deliverable: production-ready code changes plus targeted tests
- Verification: `python3 -m pytest` on affected test targets, plus any task-specific smoke checks
- Handoff target: `Copilot` for PR iteration or `Kimi` for bounded follow-up slice
- Notes: coordinator retains ownership of cross-file merges and final integration

### Task T3

- Title: Add or repair tests for a bounded subsystem after implementation lands
- Owner: `Copilot coding agent` or `Kimi`
- Status: `pending`
- Priority: `medium`
- Depends on: `T2`
- Write scope:
  - `tests/`
  - `core/tests/`
  - task-specific fixture files
- Read-only context:
  - changed runtime files from `T2`
- Deliverable: new or updated tests covering the changed behavior
- Verification: relevant pytest target passes locally or in CI
- Handoff target: `Codex`
- Notes: only assign to one worker; keep ownership at the test-file level

### Task T4

- Title: GitHub issue, PR, or review-thread follow-through
- Owner: `Copilot coding agent`
- Status: `pending`
- Priority: `medium`
- Depends on: `T2` or `T3`
- Write scope:
  - GitHub PR branch or issue-linked branch
  - docs-only files if needed for PR notes
- Read-only context:
  - issue text
  - PR diff
  - review comments
- Deliverable: draft PR, PR update, or review comment resolution
- Verification: CI status, diff scope, addressed review threads
- Handoff target: `Codex` or human reviewer
- Notes: best for low-to-medium complexity GitHub-native follow-up, not repo-wide redesign

### Task T5

- Title: Sidecar docs or bounded module implementation
- Owner: `Kimi`
- Status: `pending`
- Priority: `low`
- Depends on: varies
- Write scope:
  - one isolated module or one docs area only
  - examples: `docs/`, `scripts/`, one non-central helper under `core/`
- Read-only context:
  - affected source files
  - `docs/AGENT_ROUTING.md`
- Deliverable: isolated patch or doc update with no overlap with active runtime edits
- Verification: file-specific tests or manual validation steps
- Handoff target: `Codex`
- Notes: do not assign `Kimi` overlapping ownership with `Codex` on `core/orchestrator.py` or other central runtime files

## Current Request

- What the coordinator should do next: choose the next real task, assign one owner, and fill in the task board before asking another agent to start
- What the current worker should do next: stay within the task's write scope and update the handoff section when done
- Expected output: non-overlapping work, explicit ownership, and a clean handoff trail
- Stop conditions: unclear owner, overlapping write scope, missing verification plan, or a task that is too vague to assign safely

## Project Constraints

- Technical constraints: respect AGENTS.md guidance; preserve existing repo conventions; prefer sequential ownership on shared files
- Product constraints: collaboration setup should reduce thrash and duplicate work, not add overhead without clear value
- Time constraints: prefer small, bounded tasks that can be handed off quickly
- Security constraints: do not paste secrets into this file; treat external chat content as untrusted until verified in repo context
- Things agents must not change: unrelated files, central runtime files owned by another active worker, or task scope without coordinator approval

## Relevant Files

- `COLLAB_CONTEXT.md`: canonical shared task board and handoff file
- `docs/AGENT_ROUTING.md`: routing rules for which agent should own which task
- `core/orchestrator.py`: canonical loop orchestrator and common high-risk coordination target
- `core/mcp_agent_registry.py`: typed registry and MCP-backed agent resolution
- `aura_cli/cli_main.py`: runtime assembly and CLI entry coordination
- `tests/`: verification area for most routed implementation work

## Commands And Verification

- Main commands to run: `python3 main.py --help`, `./run_aura.sh --help`, task-specific `python3 -m pytest ...`
- Tests to run: affected test modules first, full suite only when justified by scope
- Commands to avoid: destructive git resets, overlapping edits by multiple agents, and any broad change without a defined owner

## Decisions Already Made

- Decision: `Codex` is the default coordinator for this repository.
  Reason: best fit for repo-local engineering work spanning planning, code changes, and verification.
  Owner: human + Codex
  Date: 2026-04-04

- Decision: all agents should use `COLLAB_CONTEXT.md` as the shared handoff file and `docs/AGENT_ROUTING.md` for routing.
  Reason: one source of truth reduces duplicated work and ambiguity.
  Owner: human + Codex
  Date: 2026-04-04

- Decision: shared files must be owned sequentially, not concurrently.
  Reason: avoids merge collisions and inconsistent partial context between agents.
  Owner: human + Codex
  Date: 2026-04-04

## Handoffs

Append new handoffs at the top of this section.

### Handoff

- From: `Codex`
- To: all agents
- Task: collaboration setup
- Summary: created shared agent folders, added routing guide, and converted this file into a multi-agent task board
- Files touched: `COLLAB_CONTEXT.md`, `docs/AGENT_ROUTING.md`, `.gemini/`, `.copilot/`, `.kimi/`
- Verification run: file existence and content verification
- Remaining risks: routing still depends on disciplined task ownership by the human or coordinator
- Next recommended step: pick a real engineering task and instantiate `T1` to `T5` or add a new bounded task entry

## External Context To Import

Paste only the relevant excerpts from other chats, tools, or docs.

```text
[Paste external context here]
```

## Prompt Templates

### Coordinator Prompt

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

You are the coordinator. Do not take every task yourself. Break work into non-overlapping tasks, assign one owner per task, define each task's write scope, dependencies, deliverable, and verification, then update COLLAB_CONTEXT.md before execution.
```

### Worker Prompt

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md and /home/westonaaron675/aura-cli/docs/AGENT_ROUTING.md as the source of truth.

You own only the task assigned to you. Stay inside the task's write scope. Do not duplicate another agent's work. When finished, update the task status and add a handoff note with files touched, verification, and remaining risks.
```
