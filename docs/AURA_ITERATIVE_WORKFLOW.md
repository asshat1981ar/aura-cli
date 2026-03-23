# AURA Iterative Development Workflow

This workflow is the companion operating procedure for the AURA operator prompt. It is tuned for daily coding work in this repository.

For bounded multi-agent delegation, CI triage, and PR-unblocking policy, use [AURA_MULTI_AGENT_WORKFLOW.md](./AURA_MULTI_AGENT_WORKFLOW.md) as the detailed execution playbook. This file stays the day-to-day loop; the multi-agent doc defines exact subagent roles, spawn triggers, and reconciliation rules. Use [AURA_SWEEP_TEMPLATES.md](./AURA_SWEEP_TEMPLATES.md) when the task expands into audit-first branch stabilization or repo-wide sweep work.

## 1. Orient

- Translate the request into a concrete engineering task.
- If the task is a broad cleanup, CI sweep, or multi-PR push, start with an audit summary before deciding what to edit.
- Identify the active subsystem first:
  - CLI and runtime: `main.py`, `aura_cli/cli_main.py`
  - orchestration and agents: `core/orchestrator.py`, `agents/`
  - operator-facing behavior: `core/operator_runtime.py`, `aura_cli/commands.py`, server/TUI surfaces
  - MCP or external integrations: `.mcp.json`, `core/mcp_*`, configured MCP servers
- Use `docs/INTEGRATION_MAP.md` to confirm runtime boundaries before going wider.

## 2. Map The Relevant Slice

- Trace the smallest real execution path that owns the behavior.
- Read nearby tests and contracts before editing code.
- If the task came from a GitHub issue or PR, read the issue/PR text, linked comments, and requested acceptance conditions before changing code.
- Identify:
  - entrypoint
  - downstream modules
  - state or persistence layer
  - external interfaces
  - likely verification points
- Ignore unrelated architecture once the owning slice is clear.

## 3. Choose The Working Mode

- **Local-first mode**
  - Default for most feature work, bug fixes, and refactors.
  - Use shell inspection, local files, and repo tests first.

- **Tool-assisted mode**
  - Use when docs, browser behavior, external APIs, or structured service access matter.
  - Prefer the existing MCP setup over ad hoc new tooling.

- **Delegated mode**
  - Use when bounded work can run in parallel without blocking the next local step.
  - Good fits:
    - targeted codebase exploration
    - regression review
    - isolated test writing or implementation slices

## 4. Delegate With Boundaries

- Keep the main agent on the critical path.
- Delegate only tasks with a clear output and narrow scope.
- Use explorers for specific codebase questions.
- Use workers for isolated implementation tasks with disjoint write ownership.
- Give each delegated task:
  - the exact question or output needed
  - the file or subsystem boundary
  - the warning not to revert unrelated edits
- For broad sweep work, use the repo packet format from `docs/AURA_SWEEP_TEMPLATES.md` instead of ad hoc prompts.

Default delegation triggers:

- Spawn an explorer when the next decision depends on locating the right code path quickly.
- Spawn a worker when implementation can be split cleanly and the main agent can continue on a separate track.
- Do not spawn if the answer is needed immediately for the very next step and the task is small enough to do locally.
- When the task is CI stabilization or PR unblocking, follow the exact main-agent/subagent split in `docs/AURA_MULTI_AGENT_WORKFLOW.md` instead of improvising new agent roles.

## 5. Use Tools By Evidence Need

- Use shell tools first for:
  - symbol search
  - file discovery
  - config inspection
  - test discovery
  - narrow verification
- Use MCP servers for:
  - structured docs lookup
  - external systems or APIs
  - browser workflows
  - GitHub or service integrations
  - memory/context retrieval when relevant
- Use skills when they directly match the task, such as:
  - OpenAI docs guidance
  - Smithery/MCP setup and discovery
  - skill authoring or installation

## 6. Handle PRs, Issues, And Comments Explicitly

- For issue-driven work:
  - extract the reported problem, expected behavior, and any linked repro
  - confirm the issue against code or tests before accepting the proposed cause
  - keep the fix scoped to the validated problem

- For PR-driven work:
  - align the implementation summary with `.github/PULL_REQUEST_TEMPLATE.md`
  - use `docs/PR_REVIEWER_SUMMARY_TEMPLATE.md` when the branch needs a reviewer-facing closeout after CI or comment repair
  - keep track of linked issues, type of change, and testing notes
  - distinguish between code-complete and reviewer-complete

- For review comments:
  - group comments into:
    - requires code change
    - requires explanation
    - requires product or architecture decision
  - resolve each group deliberately instead of mixing them into one broad edit pass
  - if a comment cannot be fully resolved, state the blocker and the smallest next step

- For stale or closure-sensitive PRs:
- use `docs/GITHUB_TRIAGE_POLICY.md` as the repo source of truth
- preserve unblock conditions in the final summary when relevant
- for multi-agent CI or review triage, keep Actions/workflow diagnosis aligned with `docs/AURA_MULTI_AGENT_WORKFLOW.md`

## 7. Implement In Tight Loops

- Make the smallest coherent code change that advances the task.
- Re-check assumptions after each meaningful edit.
- Run narrow verification early.
- Expand verification only after the local slice looks correct.

Preferred verification order:

1. targeted unit or snapshot tests
2. subsystem-focused checks
3. broader regression tests when risk justifies them

## 8. Close The Session Cleanly

- Report:
  - what changed
  - what was verified
  - what remains uncertain
- For audit-first or sweep work, report the next queue item using the closeout format from `docs/AURA_SWEEP_TEMPLATES.md`.
- If the work is PR-backed, pair the internal closeout with the reviewer summary format in `docs/PR_REVIEWER_SUMMARY_TEMPLATE.md`.
- If the work addressed review feedback, state which comments are now resolved and which still need follow-up.
- Mention the owning files and interfaces, not every touched line.
- If verification was partial, state the exact gap.

## Default Session Template

Use this operating shape for most tasks:

1. Inspect the active entrypoint and owning subsystem.
2. Read the nearest architecture, workflow, issue, or PR context if needed.
3. Trace the execution path and adjacent tests.
4. State the intended change, review target, and verification target.
5. Delegate only bounded sidecar work.
6. Implement the smallest viable slice.
7. Run targeted verification.
8. Summarize outcome, tests, resolved comments, and residual risk.

## Acceptance Standard

The workflow is being followed correctly when:

- decisions are based on repo evidence
- delegation is bounded and non-duplicative
- tool use is justified by uncertainty, not habit
- edits are incremental
- verification is explicit
- the final summary is concise and operational
