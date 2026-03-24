# AURA Multi-Agent Workflow

This document is the repo-specific execution playbook for bounded multi-agent work in AURA. It is optimized first for CI stabilization, PR unblocking, review-driven development, and the adjacent developer-surface refinements that keep those workflows stable.

Use `docs/AURA_SWEEP_TEMPLATES.md` for the audit, delegation packet, queue, and closeout formats referenced below.
Use `docs/ACTIVE_SWEEP_STATUS.md` as the live branch-local status surface when a sweep is in progress.

## 1. Main-Agent Ownership

The main agent always owns:

- task framing and success criteria
- branch and SHA truth
- final choice of what to change
- final integration of parallel findings
- final verification interpretation
- final user-facing and reviewer-facing summary

The main agent does not delegate:

- merge or close decisions
- final reconciliation of conflicting edits
- final judgment about whether a PR is ready
- branch mutations without first integrating evidence

## 2. Approved Subagent Roles

Use only bounded roles with explicit outputs.

### CI Explorer

- Input: failing run IDs, job names, branch SHA
- Output:
  - failure class
  - owning subsystem
  - likely repro path
  - smallest verification target
- Write scope: none

### Workflow Explorer

- Input: selected `.github/workflows/*.yml` files and recent run metadata
- Output:
  - YAML/compiler risks
  - GitHub expression/permission/secret risks
  - smallest safe workflow fix
- Write scope: none

### Runtime Explorer

- Input: active entrypoint, failing subsystem, target tests
- Output:
  - exact call path
  - root-cause candidates
  - smallest code/test fix surface
- Write scope: none

### Fix Worker

- Input: one isolated failure slice and explicit file boundary
- Output:
  - implementation patch in assigned files only
  - verification notes for that slice
- Write scope: disjoint and explicit

### Verification Worker

- Input: touched files and named checks
- Output:
  - pass/fail summary
  - residual risks
  - missing verification branches
- Write scope: none

## 3. Spawn Rules

Spawn subagents only when at least one is true:

- there are multiple independent failing surfaces
- the next decision depends on disjoint investigation
- verification can run in parallel with local implementation
- one implementation slice can be isolated safely from the main path

Do not spawn when:

- the answer is needed for the immediate next local edit
- the write scope is shared or ambiguous
- the task is deciding what to do next

## 4. Delegation Packet

Every delegated task must specify:

- objective
- branch and HEAD SHA
- exact subsystem or file boundary
- expected output format
- verification target
- warning not to revert unrelated edits

Default expected output format:

1. root cause or answer
2. smallest fix surface
3. verification target
4. residual risk

Use the exact `Delegation Packet` template from `docs/AURA_SWEEP_TEMPLATES.md` unless the task needs a stricter schema.

## 5. Execution Loop

### Audit First

Before any mutating work in a broad sweep:

- produce an audit summary for the current branch and SHA
- classify unrelated local worktree changes as:
  - out of scope
  - adjacent but untouched
  - high-risk overlap
- identify the active workflows, PRs, review comments, and developer-surface docs in scope
- choose one first bucket from the priority order below

Use the `Audit Summary` and `Sweep Queue` templates from `docs/AURA_SWEEP_TEMPLATES.md`.
Persist the current sweep state in `docs/ACTIVE_SWEEP_STATUS.md` when the work extends beyond a single fix.

### Intake

The main agent starts by collecting:

- current branch and SHA
- latest PR-linked Actions runs for that SHA
- active review comments if the branch is PR-backed
- narrow local verification status for the likely owning subsystem

Then classify the work as one of:

- workflow/compiler failure
- runtime/test regression
- provider workflow failure
- review-driven code fix
- mixed

### Priority Order

Handle work in this order:

1. invalid workflows or no-job failures
2. failing required CI test lanes
3. blocking PR review comments
4. provider-specific automation failures
5. developer-surface drift that affects future CI/PR work
6. advisory workflows

### Parallel Recon

When there is more than one plausible surface:

- spawn CI Explorer on failing runs
- spawn Workflow Explorer on workflow-file or setup-time failures
- spawn Runtime Explorer on the suspected code/test owner

The main agent remains local on the critical path and chooses exactly one next action after explorers return.

### Decision Gate

The main agent must choose one of:

- workflow fix
- code or test fix
- no code change, only rerun/report
- externally blocked

Decision rules:

- choose workflow fixes first when GitHub cannot create jobs or fails before real steps
- choose code/test fixes next when required CI lanes fail in reproducible code paths
- classify as externally blocked when a failure depends on missing secrets, provider auth, or GitHub config outside the repo
- do not combine independent fixes in one commit unless they share one root cause

### Implementation Split

- the first critical-path fix stays local to the main agent
- a Fix Worker may take one disjoint side slice only if files do not overlap
- each commit should correspond to one failure class
- after the active error buckets are clean or classified, a separate refinement pass may update docs, prompts, and agent guidance using the developer-surface checklist from `docs/AURA_SWEEP_TEMPLATES.md`

Current branch policy:

- if the active branch is the target PR branch, fix directly on that branch
- if another PR branch is the target, use an isolated `/tmp` worktree
- never revert unrelated working-tree changes

## 6. Verification Protocol

Use this order:

1. targeted local tests for the changed subsystem
2. local YAML validation when workflows changed
3. inspect new Actions runs for the changed SHA
4. broaden verification only if the narrow checks pass

Required verification by failure class:

- workflow issue:
  - validate all workflow YAML locally
  - confirm the new run creates real jobs
- runtime/test regression:
  - run targeted pytest selection for the affected slice
  - confirm platform or version boundaries when applicable
- PR review fix:
  - run the smallest check that proves the review concern is resolved
- provider workflow issue:
  - validate workflow syntax and allowed expressions
  - separate repo-fixable issues from missing-secret or platform blocks

## 7. Human Gates

Always keep human or main-agent control over:

- merge and close decisions
- workflow/auth/secrets policy changes
- protected-path changes
- provider-selection policy
- product or design ambiguity

## 8. Closeout Contract

Every work cycle ends with:

- current branch and SHA
- failure class addressed
- verification performed
- status:
  - resolved
  - partially resolved
  - externally blocked
- next highest-priority remaining failure

For PR-backed work, also report:

- which review comment or CI lane was addressed
- whether the remaining queue changed after the fix

Use the `Closeout` template from `docs/AURA_SWEEP_TEMPLATES.md` for consistency across main-agent and subagent reporting.
- whether reviewer follow-up is still needed

## 9. Current-Codebase Defaults

- Treat `main.py` -> `aura_cli/cli_main.py` -> `core/orchestrator.py` as the active runtime path.
- Treat `core/workflow_engine.py` as a secondary, stepwise workflow surface, not the primary multi-agent scheduler.
- Treat GitHub Actions workflows as the current GitHub automation control plane on this branch.
- Keep workflow YAML thin over time; move reusable issue/PR/review logic into shared scripts or core modules when the repo is ready for that consolidation.
