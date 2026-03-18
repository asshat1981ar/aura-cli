# GitHub Agent Automation Implementation Plan

**Date:** 2026-03-18
**Status:** Proposed
**Scope:** Add governed GitHub automations for issue intake, code review, coding-agent execution, PR merge readiness, and nightly repo maintenance across Copilot, Codex, Claude, Gemini, and AURA.

---

## 1. Objective

Build a single GitHub automation system for this repo with these properties:

- GitHub is the control plane for issues, PRs, reviews, labels, merge queue, and audit trail.
- AURA is the orchestration layer that normalizes findings, applies repo-specific policy, and decides when to stop, retry, or escalate.
- Copilot, Codex, Claude, and Gemini are specialized workers, not independent sources of truth.
- Human reviewers retain final authority for risky changes and production merges.

Success means:

- New issues can be routed into implementation-ready plans.
- PRs receive one synthesized review result instead of duplicate bot noise.
- Agents can help with coding tasks without bypassing CI, CODEOWNERS, or merge rules.
- Merge queue becomes the only autonomous path to `main`.

---

## 2. Current Starting Point

The repo already has useful partial building blocks:

- CI workflow: `.github/workflows/ci.yml`
- Copilot issue intake prototype: `.github/workflows/copilot-workspace.yml`
- Copilot PR summary prototype: `.github/workflows/copilot-autofix.yml`
- Gemini PR summary prototype: `.github/workflows/gemini-code-assist.yml`
- Scheduled autonomous loop: `.github/workflows/aura-agentic-loop.yml`
- One custom agent profile: `.github/agents/agentic-workflows-dev.agent.md`
- Reusable workflow engine: `core/workflow_engine.py`
- Human approval gate: `core/human_gate.py`
- GitHub API client: `tools/github_tools.py`
- GitHub MCP bridge: `aura_cli/mcp_github_bridge.py`
- Copilot-style GitHub MCP server: `tools/github_copilot_mcp.py`

Key gaps:

- No `CODEOWNERS`
- No issue forms
- No PR template
- No merge queue readiness in CI (`merge_group` is missing)
- No unified PR review orchestrator
- No normalized provider output schema
- Current bot workflows comment directly on PRs and can create duplicate reporting

---

## 3. Target Architecture

```text
Issue / PR / Comment Event
        |
        v
GitHub Actions Router
        |
        v
AURA GitHub Automation Layer
  - provider router
  - review synthesizer
  - policy engine
  - human gate
        |
        +--> Copilot
        +--> Codex
        +--> Claude
        +--> Gemini
        +--> Local AURA skills
        |
        v
Normalized Result
  - check run
  - single summary comment
  - labels
  - issue or follow-up task
        |
        v
GitHub Rulesets + Merge Queue
```

Design rules:

1. Deterministic checks run before model-driven review.
2. Model outputs are converted into one repo-owned schema.
3. Only one workflow posts the final synthesized review comment.
4. Agents never merge directly; GitHub auto-merge and merge queue handle merge once checks pass.

---

## 4. Repo Changes By Phase

## Phase 0: Governance Baseline

**Goal:** Make GitHub safe and predictable before adding more agent power.
**Status:** Completed

### Files created
- `.github/CODEOWNERS`
- `.github/pull_request_template.md`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/ISSUE_TEMPLATE/agent_task.yml`
- `.github/ISSUE_TEMPLATE/config.yml`

### Concrete work
1. Add `merge_group` to `.github/workflows/ci.yml`.
2. Add required-review expectations to the PR template.
3. Add issue forms for bug reports, feature requests, and agent tasks.
4. Add baseline labels.
5. Add `CODEOWNERS` for sensitive paths.

---

## Phase 1: Unified PR Review Orchestrator

**Goal:** Replace comment spam with one synthesized PR review pipeline.
**Status:** Completed

### Files created
- `.github/workflows/pr-review-orchestrator.yml`
- `core/github_automation/__init__.py`
- `core/github_automation/models.py`
- `core/github_automation/provider_router.py`
- `core/github_automation/review_synthesizer.py`
- `core/github_automation/policy.py`
- `core/github_automation/pr_context.py`

### Concrete work
1. Introduced a normalized review schema.
2. Add a router that decides which providers run for a PR.
3. Post exactly one final summary comment and one final check run.
4. Use `core/human_gate.py` to block autonomous approval on risky changes.

---

## Phase 2: Issue Intake, Planning, and Task Generation

**Goal:** Turn issues into high-quality implementation tasks instead of raw queue entries.
**Status:** Completed

### Files created
- `.github/workflows/issue-intake.yml`
- `.github/workflows/issue-comment-commands.yml`
- `core/github_automation/issue_triage.py`
- `core/github_automation/issue_planner.py`
- `core/github_automation/command_router.py`

### Concrete work
1. Structured parsing from issue forms.
2. Generate a normalized issue plan (problem statement, candidate files, suggested tests).
3. Add slash-command handling on `issue_comment` (`/plan`, `/review`, `/queue`).

---

## Phase 3: Coding-Agent Execution Lanes

**Goal:** Support Copilot, Codex, Claude, and optionally Gemini-assisted code changes under policy.
**Status:** Completed

### Files created
- `.github/agents/pr-reviewer.agent.md`
- `.github/agents/issue-planner.agent.md`
- `.github/agents/bugfix.agent.md`
- `.github/workflows/agent-task-dispatch.yml`
- `core/github_automation/agent_dispatch.py`
- `core/github_automation/agent_profiles.py`

### Concrete work
1. Split into role-specific agent profiles.
2. Add dispatch rules for Copilot, Codex, Claude, and Gemini.
3. Require all coding-agent lanes to operate on branches or PRs only.

---

## Phase 4: Merge Governance and Auto-Merge

**Goal:** Make merge automation reliable and boring.
**Status:** Completed

### Files created
- `.github/workflows/merge-readiness.yml`
- `core/github_automation/merge_advisor.py`

### Concrete work
1. Add a merge readiness check consuming CI, review output, and human gate status.
2. Gate `merge-ready` label assignment on policy compliance.
3. Enable GitHub merge queue support.

---

## Phase 5: Nightly and Asynchronous Automation

**Goal:** Keep long-running automation out of the PR critical path.
**Status:** Completed

### Files created
- `.github/workflows/nightly-repo-health.yml`
- `core/github_automation/repo_health.py`

### Concrete work
1. Nightly quality scans and backlog grooming.
2. Repo health summaries (stale PRs, flaky tests, failure clusters).
3. Automatic follow-up issue recommendations.

---

## 5. Definition of Done (VERIFIED)

- [x] PRs receive one synthesized automation verdict instead of multiple bot summaries
- [x] risky changes require human review automatically
- [x] issue intake produces structured plans
- [x] coding agents operate only through branches and PRs
- [x] merge queue handles safe merges after required checks pass
- [x] nightly automation is useful but not on the critical path

