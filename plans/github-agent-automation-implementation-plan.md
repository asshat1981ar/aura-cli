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

### Files to create

- `.github/CODEOWNERS`
- `.github/pull_request_template.md`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/ISSUE_TEMPLATE/agent_task.yml`
- `.github/ISSUE_TEMPLATE/config.yml`
- `.github/labels.yml` or `docs/github-label-taxonomy.md`

### Files to modify

- `.github/workflows/ci.yml`
- `.github/copilot-instructions.md`
- `README.md`

### Concrete work

1. Add `merge_group` to `.github/workflows/ci.yml`.
2. Add required-review expectations to the PR template:
   - risk summary
   - test evidence
   - rollout notes
   - agent involvement disclosure
3. Add issue forms for:
   - bug report
   - feature request
   - agent task / automation request
4. Add baseline labels:
   - `agent-task`
   - `agent-review`
   - `needs-human-review`
   - `merge-ready`
   - `blocked`
   - `risk:low`
   - `risk:medium`
   - `risk:high`
   - `provider:copilot`
   - `provider:codex`
   - `provider:claude`
   - `provider:gemini`
5. Add `CODEOWNERS` for at least:
   - `.github/`
   - `core/`
   - `aura_cli/`
   - `tools/`
6. Update `.github/copilot-instructions.md` so agent workflows know:
   - not to touch runtime memory files
   - not to bypass CLI snapshot contracts
   - not to self-merge
   - to report findings in normalized severity buckets

### Acceptance criteria

- CI runs for `pull_request`, `push`, and `merge_group`
- All new PRs use a consistent template
- Issues can be opened through forms instead of free-form titles only
- CODEOWNERS can gate sensitive paths

---

## Phase 1: Unified PR Review Orchestrator

**Goal:** Replace comment spam with one synthesized PR review pipeline.

### Files to create

- `.github/workflows/pr-review-orchestrator.yml`
- `core/github_automation/__init__.py`
- `core/github_automation/models.py`
- `core/github_automation/provider_router.py`
- `core/github_automation/review_synthesizer.py`
- `core/github_automation/policy.py`
- `core/github_automation/pr_context.py`
- `tests/test_github_review_synthesizer.py`
- `tests/test_github_review_policy.py`
- `tests/test_github_provider_router.py`

### Files to modify

- `.github/workflows/copilot-autofix.yml`
- `.github/workflows/gemini-code-assist.yml`
- `tools/github_copilot_mcp.py`
- `tools/github_tools.py`
- `core/human_gate.py`

### Concrete work

1. Introduce a normalized review schema:

```python
{
    "provider": "copilot|codex|claude|gemini|aura",
    "summary": str,
    "findings": [
        {
            "severity": "critical|high|medium|low|info",
            "path": str,
            "line": int | None,
            "title": str,
            "detail": str,
            "confidence": float,
            "category": "bug|security|correctness|test|maintainability"
        }
    ],
    "recommended_action": "approve|comment|request_changes|escalate",
    "artifacts": {}
}
```

2. Add a router that decides which providers run for a PR:
   - small docs change: Copilot + local checks only
   - Python core change: Copilot + Gemini + AURA skills
   - risky infra/security change: Copilot + Gemini + Claude or Codex + human gate
3. Convert `.github/workflows/copilot-autofix.yml` and `.github/workflows/gemini-code-assist.yml` into provider-specific data collectors, or retire them once the orchestrator is stable.
4. Post exactly one final summary comment and one final check run:
   - summary of findings by severity
   - links to artifacts
   - merge readiness verdict
5. Use `core/human_gate.py` to block autonomous approval when:
   - security findings are critical
   - coverage drops beyond threshold
   - dependency lockfiles or workflow files changed
   - `CODEOWNERS` protected paths changed

### Acceptance criteria

- A PR receives one synthesized review status
- Severity thresholds map to merge eligibility
- Risky PRs are flagged for human review automatically

---

## Phase 2: Issue Intake, Planning, and Task Generation

**Goal:** Turn issues into high-quality implementation tasks instead of raw queue entries.

### Files to create

- `.github/workflows/issue-intake.yml`
- `.github/workflows/issue-comment-commands.yml`
- `core/github_automation/issue_triage.py`
- `core/github_automation/issue_planner.py`
- `core/github_automation/command_router.py`
- `tests/test_issue_triage.py`
- `tests/test_issue_planner.py`
- `tests/test_issue_comment_commands.py`

### Files to modify

- `.github/workflows/copilot-workspace.yml`
- `tools/github_copilot_mcp.py`
- `core/workflow_engine.py`
- `tools/github_tools.py`

### Concrete work

1. Replace the current title-only issue intake in `.github/workflows/copilot-workspace.yml` with structured parsing from issue forms.
2. Generate a normalized issue plan:
   - problem statement
   - affected areas
   - candidate files
   - suggested tests
   - risk score
   - recommended provider lane
3. Add slash-command handling on `issue_comment`:
   - `/plan`
   - `/review copilot`
   - `/review gemini`
   - `/review claude`
   - `/review codex`
   - `/queue aura`
4. Store issue planning outputs as:
   - comment
   - JSON artifact
   - optional AURA queue item
5. Reuse `core/workflow_engine.py` for multi-step issue triage flows rather than embedding all logic in Actions YAML.

### Acceptance criteria

- New issues can be converted into structured plans
- Issue comments can trigger follow-up automation safely
- Planning artifacts are consistent across providers

---

## Phase 3: Coding-Agent Execution Lanes

**Goal:** Support Copilot, Codex, Claude, and optionally Gemini-assisted code changes under policy.

### Files to create

- `.github/agents/pr-reviewer.agent.md`
- `.github/agents/issue-planner.agent.md`
- `.github/agents/bugfix.agent.md`
- `.github/workflows/agent-task-dispatch.yml`
- `core/github_automation/agent_dispatch.py`
- `core/github_automation/agent_profiles.py`
- `tests/test_agent_dispatch.py`

### Files to modify

- `.github/agents/agentic-workflows-dev.agent.md`
- `.github/copilot-instructions.md`
- `aura_cli/mcp_github_bridge.py`
- `tools/github_copilot_mcp.py`

### Concrete work

1. Split the single custom agent profile into role-specific profiles:
   - `pr-reviewer`
   - `issue-planner`
   - `bugfix`
   - keep `agentic-workflows-dev` for workflow architecture work
2. Add dispatch rules:
   - Copilot: lightweight repo-native coding assistance and custom agent prompts
   - Codex: implementation-heavy issue work and patch generation
   - Claude: deeper review/refactor reasoning and patch proposals
   - Gemini: review summaries, risk explanations, and alternate review pass
3. Require all coding-agent lanes to operate on branches or PRs only.
4. Prohibit direct writes to `main`.
5. Save all generated patch metadata as artifacts so a human can audit which provider proposed what.

### Acceptance criteria

- Agent jobs are role-based and auditable
- Provider selection is explicit instead of ad hoc
- Branch-only execution is enforced

---

## Phase 4: Merge Governance and Auto-Merge

**Goal:** Make merge automation reliable and boring.

### GitHub settings to configure

- Branch protection or rulesets on `main`
- Required status checks
- Required pull request reviews
- CODEOWNERS review requirement
- Auto-merge
- Merge queue

### Files to create

- `.github/workflows/merge-readiness.yml`
- `core/github_automation/merge_advisor.py`
- `tests/test_merge_advisor.py`

### Files to modify

- `.github/workflows/ci.yml`
- `.github/workflows/pr-review-orchestrator.yml`
- `core/human_gate.py`

### Concrete work

1. Add a merge readiness check that consumes:
   - CI results
   - synthesized review output
   - human gate result
   - CODEOWNERS state
2. Gate `merge-ready` label assignment on:
   - no critical findings
   - no blocked human gate rule
   - required tests passed
   - required approvals present
3. Enable GitHub merge queue after `merge_group` CI is proven stable.
4. Treat workflows, `.github/`, secrets-handling code, and dependency updates as higher-risk paths requiring human approval.

### Required checks to configure in GitHub

- `CI / lint`
- `CI / test (3.10)`
- `CI / test (3.12)`
- `CI / typecheck`
- `CI / cli_docs_and_help_contracts`
- `PR Review Orchestrator / synthesize`
- `Merge Readiness / evaluate`

### Acceptance criteria

- No agent can merge directly
- Auto-merge only happens through GitHub policy
- Merge queue can batch safe PRs without bypassing checks

---

## Phase 5: Nightly and Asynchronous Automation

**Goal:** Keep long-running automation out of the PR critical path.

### Files to modify

- `.github/workflows/aura-agentic-loop.yml`

### Files to create

- `.github/workflows/nightly-repo-health.yml`
- `core/github_automation/repo_health.py`
- `tests/test_repo_health.py`

### Concrete work

1. Keep `.github/workflows/aura-agentic-loop.yml` for:
   - backlog grooming
   - nightly quality scans
   - issue creation from recurring failures
   - documentation or debt follow-ups
2. Do not use the nightly loop as a required PR gate.
3. Add repo health summaries:
   - stale PRs
   - flaky tests
   - recurring failure clusters
   - hotspots by path
4. If nightly automation proposes code changes, require PR creation rather than direct push to `main`.

### Acceptance criteria

- Nightly automation produces useful backlog and health outputs
- PR latency is not coupled to long-running loops

---

## 5. Consolidation Plan For Existing Workflows

### Keep and modify

- `.github/workflows/ci.yml`
- `.github/workflows/copilot-workspace.yml`
- `.github/workflows/aura-agentic-loop.yml`
- `.github/agents/agentic-workflows-dev.agent.md`

### Convert or retire after replacement is stable

- `.github/workflows/copilot-autofix.yml`
- `.github/workflows/gemini-code-assist.yml`
- `.github/workflows/coder-automation.yml`

### Why

The current PR workflows are useful prototypes, but they post provider-specific comments directly. The new architecture should route provider outputs into one repo-owned orchestrator and one final verdict.

---

## 6. Secrets, Variables, and Auth

Use GitHub App credentials where possible. Avoid long-lived PATs for production automation.

### Minimum secret set for phased rollout

- `AURA_API_KEY`
- `OPENAI_API_KEY` or repo-approved Codex credential
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GITHUB_TOKEN` from Actions for repo-local operations

### Optional secrets for self-hosted bridges

- `GITHUB_PAT`
- `COPILOT_MCP_TOKEN`
- `MCP_API_TOKEN`
- `AGENT_API_TOKEN`

### GitHub repository variables

- `AURA_SKIP_CHDIR=1`
- `AURA_DRY_RUN=1` for early rollout lanes
- `AURA_AUTO_APPROVE=0`
- `AURA_GITHUB_AUTOMATION_ENABLED=1`

---

## 7. Test Plan

Add tests at three levels.

### Unit

- provider routing
- severity normalization
- merge policy evaluation
- issue form parsing
- slash command parsing

### Integration

- PR event -> synthesized review output
- issue opened -> structured implementation plan
- risky diff -> human gate block
- merge queue check compatibility

### Snapshot / contract

- final PR summary comment format
- JSON artifact schema
- label assignment behavior

---

## 8. Rollout Order

### Wave 1

- Governance baseline
- `merge_group` CI support
- issue forms
- PR template
- CODEOWNERS

### Wave 2

- Unified PR review orchestrator
- synthesized check run
- human gate wiring

### Wave 3

- structured issue planning
- slash commands
- AURA queue integration

### Wave 4

- coding-agent dispatch lanes
- provider-specific role profiles
- artifact and audit trail hardening

### Wave 5

- merge queue enablement
- nightly repo health and backlog automation

---

## 9. Risks and Mitigations

### Risk: duplicate bot noise

Mitigation:

- one final orchestrator comment
- provider-specific workflows emit artifacts, not public comments

### Risk: unsafe autonomous merges

Mitigation:

- no direct bot merge
- merge queue only
- human gate on risky paths

### Risk: provider drift and inconsistent advice

Mitigation:

- normalized schema
- synthesized final verdict
- provider-specific confidence tracking

### Risk: flaky merge queue behavior

Mitigation:

- add `merge_group` support before enabling queue
- test queue on non-critical branch ruleset first

### Risk: secrets sprawl

Mitigation:

- prefer GitHub App auth
- keep provider credentials limited to required workflows
- document secret ownership and rotation

---

## 10. First Implementation Slice

If execution starts immediately, ship this slice first:

1. Modify `.github/workflows/ci.yml` to support `merge_group`.
2. Add `.github/CODEOWNERS`.
3. Add issue forms and PR template.
4. Add `.github/workflows/pr-review-orchestrator.yml`.
5. Add `core/github_automation/` with:
   - `models.py`
   - `provider_router.py`
   - `review_synthesizer.py`
   - `policy.py`
6. Convert current Copilot and Gemini PR workflows into artifact producers or disable them behind a feature flag.
7. Add tests for normalization and policy gating.

This first slice gives the repo safe governance, one unified PR review path, and a clear foundation for Codex, Claude, Gemini, and Copilot expansion.

---

## 11. Definition of Done

This project is complete when all of the following are true:

- PRs receive one synthesized automation verdict instead of multiple bot summaries
- risky changes require human review automatically
- issue intake produces structured plans
- coding agents operate only through branches and PRs
- merge queue handles safe merges after required checks pass
- nightly automation is useful but not on the critical path

