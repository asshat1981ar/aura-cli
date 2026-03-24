# AURA Sweep Templates

Use these templates when running a repo-wide CI, PR, or developer-surface sweep. They are designed to pair with `docs/AURA_MULTI_AGENT_WORKFLOW.md` and keep audit, delegation, and closeout output consistent across main-agent and subagent work.

## 1. Audit Summary Template

Use this before any mutating work in a broad sweep.

```text
Audit Summary

Branch:
HEAD SHA:
Target PR(s):

Active error buckets:
- workflow/compiler/setup:
- required CI lanes:
- PR review blockers:
- provider/external blockers:
- developer-surface drift:

Unrelated local worktree changes:
- out of scope:
- adjacent but untouched:
- high-risk overlap:

Execution surfaces in scope:
- workflows:
- runtime/test paths:
- docs/prompts/agent guidance:

Recommended first bucket:
Verification target:
Notes:
```

## 2. Delegation Packet Template

Use this for explorers, workers, and verification agents.

```text
Delegation Packet

Objective:
Branch:
HEAD SHA:
Role:

Scope boundary:
- files:
- subsystem:
- explicit write scope:

Expected output:
1. root cause or answer
2. smallest fix surface
3. verification target
4. residual risk

Constraints:
- do not revert unrelated edits
- do not expand scope beyond the listed files/subsystem
- report blockers instead of guessing
```

## 3. Sweep Queue Template

Use one line per actionable bucket. Keep the main queue grouped by failure class instead of by file.

```text
Sweep Queue

Bucket | Owner | Status | Branch/SHA | Verification target | Notes
workflow/setup |
required CI lane |
PR review blocker |
developer-surface drift |
external blocker |
```

## 4. Closeout Template

Use this at the end of every sweep cycle.

```text
Closeout

Branch:
HEAD SHA:
Bucket addressed:
Files/surfaces changed:
Verification performed:
Status: resolved | partially resolved | externally blocked
Next highest-priority bucket:

PR-facing note:
- comment or check addressed:
- follow-up still needed:
```

## 5. Developer-Surface Drift Checklist

Use this when the refinement phase extends beyond immediate CI/PR repair.

- `docs/AURA_OPERATOR_PROMPT.md` matches the active execution model
- `docs/AURA_ITERATIVE_WORKFLOW.md` matches the current local-first loop
- `docs/AURA_MULTI_AGENT_WORKFLOW.md` matches current CI/PR triage behavior
- `.github/agents/*.agent.md` aligns with approved main-agent/subagent roles
- `README.md` exposes the current operator and workflow entry points
- workflow guidance distinguishes repo-fixable failures from external blockers
