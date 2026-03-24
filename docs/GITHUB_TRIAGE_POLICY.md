# GitHub Triage and Closure Policy

This repository uses an automated weekly triage workflow to keep issues and pull requests healthy and actionable.

For active branch stabilization and repo-wide CI/PR sweeps, use `docs/AURA_MULTI_AGENT_WORKFLOW.md`, `docs/AURA_SWEEP_TEMPLATES.md`, and `docs/ACTIVE_SWEEP_STATUS.md` alongside this policy. This file defines the repo's GitHub-side closure and stale-handling rules; the AURA workflow docs define how active engineering triage is executed.

## Active sweep policy

When a branch is under active CI/PR triage:

1. Create or update `docs/ACTIVE_SWEEP_STATUS.md` with:
   - branch and HEAD SHA
   - target PR numbers
   - active error buckets
   - unresolved review blockers
   - next highest-priority bucket
2. Treat GitHub failures in this order:
   - workflow/setup failures
   - required CI lanes
   - PR review blockers
   - provider/external blockers
   - developer-surface drift affecting future CI/PR work
3. Do not classify an active PR as stale while:
   - the branch has fresh CI repair commits,
   - required checks are still being actively driven,
   - the active sweep status file shows the PR as in progress.
4. Closeout for an active sweep must state:
   - which check or review comment was resolved,
   - which blockers remain,
   - whether the remaining blockers are repo-fixable or external.

## Labels managed by policy

- `stale-pr`: Pull request has exceeded stale inactivity threshold.
- `blocked-needs-author`: Maintainer is waiting on author action.
- `superseded`: Replaced by a newer issue or pull request.
- `cannot-reproduce`: Maintainers could not reproduce the report.
- `needs-design`: Requires product/design decision.
- `close-candidate`: Explicit maintainer signal that PR should be evaluated for closure.
- `stale-warning`: Warning posted before stale issue closure.

## Time thresholds

- **Warning threshold:** 30 days of no activity.
- **Closure threshold:** 60 days of no activity (after warning grace period).

## Pull request closure flow (`close-candidate`)

For open PRs labeled `close-candidate`, automation performs this gate:

1. Verify there is no active dependency:
   - no open linked closing issue references,
   - no open milestone.
2. If dependency-free, post a final maintainer comment with exact unblock conditions:
   - rebase cleanly on `main`,
   - resolve CI and review feedback,
   - resolve blocker labels with maintainer confirmation.
3. Apply closure label and close:
   - use `superseded` when that label exists,
   - otherwise use policy fallback labels and close as `not_planned`.

## Pull request readiness expectations

Before a PR is considered reviewer-complete in an active sweep:

- required GitHub checks should be green or explicitly marked external
- active review comments should be either:
  - resolved by code,
  - resolved by explanation with evidence,
  - explicitly blocked by a larger decision
- the latest closeout should map the resolved work back to:
  - the PR number
  - the exact check or comment addressed
  - verification performed
- the reviewer-facing handoff should use `docs/PR_REVIEWER_SUMMARY_TEMPLATE.md`

## Issue triage flow

- **De-duplication:** close issues labeled `duplicate` or `superseded`.
- **Solved-by-PR:** close issue automatically when timeline shows a merged linked PR.
- **Stale process:**
  1. after 30 inactive days, post warning + add `stale-warning`;
  2. after 60 inactive days with warning still present, close as `not_planned`.

## Weekly report

Every weekly run emits a triage report with:

- count by status (open issues, open PRs),
- newly closed items in the last 7 days,
- reopened items in the last 7 days,
- blocked items that need maintainer decision.
