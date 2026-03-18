# GitHub Triage and Closure Policy

This repository uses an automated weekly triage workflow to keep issues and pull requests healthy and actionable.

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
