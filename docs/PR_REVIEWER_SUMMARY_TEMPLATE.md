# PR Reviewer Summary Template

Use this template at the end of an active CI/PR sweep or any review-driven branch update. It is a reviewer-facing companion to `docs/ACTIVE_SWEEP_STATUS.md` and should map cleanly onto `.github/PULL_REQUEST_TEMPLATE.md`.

```text
PR Reviewer Summary

PR:
Branch:
HEAD SHA:

Description:
- concise summary of what changed

Motivation:
- blocker, issue, or review thread being addressed

Changes Made:
- key implementation or workflow changes
- key test or assertion changes
- key doc/prompt/workflow changes

Checks/Comments Addressed:
- exact CI lane fixed:
- exact review comment resolved:

Testing Performed:
- targeted local verification:
- broader CI/workflow verification:
- anything intentionally not verified:

Reviewer Notes:
- remaining risks:
- external blockers, if any:
- follow-up still needed:
```

## Usage Rules

- Keep this summary short and operational.
- Prefer exact CI lane names and direct review-comment mapping over generic prose.
- If the branch is in an active sweep, update `docs/ACTIVE_SWEEP_STATUS.md` first, then derive the reviewer summary from the latest closeout.
- If the PR is reviewer-complete, say so explicitly.
- If the PR is only code-complete, state what still blocks reviewer completion.
