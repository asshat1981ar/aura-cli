# Plan: Link Story Artifacts to Plans/

**Forge Story:** AF-STORY-0006  
**Status:** Implemented (Sprint S002)

## Problem

Ready-stage Forge stories have a `plans_link` field that references implementation plan documents in `plans/`. Previously, no tooling validated these links — broken references went undetected, making the Forge→Plan traceability relationship unreliable.

## Approach

Create `scripts/link_story_plans.py` — a standalone validator that:
1. Scans all `.aura_forge/backlog/ready/*.yaml` files
2. Extracts `plans_link` fields
3. Verifies the referenced `plans/<file>` exists in the repo
4. Reports broken links with story filename and expected path
5. Exits `1` on any broken link, `0` if all links valid (or no links)

Integrate into the existing `scripts/lint_forge_index.py` linting pipeline as a companion check.

## Files

| File | Change |
|------|--------|
| `scripts/link_story_plans.py` | **Created** — validator script |
| `.aura_forge/backlog/ready/AF-STORY-0006.yaml` | Promoted from refined; `plans_link` set to this file |

## Usage

```bash
# Check all ready/ stories
python3 scripts/link_story_plans.py

# Verbose: show OK links too
python3 scripts/link_story_plans.py --verbose

# Expected output (clean):
# Checking plans_link in .aura_forge/backlog/ready ...
# All plans_link references valid.
```

## Acceptance Criteria

- [x] `scripts/link_story_plans.py` exists and is executable
- [x] Reports broken `plans_link` targets with story name and missing path
- [x] Exits `1` on broken links, `0` on clean
- [x] Handles YAML parse errors gracefully (skip + warn, continue)
- [x] `AF-STORY-0011.yaml` (existing plan) reports OK

## Notes

- Plan files for stories-in-flight are legitimately absent until implementation begins — this is expected and the `link_story_plans.py` exit-1 is intentional (signals work to do, not a bug)
- The validator is a pre-merge gate, not a CI blocker for WIP branches
