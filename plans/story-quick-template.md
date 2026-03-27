# Plan: Lightweight Story Template for Quick Ideation

**Forge Story:** AF-STORY-0009  
**Status:** Implemented (Sprint S002)

## Problem

The full Forge story template (`templates/story.md`) is comprehensive but heavyweight — it requires SCAMPER, AutoTRIZ, SixHats, and full design-pass notes before a story can progress. This friction discourages capturing rough ideas before they are lost.

The backlog needed a fast-path for inbox-stage capture: minimal required fields, maximum velocity.

## Approach

Two artifacts:

### 1. `templates/story_quick.yaml` — 8-field template

Captures the essential signal without forcing early design commitment:

```yaml
id: "AF-STORY-XXXX"
title: ""
type: "feature"
lane: "inbox"
trigger: ""       # one sentence: what prompted this
hypothesis: ""    # one sentence: expected gain
contract_impact: "none"
safety_impact: "none"
```

Quick stories enter `inbox/` and are promoted to the full template when they reach the `refined` stage.

### 2. `scripts/new_story.py` — scaffolding CLI

```bash
# Create a quick story
python3 scripts/new_story.py --quick --title "Add retry logic to MCP calls"

# Create a full-template story
python3 scripts/new_story.py --title "Redesign goal archive schema" --type improvement
```

Auto-increments the story ID by scanning all existing `AF-STORY-XXXX` IDs in the backlog.

## Files

| File | Change |
|------|--------|
| `.aura_forge/templates/story_quick.yaml` | **Created** — 8-field quick template |
| `scripts/new_story.py` | **Created** — story scaffolder CLI |
| `.aura_forge/backlog/ready/AF-STORY-0009.yaml` | Promoted from refined; `plans_link` set to this file |

## Usage

```bash
# Quick ideation capture (8 fields)
python3 scripts/new_story.py --quick --title "My rough idea"
# → .aura_forge/backlog/inbox/AF-STORY-XXXX.yaml

# Full story scaffold
python3 scripts/new_story.py --title "Detailed feature" --type improvement

# List available types
python3 scripts/new_story.py --help
```

## Acceptance Criteria

- [x] `templates/story_quick.yaml` exists with ≤ 10 required fields
- [x] `scripts/new_story.py --quick` creates a valid inbox YAML entry
- [x] Story ID is auto-incremented (no manual ID assignment required)
- [x] Full template scaffold also supported (default mode)
- [x] Script exits `1` if template file is missing

## Notes

- Quick stories intentionally skip `design_pass_notes` — those are added during refinement
- The `lane: "inbox"` field is hardcoded in quick mode; stories graduate via normal Forge promotion
- ID collision is avoided by scanning all `backlog/**/*.yaml` files globally
