# AURA Forge

AURA Forge is a repo-native development cognition layer for aura-cli. It sits upstream of implementation and turns vague goals into structured, reviewable work items, design passes, sprint artifacts, and capability deltas.

## Purpose

Forge improves decision quality before code changes, queue insertion, or orchestration. It is a pre-execution intelligence system—not a runtime mutation engine.

## Where Artifacts Live

```
.aura_forge/
├── schemas/           # Machine-readable artifact definitions
├── templates/         # Working forms for contributors and agents
├── backlog/           # File-based pipeline for idea maturation
│   ├── inbox/         # Raw ideas
│   ├── refined/       # Analyzed ideas
│   ├── ready/         # Implementation-ready items
│   ├── in_progress/   # Active work
│   ├── review/        # Awaiting validation
│   └── done/          # Completed items
├── sprints/           # Sprint packets
├── retros/            # Sprint retrospectives
├── examples/          # Worked examples
└── indexes/           # Discoverability without database
```

## What Stays Manual vs. Assisted

| Activity | Mode |
|----------|------|
| Creating inbox items | Manual or assisted |
| Refinement with design passes | Assisted |
| Promotion between stages | Manual review |
| Sprint planning | Manual |
| Retro generation | Assisted |
| Capability delta creation | Assisted |
| Memory promotion | Manual after validation |

## Quick Start

1. Put raw ideas in `.aura_forge/backlog/inbox/`
2. Refine them with reverse engineering and reverse decomposition
3. Run SCAMPER, AutoTRIZ, and Six Hats before promoting to ready
4. Link ready items to `plans/` when implementation begins
5. After completion, create a retro and optional capability delta
6. Promote repeated lessons into `memory/` only after validation

## Promotion Rules

### inbox → refined
- Problem statement
- Likely affected repo areas
- At least one reverse engineering note (including "Gap Identified" field)
- At least one reverse decomposition note
- **Note on batch refinement:** All design passes (reverse_engineering, reverse_decomposition,
  SCAMPER, AutoTRIZ, Six Hats) are typically completed in a single session before the story
  moves to `refined/`. The promotion rules list the *minimum* required — in practice, running
  all applicable passes in one batch and promoting directly to `refined/` (or even `ready/`)
  reduces intermediate state and review overhead. Iterative pass-by-pass promotion is valid
  but uncommon.

### refined → ready
- SCAMPER pass
- AutoTRIZ contradiction analysis
- Six Hats review
- Constraints and acceptance criteria
- Explicit contract impact declaration
- Explicit safety impact declaration (if relevant)

### ready → in_progress
- Linked implementation plan OR explicit note that the story is documentation-only

### review → done
- Outputs created
- Docs updated where needed
- Tests or validation notes included
- Retro seed recorded if needed

## Connection to Existing Repo Areas

- **plans/**: Forge stories point to implementation plans rather than duplicate them
- **memory/**: Recurring lessons promoted after repetition or validation
- **task_queue/**: Forge generates queue candidates; does not auto-enqueue in phase 1
- **docs/**: Cross-cutting contributor documentation only; Forge artifacts stay in `.aura_forge/`

## CLI Policy

No Forge CLI commands in phase 1. The repo's CLI has generated reference docs and snapshot checks—command additions impose nontrivial maintenance cost. Start file-first.

## What Not to Build Yet

- Autonomous sprint planning
- Hidden prioritization logic
- Automatic queue insertion
- Automatic memory promotion
- Forge-specific CLI commands
- Runtime parsers that make the artifact system opaque

These would add complexity before the artifact model has stabilized.

## Contributor Guidelines

- Forge artifacts live in `.aura_forge/`
- Before implementing a nontrivial story, create or update a Forge work item
- If a story affects CLI behavior, declare contract impact and follow the repo's CLI maintenance steps
- If a story affects autonomous apply or overwrite behavior, declare safety impact
- After finishing a story, record a retro and any capability delta
