# Development Automation Guide

This guide is the repo-local operating model for research-first development in AURA. It ties together subagents, MCP servers, local skills, Claude plugins, and hook recommendations so the codebase can analyze itself, propose bounded improvements, and execute follow-up work safely.

## Primary Goal

Use automation to improve research and analysis quality first, then turn the best findings into bounded implementation goals.

The default flow is:

1. analyze the repo
2. synthesize innovation proposals
3. queue bounded follow-up work
4. implement through the normal orchestrator path

## Approved Research Subagents

Use these repo-local agent specs under [`.github/agents/architecture-explorer.agent.md`](/home/westonaaron675/aura-cli/.github/agents/architecture-explorer.agent.md), [`.github/agents/capability-researcher.agent.md`](/home/westonaaron675/aura-cli/.github/agents/capability-researcher.agent.md), [`.github/agents/innovation-synthesizer.agent.md`](/home/westonaaron675/aura-cli/.github/agents/innovation-synthesizer.agent.md), and [`.github/agents/verification-reviewer.agent.md`](/home/westonaaron675/aura-cli/.github/agents/verification-reviewer.agent.md).

- `architecture-explorer`: structural hotspots, risky boundaries, dependency bottlenecks
- `capability-researcher`: skills, MCP, plugin, and tooling gaps
- `innovation-synthesizer`: ranked bounded proposals suitable for queueing
- `verification-reviewer`: smallest proof path and residual risk

Main-agent ownership remains unchanged:

- final prioritization
- final integration
- final verification interpretation
- final user-facing closeout

## MCP Usage Matrix

Use the existing repo MCP set in [`.mcp.json`](/home/westonaaron675/aura-cli/.mcp.json) intentionally by phase.

| Phase | Preferred MCP / tool | Purpose |
| --- | --- | --- |
| Architecture analysis | `tree-sitter`, `filesystem`, `git`, `semgrep` | Structural investigation, cross-file inspection, static analysis |
| Capability research | `context7`, `fetch`, `brave-search`, `memory` | Docs lookup, external research, retained findings |
| Innovation synthesis | `memory`, `github` | Compare findings with recent repo context and issue/PR surfaces |
| Verification review | `semgrep`, `filesystem`, `git`, `playwright` | Static safety checks, narrow inspection, UI proof when needed |

Current repo-standard MCPs to prioritize before adding more:

- `context7`
- `fetch`
- `github`
- `memory`
- `tree-sitter`
- `semgrep`
- `playwright`

Curated expansion only:

- add one deeper repo intelligence surface if cross-file reasoning remains weak
- add one issue/project tracking surface if innovation findings need stronger execution tracking than the goal queue alone

## Skill Strategy

Prefer repo-native Python skills for reusable analysis logic.

Current skills that should be composed into research loops:

- `structural_analyzer`
- `tech_debt_quantifier`
- `code_clone_detector`
- `observability_checker`
- `database_query_analyzer`
- `skill_failure_analyzer`
- `evolution_skill`

High-value next skill additions:

- `innovation_proposal_ranker`
- `capability_gap_reporter`
- `research_closeout`

Skill design rules:

- structured dict output only
- one clear purpose per skill
- bounded file/test surface
- useful both in autonomous loops and operator-guided work

## Plugin Guidance

The current home-installed Claude plugin set already covers most supporting needs. Prefer deeper use of these before adding more:

- `superpowers`
- `context7`
- `github`
- `playwright`
- `hookify`
- `semgrep`
- `pr-review-toolkit`
- `mcp-server-dev`

Use plugins for leverage around the repo, not as the main runtime logic of AURA.

Good plugin roles here:

- documentation lookup
- browser proof flows
- review and PR hygiene
- hook scaffolding
- MCP development and testing

## Hook Recommendations

Recommended project-local hooks to add through the existing Claude tooling rather than custom one-off shell scripts:

- pre-edit guard for `.env`, token files, and lockfiles
- post-edit lint/type/test hook for touched Python files
- confirmation gate for broad repo-wide edits or destructive file operations
- optional targeted test hook for files under `core/`, `agents/`, and `aura_cli/`

Hook priorities:

1. protect secrets and critical config
2. catch obvious regressions early
3. avoid heavy hooks that slow normal development too much

## Evolve Workflow

The `evolve` command is the current control surface for the innovation workflow. It should be used in two modes:

- queue-only research mode for background or operator-guided analysis
- queue-and-implement mode for bounded immediate follow-up work

Recommended usage:

- `python3 main.py evolve --queue-only --proposal-limit 3 --focus research`
- `python3 main.py evolve --execute-queued --proposal-limit 2 --focus capability`

Focus modes:

- `capability`
- `quality`
- `throughput`
- `research`

## 90-Day Rollout

### Days 1-30

- standardize subagent roles
- expose innovation artifacts clearly
- stabilize queue-only research sessions

### Days 31-60

- deepen MCP usage by role
- add the next reusable research skills
- tighten verification heuristics for proposed work

### Days 61-90

- add curated hooks
- review whether a repo-local plugin bundle is justified
- measure proposal quality, queue completion, and operator time saved
