# Agent Routing Guide

This guide defines which agent should own which kind of task in this repository and how to split work without collisions.

Research date: 2026-04-04

## Scope

This is a pragmatic routing guide, not a claim that one agent is universally better than another. Where a recommendation goes beyond an official product claim, it is marked as an inference.

## Executive Summary

- `Codex` should usually be the coordinator and primary owner for complex repository changes, deep refactors, code review, and tasks that require integrating planning, implementation, and verification.
- `Copilot coding agent` should own GitHub-native background work such as issue-to-PR execution, low-to-medium complexity backlog items, test extension, documentation updates, and GitHub review iteration.
- `Copilot agent mode` should own interactive IDE-local edits when a human wants to supervise file-by-file changes and approve terminal actions.
- `Gemini CLI` should own large-context analysis, live web-grounded research, multimodal understanding, terminal automation, and workflow tasks that benefit from search plus shell plus file operations.
- `Kimi Code` should own fast coding assistance, project exploration, IDE or CLI help, and sidecar implementation or analysis tasks when you want another local coding agent in parallel.
- `Kimi Claw` should own persistent, scheduled, or memory-heavy recurring tasks rather than repository-critical refactors.

## What The Official Sources Support

### Codex

OpenAI describes Codex as a coding agent for end-to-end engineering work including features, complex refactors, migrations, code understanding, documentation, testing, code review, and multi-agent workflows with parallel worktrees and cloud environments. OpenAI also says Codex can run commands, test harnesses, linters, and type checkers, and uses AGENTS.md files to adapt to repository-specific instructions.

### GitHub Copilot

GitHub documents two distinct modes:

- `Copilot coding agent`: asynchronous, GitHub-native, background execution in a GitHub Actions-powered environment that creates or updates pull requests.
- `Copilot agent mode`: IDE-local autonomous editing with human review of edits and approval of terminal commands.

GitHub explicitly says the coding agent can fix bugs, implement incremental features, improve test coverage, update documentation, address technical debt, resolve merge conflicts, and that it excels at low-to-medium complexity tasks in well-tested codebases.

### Gemini CLI

Google's official Gemini CLI repository describes it as an open-source terminal AI agent with a 1M token context window, built-in tools for Google Search grounding, file operations, shell commands, and web fetching, plus MCP support. Google also highlights code understanding and generation, debugging, automation, GitHub workflow integration, multimodal generation from PDFs, images, or sketches, and custom context files via `GEMINI.md`.

### Kimi Code and Kimi Claw

Moonshot describes Kimi Code CLI as a terminal AI agent that can read and edit code, execute shell commands, search and fetch web pages, and autonomously plan and adjust actions. Kimi also documents VS Code integration, ACP-based IDE integration, and support for project understanding, coding, and automation tasks.

Moonshot separately describes Kimi Claw as an always-on cloud agent with persistent memory, scheduled tasks, skill chaining, cloud storage, and coding-capable workflows.

## Best-Fit Roles By Agent

### Codex

Best fit:

- Coordinator for multi-agent software work
- Complex cross-file implementation
- Refactors that must preserve behavior
- Tasks needing strong code review and verification discipline
- Work that needs explicit task decomposition and bounded subtask ownership

Why:

- Official positioning emphasizes end-to-end software engineering, testing, code review, multi-agent workflows, and adaptation through AGENTS.md.

Routing note:

- Inference: use Codex whenever task quality matters more than raw speed and when the work crosses planning, coding, and validation boundaries.

### Copilot Coding Agent

Best fit:

- GitHub issue to pull request workflows
- Straightforward backlog items
- Test coverage extensions
- Documentation or technical debt cleanup
- Merge conflict resolution
- Background GitHub-native iteration on a PR

Why:

- GitHub explicitly positions the coding agent for background PR work and says it excels at low-to-medium complexity tasks in well-tested codebases.

Routing note:

- Do not make this the coordinator unless the entire workflow is GitHub-native and PR-centric.

### Copilot Agent Mode

Best fit:

- Interactive IDE-local changes
- Edits where the human wants to supervise file selection and terminal commands
- Complex but narrow tasks where quick iteration in the editor matters more than asynchronous PR automation

Why:

- GitHub explicitly says agent mode is best when a task is complex, multi-step, and may need tool integration, while staying local to the IDE.

Routing note:

- Use when a developer is actively in the editor and wants tight feedback loops.

### Gemini CLI

Best fit:

- Large-context repository analysis
- Tasks requiring current web research before coding
- Multimodal inputs like screenshots, PDFs, and sketches
- Terminal-heavy automation
- GitHub workflow glue in teams already using Gemini Actions

Why:

- Google emphasizes large context, Google Search grounding, web fetching, shell commands, multimodal inputs, and automation.

Routing note:

- Inference: use Gemini first when the task starts as research, investigation, environment analysis, or visual-to-code interpretation; hand implementation to Codex if the patch is high risk.

### Kimi Code

Best fit:

- Sidecar coding assistance
- Fast exploratory edits
- Parallel local analysis or implementation on isolated write scopes
- IDE-assisted work via VS Code or ACP-compatible editors

Why:

- Moonshot documents code editing, shell execution, web access, project understanding, and IDE integration.

Routing note:

- Inference: use Kimi as a worker, not the coordinator, unless your team already relies on its CLI or IDE workflow.

### Kimi Claw

Best fit:

- Persistent memory tasks
- Scheduled and recurring automations
- Long-running research or reporting
- Skill-chained workflows outside the core repo change path

Why:

- Moonshot explicitly highlights persistent memory, scheduled tasks, cloud storage, and large skill catalogs.

Routing note:

- Keep Kimi Claw away from concurrent edits to critical repository files unless a human is supervising closely.

## Routing Matrix

## 1. Repository Understanding

- Unknown subsystem walkthrough: `Gemini` first, `Codex` second
  Why: Gemini is strongest on large-context plus live research; Codex is better for converting findings into implementation plans.
- Architecture map from source only: `Codex`
  Why: better fit when the answer must immediately feed into code changes in this repo.

## 2. Coding And Refactoring

- Complex multi-file refactor with behavior preservation: `Codex`
- Cross-module feature with tests and docs: `Codex`
- Repetitive local editor refactor with human supervision: `Copilot agent mode`
- Isolated auxiliary implementation in a clearly bounded module: `Kimi` or `Codex`

## 3. GitHub Workflow

- Issue to draft PR in background: `Copilot coding agent`
- Respond to PR comments on changed lines: `Copilot coding agent`
- Merge-conflict cleanup on an existing PR: `Copilot coding agent`
- PR review or bug sweep where correctness matters most: `Codex`, then human or GitHub review tooling

## 4. Research Before Code

- Needs current docs, standards, APIs, or vendor behavior: `Gemini`
- Needs comparison of implementation options after research: `Codex`
- Needs recurring market or ecosystem monitoring: `Kimi Claw`

## 5. Visual Or Multimodal Inputs

- Screenshot or PDF to code draft: `Gemini`
- High-fidelity visual-to-code experiment: `Kimi`
- Production integration of the result into the repo: `Codex`

## 6. Docs And Reports

- Repo-coupled technical docs that depend on code truth: `Codex`
- Long-form generated reports, recurring summaries, or document-heavy workflows: `Kimi` or `Kimi Claw`
- GitHub-facing issue and PR automation: `Copilot`

## 7. Parallelization

- Coordinator: `Codex`
- Independent read-only investigations: `Gemini` and `Kimi`
- Independent implementation slices with disjoint write scopes: `Codex` and `Kimi`
- GitHub-side background PR task in parallel with local development: `Copilot coding agent`

## Task Breakdown Rules

- One owner per task.
- One write scope per owner.
- Shared files must be owned sequentially, not concurrently.
- If a task needs current internet facts, split it into:
  - research owner
  - implementation owner
- If a task will end as a PR, split it into:
  - implementation owner
  - PR owner
- If a task is vague, send it to the coordinator first, not a worker.

## Recommended Breakdown Patterns

### Pattern A: Research -> Build -> Verify

- `Gemini`: research APIs, upstream docs, or screenshots
- `Codex`: implement code changes
- `Copilot coding agent`: open or iterate on the GitHub PR if desired

### Pattern B: GitHub Backlog Sweep

- `Copilot coding agent`: take low-risk issues one by one
- `Codex`: review anything with architectural or regression risk

### Pattern C: Local Swarm

- `Codex`: coordinator and integration owner
- `Kimi`: isolated worker on a bounded module
- `Gemini`: side investigation or live-doc lookup

### Pattern D: Persistent Operations

- `Kimi Claw`: scheduled reports, recurring searches, or long-running automations
- `Codex`: only when the task graduates into repository changes

## Task Assignment Template

Use this exact structure in `COLLAB_CONTEXT.md`:

- Title
- Owner
- Status
- Depends on
- Write scope
- Read-only context
- Deliverable
- Verification
- Handoff target

## Recommended Default For This Repo

- Coordinator: `Codex`
- Research worker: `Gemini`
- GitHub PR worker: `Copilot coding agent`
- Optional sidecar worker: `Kimi`

Reason:

- This repository already has strong local instructions, AGENTS.md conventions, and local collaboration files. Codex is the best fit to coordinate repo-local engineering work from the terminal, while Gemini and Copilot are best used as specialist workers.

## Sources

- OpenAI Codex product page: https://openai.com/codex/
- OpenAI Introducing Codex: https://openai.com/index/introducing-codex/
- GitHub Copilot coding agent docs: https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-coding-agent
- GitHub Copilot IDE agent mode docs: https://docs.github.com/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide?tool=vscode
- GitHub Copilot coding agent press release: https://github.com/newsroom/press-releases/coding-agent-for-github-copilot
- Gemini CLI repository: https://github.com/google-gemini/gemini-cli
- Kimi Code CLI docs: https://www.kimi.com/code/docs/en/kimi-cli/guides/getting-started.html
- Kimi Code VS Code docs: https://www.kimi.com/code/docs/en/kimi-code-for-vscode/guides/getting-started.html
- Kimi Claw overview: https://www.kimi.com/resources/kimi-claw-introduction
