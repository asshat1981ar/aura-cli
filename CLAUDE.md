# CLAUDE.md — AI Assistant Guide for AURA CLI

This file documents the codebase structure, conventions, and workflows for AI assistants working in this repository.

> 🧠 **Memory System Active**: Hot cache (this file) + Deep storage (`memory/` directory)
> - Decode shorthand: `memory/glossary.md`
> - People/aliases: `memory/people/`
> - Projects: `memory/projects/`
> - Tools/context: `memory/context/`

---

## What Is AURA CLI?

AURA CLI is an autonomous software development platform. It runs a multi-agent loop that takes natural-language goals, plans code changes, applies them atomically, sandboxes execution, verifies with tests/linters, and reflects on outcomes — all with persistent memory and adaptive skill weights.

---

## Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Primary CLI shim — loads env, parses args, delegates to `aura_cli.cli_main:main()` |
| `run_aura.sh` | Bash convenience wrapper with shorthand aliases (`run`, `once`, `add`, `status`, `interactive`) |

**Quick usage:**
```bash
python3 main.py --help
./run_aura.sh --help
./run_aura.sh run --dry-run
./run_aura.sh once "Refactor goal queue" --max-cycles 1
python3 main.py goal add "Add unit tests" --run
```

Set `AURA_SKIP_CHDIR=1` to keep the current working directory (recommended for local dev and tests).

---

## Repository Structure

```
aura-cli/
├── main.py                    # CLI entry shim
├── run_aura.sh                # Shell wrapper
├── aura.config.json           # Runtime config (models, MCP ports, memory)
├── settings.json              # Model routing and autonomous-loop settings
├── requirements.txt           # Python runtime dependencies
├── package.json               # Node deps (@beads/bd, copilot, codex-cli)
├── pytest.ini                 # pytest config (pythonpath=., testpaths=.)
├── Dockerfile                 # Container: python:3.12-slim, exposes 8000
│
├── aura_cli/                  # CLI layer
│   ├── cli_main.py            # Command dispatcher + runtime init
│   ├── cli_options.py         # argparse setup, legacy flags, help
│   ├── options.py             # Command/action specs and CLI contract defs
│   ├── commands.py            # Handlers: help, doctor, add, run, status, exit
│   ├── doctor.py              # System diagnostics
│   ├── server.py              # FastAPI remote-ops server
│   ├── mcp_github_bridge.py   # GitHub MCP integration
│   └── tui/                   # Terminal UI components
│
├── core/                      # Orchestration engine
│   ├── orchestrator.py        # Main 10-phase loop (ingest→reflect)
│   ├── model_adapter.py       # LLM interface (Gemini, OpenAI, OpenRouter, local)
│   ├── workflow_engine.py     # Orchestrated workflows with cycle limits
│   ├── hybrid_loop.py         # Deprecated legacy loop; prefer orchestrator.py
│   ├── file_tools.py          # Atomic file operations + overwrite safety policy
│   ├── task_handler.py        # Goal/task execution coordination
│   ├── capability_manager.py  # Skill bootstrapping and MCP provisioning
│   ├── config_manager.py      # Configuration validation and defaults
│   ├── skill_dispatcher.py    # Adaptive skill selection
│   ├── vector_store.py        # Semantic memory (embeddings)
│   ├── goal_queue.py          # Persistent JSON-backed goal queue
│   ├── goal_archive.py        # Completed goals archive
│   ├── beads_bridge.py        # Beads contract integration
│   ├── evolution_loop.py      # Self-improvement / RSI loop
│   ├── reflection_loop.py     # Outcome reflection and learning
│   ├── convergence_escape.py  # Convergence detection
│   ├── human_gate.py          # Human approval workflow
│   ├── health_monitor.py      # System health tracking
│   ├── git_tools.py           # Git operations
│   ├── logging_utils.py       # Structured JSON logging with secret masking
│   ├── runtime_auth.py        # API key management
│   ├── policy.py              # Stopping conditions
│   ├── sadd/                  # Sub-Agent Driven Development
│   │   ├── types.py           # Dataclasses and validation
│   │   ├── design_spec_parser.py  # Markdown → workstreams
│   │   ├── workstream_graph.py    # DAG + state machine
│   │   ├── sub_agent_runner.py    # Per-workstream orchestrator
│   │   ├── session_coordinator.py # Parallel execution
│   │   ├── session_store.py       # SQLite persistence
│   │   └── mcp_tool_bridge.py     # MCP tool matching
│   └── tests/                 # Core module unit tests
│
├── agents/                    # Specialized agents
│   ├── registry.py            # Agent factory + pipeline adapters
│   ├── planner.py             # Plan generation
│   ├── critic.py              # Adversarial plan review
│   ├── coder.py               # Code generation
│   ├── debugger.py            # Debug and error handling
│   ├── sandbox.py             # Subprocess code execution
│   ├── verifier.py            # Test/lint verification
│   ├── router.py              # Task routing
│   ├── applicator.py          # File change application
│   ├── ingest.py              # Project context ingestion
│   ├── synthesizer.py         # Plan synthesis
│   ├── reflector.py           # Outcome reflection
│   ├── tester.py              # Test execution
│   ├── mutator.py             # Code mutation
│   └── skills/                # 15+ static analysis skills
│       ├── base.py
│       ├── linter_enforcer.py
│       ├── type_checker.py
│       ├── structural_analyzer.py
│       ├── test_coverage_analyzer.py
│       ├── code_clone_detector.py
│       └── ...
│
├── memory/                    # State persistence
│   ├── brain.py               # SQLite semantic memory (NetworkX graph)
│   ├── store.py               # JSONL decision log with rotation
│   ├── controller.py          # Memory tier management
│   ├── embedding_provider.py  # Local embedding generation
│   ├── local_cache_adapter.py # Local cache interface
│   ├── momento_adapter.py     # Momento distributed cache
│   ├── cache_adapter_factory.py
│   ├── brain.db               # SQLite DB (git-ignored)
│   ├── goal_queue.json        # Persistent goal queue (git-ignored)
│   ├── goal_archive.json      # Completed goals (git-ignored)
│   └── skill_weights.json     # Adaptive skill weights (git-ignored)
│
├── tools/                     # External tool / MCP servers
│   ├── mcp_server.py          # FastAPI MCP server (port 8001+)
│   ├── agentic_loop_mcp.py    # Loop orchestration MCP
│   ├── aura_control_mcp.py    # Control plane MCP
│   ├── aura_mcp_skills_server.py
│   ├── github_copilot_mcp.py  # Copilot integration
│   ├── sadd_mcp_server.py     # SADD MCP server (port 8020)
│   └── sequential_thinking_mcp.py
│
├── tests/                     # Test suite (120+ files)
│   ├── snapshots/             # JSON snapshot contracts (40+ files)
│   ├── integration/           # Integration tests
│   ├── fakes/                 # Test fakes
│   └── fixtures/              # Test fixtures
│
├── scripts/                   # Utility and automation
│   ├── generate_cli_reference.py   # Auto-generate docs/CLI_REFERENCE.md
│   ├── benchmark_loop.py
│   ├── mcp_server_setup.sh
│   └── ...
│
├── docs/                      # Architecture documentation
│   ├── CLI_REFERENCE.md       # Generated — do not edit manually
│   ├── INTEGRATION_MAP.md     # Architecture overview
│   └── FINAL_DEVELOPMENT_GUIDE.md
│
└── .github/
    ├── workflows/
    │   ├── ci.yml             # Python 3.10/3.11 tests + CLI contracts
    │   ├── aura-agentic-loop.yml
    │   └── ...
    └── copilot-instructions.md
```

---

## Development Commands

```bash
# Run all tests
python3 -m pytest

# Run a specific test file
python3 -m pytest tests/test_task_handler.py -v

# Run tests matching a keyword
python3 -m pytest -k "snapshot" -q

# Show CLI help
python3 main.py --help

# Run the doctor (system diagnostics)
python3 main.py doctor

# Regenerate CLI docs (required after any CLI changes)
python3 scripts/generate_cli_reference.py

# Verify CLI docs are current (used in CI)
python3 scripts/generate_cli_reference.py --check
```

---

## Autonomous Loop Phases

The main loop in `core/orchestrator.py` runs these phases sequentially per goal:

1. **Ingest** — gather project context + memory hints (`agents/ingest.py`)
2. **Skill Dispatch** — run adaptive static-analysis skills (`core/skill_dispatcher.py`)
3. **Plan** — generate step-by-step plan with retries (`agents/planner.py`)
4. **Critique** — adversarial plan review (`agents/critic.py`)
5. **Synthesize** — merge plan + critique into task bundle (`agents/synthesizer.py`)
6. **Act** — generate code changes with retries (`agents/coder.py`)
7. **Sandbox** — execute snippet in subprocess (`agents/sandbox.py`)
8. **Apply** — write file changes atomically (`core/file_tools.py`)
9. **Verify** — run tests/linters (`agents/verifier.py`)
10. **Reflect** — summarize outcome, update skill weights (`agents/reflector.py`)

**Failure routing:**
- Failed verify → check mismatch overwrite policy
- Sandbox failure → retry Act phase (up to 3 attempts)
- Consistent failures → re-plan
- Environmental/external failures → skip

---

## CLI Command System

CLI commands are defined in `aura_cli/options.py` and dispatched through `COMMAND_DISPATCH_REGISTRY` in `aura_cli/cli_main.py`.

**Available actions:** `goal-add`, `goal-run`, `goal-once`, `goal-status`, `doctor`, `config`, `diag`, `watch`, `studio`, `logs`, `workflow-run`, `mcp-tools`, `mcp-call`, `memory-search`, `memory-embedding`, `metrics`, `contract-report`, `bootstrap`

**When modifying CLI commands, help text, parsing, or JSON output contracts:**

1. Make changes to specs in `aura_cli/options.py` and/or handler in `aura_cli/cli_main.py`
2. Regenerate the CLI reference:
   ```bash
   python3 scripts/generate_cli_reference.py
   ```
3. Run snapshot/contract tests:
   ```bash
   python3 -m pytest -q tests/test_cli_docs_generator.py tests/test_cli_help_snapshots.py tests/test_cli_error_snapshots.py tests/test_cli_main_dispatch.py -k snapshot
   ```
4. If output changes intentionally, update affected files in `tests/snapshots/`
5. CI enforces these contracts — they must pass before merge

---

## File Safety Policy (Critical)

Autonomous code-apply paths enforce an explicit overwrite safety policy for stale-snippet mismatches. This is centralized in `core/file_tools.py`.

**Rules:**
- Stale snippet mismatch + `overwrite_file=true` is **blocked by default**
- Intentional full-file replacement requires:
  - `overwrite_file=true`
  - `old_code=""` (empty string — signals intentional full replacement)
- Policy-block failures are logged as `old_code_mismatch_overwrite_blocked` with policy `explicit_overwrite_file_required`

**Key functions:**
- `allow_mismatch_overwrite_for_change(...)` — checks policy
- `apply_change_with_explicit_overwrite_policy(...)` — enforces and applies

---

## Coding Conventions

### Python Style
- **Indentation:** 4 spaces
- **Functions/variables/modules:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private members:** leading underscore (`_my_private_method`)
- **No formatter config** — match the surrounding file style
- **Type hints** throughout (Python 3.10+)
- **`pathlib.Path`** for all path handling

### Key Patterns
- **Factory pattern:** `default_agents()`, `cache_adapter_factory()`
- **Adapter pattern:** Agent registry wrappers (`PlannerAdapter`, `CriticAdapter`)
- **Lazy imports:** TextBlob, NetworkX loaded only when needed (avoids 2s+ startup cost)
- **Atomic file ops:** `tempfile` + `move` for write safety
- **Persistent queues:** JSON-backed deque with batch operations
- **Structured logging:** `log_json(level, event, details)` — automatic secret masking
- **Custom exceptions:** `CLIParseError`, `FileToolsError`, `OldCodeNotFoundError`, `MismatchOverwriteBlockedError`

### Error Handling
- Distinguish retry-able vs. skip-able failures
- Secret masking is automatic in `core/logging_utils.py` — never log raw API keys
- Use the existing exception hierarchy rather than bare `Exception`

---

## Testing Conventions

- **Framework:** `pytest` (with `unittest`-style test classes)
- **File naming:** `test_*.py`
- **Class naming:** `Test*`
- **Method naming:** `test_*`
- **Test locations:** `tests/` (main suite), `core/tests/` (core module tests)
- **Snapshot tests:** JSON snapshots in `tests/snapshots/` — compare with `_assert_json_snapshot()`
- **Mock factories:** `MagicMock` runtime factories for isolation
- **Subprocess tests:** `run_main_subprocess()` for real CLI execution
- **Parameterized tests:** `subTest` context manager
- **Integration tests:** `tests/integration/` — separated from unit tests

**When adding new CLI commands:** always add snapshot tests alongside

---

## Model Routing

Configured in `settings.json` and `aura.config.json`:

| Task | Model |
|------|-------|
| Code generation (fast) | Gemini 2.0 Flash |
| Code generation (quality) | Claude Sonnet |
| Planning / analysis / critique | Gemini 2.0 Flash |
| Embedding | text-embedding-3-small or local BGE |
| Android local (code) | Qwen coder |
| Android local (plan) | Phi |

Model adapter: `core/model_adapter.py` — supports Gemini, OpenAI, OpenRouter, and local models.

---

## Capability Bootstrap / MCP

AURA can expand its skill set mid-cycle and provision MCP servers:

- **Skill augmentation:** enabled by default (`auto_add_capabilities=true` in config)
- **Self-development goals:** gaps queued as high-priority goals (`auto_queue_missing_capabilities=true`)
- **MCP provisioning:** opt-in via `AURA_AUTO_PROVISION_MCP=true`
- **MCP auto-start:** separately opt-in via `AURA_AUTO_START_MCP_SERVERS=true`
- **MCP ports:** 8001–8007 (configured in `aura.config.json`)
- **Setup script:** `scripts/mcp_server_setup.sh`

---

## Memory System

| Component | File | Backend |
|-----------|------|---------|
| Semantic memory | `memory/brain.py` | SQLite + NetworkX graph |
| Decision log | `memory/store.py` | JSONL with rotation |
| Goal queue | `core/goal_queue.py` | JSON deque |
| Task hierarchy | `memory/task_hierarchy.json` | JSON |
| Skill weights | `memory/skill_weights.json` | JSON |
| Distributed cache | `memory/momento_adapter.py` | Momento |

Runtime memory files are git-ignored. See `.gitignore` for the exact list — only specific named files are excluded (e.g. `memory/brain.db`, `memory/capability_status.json`), not all `memory/*.json` files.

---

## SADD (Sub-Agent Driven Development)

SADD decomposes design specs into parallel workstreams executed by sub-agents, each running the full LoopOrchestrator pipeline.

| Component | File | Purpose |
|-----------|------|---------|
| Types | `core/sadd/types.py` | Dataclasses: WorkstreamSpec, WorkstreamResult, DesignSpec, SessionReport |
| Parser | `core/sadd/design_spec_parser.py` | Markdown → workstream extraction with confidence scoring |
| Graph | `core/sadd/workstream_graph.py` | DAG with state machine for dependency-ordered execution |
| Runner | `core/sadd/sub_agent_runner.py` | Per-workstream LoopOrchestrator wrapper with context injection |
| Coordinator | `core/sadd/session_coordinator.py` | ThreadPool parallel execution with failure routing |
| Store | `core/sadd/session_store.py` | SQLite checkpoints, events, resume |
| MCP Bridge | `core/sadd/mcp_tool_bridge.py` | Tool discovery and matching for workstreams |
| MCP Server | `tools/sadd_mcp_server.py` | Expose SADD as MCP tools (port 8020) |

**CLI commands:** `sadd run --spec <file> [--dry-run]`, `sadd status [--session-id <id>]`, `sadd resume --session-id <id>`

---

## Security & Secrets

- `aura.config.json` contains an API key placeholder — **never commit real keys**
- `.gitignore` excludes `.env`, specific `memory/*.db` and `memory/*.json` runtime files, `.aura_history`, `secrets/` — see `.gitignore` for the exact list
- Logging automatically masks secrets via `core/logging_utils.py`
- Runtime auth managed by `core/runtime_auth.py`
- Use environment variables for all API keys

---

## CI / CD

Workflows in `.github/workflows/`:

| Workflow | Purpose |
|----------|---------|
| `ci.yml` | Python 3.10 & 3.11 tests + CLI docs contract enforcement |
| `aura-agentic-loop.yml` | Agentic loop automation |
| `coder-automation.yml` | Automated code changes |
| `gemini-code-assist.yml` | Gemini integration |
| `copilot-*.yml` | Copilot workspace automation |

CI enforces:
- All pytest tests pass
- Generated `docs/CLI_REFERENCE.md` is current
- CLI snapshot contracts match
- Parser and dispatch contracts pass

---

## Commit and PR Guidelines

- **Commit messages:** imperative, sentence-case (e.g., `Add goal-status snapshot tests`)
- **PRs:** include clear description, rationale, and test results
- **Screenshots:** only needed for UI changes (rare)
- **Never skip hooks** (`--no-verify`) without explicit user request
- **Never force-push to `master`/`main`**

---

## Agent Structured Outputs (CoT)

Agents now use **Chain-of-Thought reasoning** with **Pydantic-structured outputs**:

| Agent | Schema | CoT Sections | Returns |
|-------|--------|--------------|---------|
| `PlannerAgent` | `PlannerOutput` | Analysis → Gaps → Approach → Risks | Plan + confidence + complexity |
| `CriticAgent` | `CriticOutput` | Initial → Completeness → Feasibility → Risks | Assessment + issues + recommendations |
| `CoderAgent` | `CoderOutput` | Problem → Approach → Design → Testing | Code + explanation + edge cases |
| `InnovationSwarm` | `InnovationOutput` | Problem → Techniques → Convergence | Ideas + novelty + diversity metrics |

**Key Features:**
- **Observability**: CoT reasoning logged for debugging
- **Type Safety**: Pydantic validation of LLM outputs
- **Fallback**: Legacy parsing if structured output fails
- **Backward Compatible**: Returns work with existing orchestrator

**Files:**
- `agents/schemas.py` — Pydantic models and prompt templates
- `agents/planner.py` — Structured planning with `plan()` returning dict with `steps`, `confidence`, `reasoning`
- `agents/critic.py` — Structured critique with severity levels
- `agents/coder.py` — Structured code generation with approach explanation
- `agents/innovation_swarm.py` — Multi-technique brainstorming with metrics

## Prompt Manager with Role-Based System Prompts

**Role-Based Personas** (`agents/prompt_manager.py`):

| Role | Persona | Key Expertise |
|------|---------|---------------|
| `planner` | Senior Software Architect | System design, risk assessment, task decomposition |
| `critic` | Principal Engineer | Code review, security analysis, quality standards |
| `coder` | Expert Python Developer | Clean code, TDD, type hints, edge case handling |

**Prompt Caching:**
- LRU cache with TTL (default 1 hour, max 100 entries)
- Reduces token rendering cost for repeated similar prompts
- Access stats via `agent.get_cache_stats()`

```python
from agents.prompt_manager import render_prompt, get_cached_prompt_stats

# Render with role-based system prompt + caching
prompt = render_prompt(
    template_name="planner",  # or "critic", "coder"
    role="planner",           # determines system prompt
    params={"goal": "...", "memory": "..."}
)

# Check cache performance
stats = get_cached_prompt_stats()
# {"hits": 45, "misses": 12, "hit_rate": 0.79, "size": 57}
```

---

## Innovation Catalyst System

**Multi-Agent Brainstorming** (`agents/innovation_swarm.py`, `agents/meta_conductor.py`):

AURA now supports structured innovation sessions with 8 brainstorming techniques:

### Quick Start

```python
from agents.meta_conductor import MetaConductor, InnovationPhase

conductor = MetaConductor()

# Start innovation session
session = conductor.start_session(
    problem_statement="How might we improve code review?",
    techniques=["scamper", "six_hats", "mind_map"]
)

# Execute divergence phase (generate ideas)
result = conductor.execute_phase(session.session_id, InnovationPhase.DIVERGENCE)
# Returns: 50-100+ ideas from 3 techniques

# Execute convergence phase (select best)
result = conductor.execute_phase(session.session_id, InnovationPhase.CONVERGENCE)
# Returns: Top 10-20 ideas with novelty > 0.7
```

### 8 Brainstorming Techniques

| Technique | Bot | Best For |
|-----------|-----|----------|
| **SCAMPER** | `SCAMPERBot` | Structured transformation |
| **Six Thinking Hats** | `SixThinkingHatsBot` | Multi-perspective analysis |
| **Mind Mapping** | `MindMappingBot` | Visual exploration |
| **Reverse Brainstorming** | `ReverseBrainstormingBot` | Problem inversion |
| **Worst Idea** | `WorstIdeaBot` | Constraint removal |
| **Lotus Blossom** | `LotusBlossomBot` | Systematic expansion |
| **Star Brainstorming** | `StarBrainstormingBot` | Structured radiating |
| **Bisociative Association** | `BIABot` | Cross-domain inspiration |

### Innovation Metrics

| Metric | Good | Description |
|--------|------|-------------|
| Diversity Score | >0.7 | Variety across techniques (0-1) |
| Novelty Score | >0.7 | Uniqueness of ideas (0-1) |
| Feasibility Score | >0.6 | Implementation ease (0-1) |
| Convergence Rate | 10-20% | Selection percentage |

### 5-Phase Innovation Process

```
IMMERSION → DIVERGENCE → CONVERGENCE → INCUBATION → TRANSFORMATION
   (1)         (2)          (3)           (4)           (5)
```

1. **Immersion**: Deep problem understanding
2. **Divergence**: Generate 50-150 ideas via InnovationSwarm
3. **Convergence**: Select top 10-20 ideas by composite score
4. **Incubation**: Let ideas develop (simulated)
5. **Transformation**: Convert to actionable tasks

### Files

- `agents/innovation_swarm.py` — Divergence/convergence with 8 techniques
- `agents/meta_conductor.py` — 5-phase session orchestration
- `agents/brainstorming_bots.py` — Individual technique implementations
- `agents/schemas.py` — `InnovationOutput`, `Idea`, `InnovationSessionState`

---

## Important Files to Know

| File | Why It Matters |
|------|---------------|
| `core/orchestrator.py` | Heart of the autonomous loop |
| `core/file_tools.py` | Overwrite safety policy — touch carefully |
| `aura_cli/options.py` | CLI contract definitions — snapshot-tested |
| `aura_cli/cli_main.py` | Command dispatch registry |
| `core/model_adapter.py` | All LLM calls go through here |
| `agents/schemas.py` | Structured output schemas — CoT reasoning |
| `tests/snapshots/` | Contract snapshots — update intentionally |
| `docs/CLI_REFERENCE.md` | Generated — regenerate via script |
| `aura.config.json` | Runtime config — no real secrets |

---

## Memory System (Two-Tier)

AURA uses a two-tier memory system for workplace context:

**Tier 1 — Hot Cache (CLAUDE.md)**
- Top ~30 people, terms, projects
- Active workstreams and current session
- Quick-reference commands and patterns

**Tier 2 — Deep Storage (`memory/` directory)**
| File | Contents |
|------|----------|
| `memory/glossary.md` | Complete term decoder (SADD, MCP, WF-X, etc.) |
| `memory/people/*.md` | Full profiles and preferences |
| `memory/projects/*.md` | Project details and architecture |
| `memory/context/*.md` | Tools, environment, external services |

**Key Abbreviations (Quick Reference)**
| Short | Full | Meaning |
|-------|------|---------|
| SADD | Sub-Agent Driven Development | Parallel workstream execution |
| MCP | Model Context Protocol | External tool integration |
| WF-X | Workflow Number | n8n workflow (WF-0 to WF-6) |
| LoC | Lines of Code | Codebase size metric |
| Swarm | Agent Swarm | Multi-agent coordination (AURA_ENABLE_SWARM=1) |
| Fleet | Workflow Fleet | 7 n8n dispatcher workflows |
| Wave | Execution Wave | Dependency-ordered phase |
| CoT | Chain-of-Thought | Step-by-step LLM reasoning |

**Current Context**
- **Session**: `b00f7213-c107-4735-a16b-33498f0f3e1c`
- **Status**: 8 workstreams executing in 4 waves
- **Goals**: 206+ in queue
- **Swarm**: Enabled (`AURA_ENABLE_SWARM=1`)
- **n8n**: 🟢 Running at http://localhost:5678 (7 workflows **ACTIVE**)
- **MCP**: 4 servers (GitHub ✅, Slack/Sentry/Supabase ⏳)
- **Innovation Catalyst**: 🟢 Active with 8 brainstorming techniques
