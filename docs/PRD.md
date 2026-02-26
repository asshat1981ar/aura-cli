# AURA CLI — Product Requirements Document

**Version:** 1.0  
**Status:** Living Document  
**Repository:** [asshat1981ar/aura-cli](https://github.com/asshat1981ar/aura-cli)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Product Vision](#3-product-vision)
4. [User Personas](#4-user-personas)
5. [Core Concepts & Terminology](#5-core-concepts--terminology)
6. [System Architecture](#6-system-architecture)
7. [Feature Specifications](#7-feature-specifications)
   - 7.1 [CLI Interface](#71-cli-interface)
   - 7.2 [HTTP API Server](#72-http-api-server)
   - 7.3 [Autonomous Loop Pipeline](#73-autonomous-loop-pipeline)
   - 7.4 [Agent System](#74-agent-system)
   - 7.5 [Skills System (23 Skills)](#75-skills-system-23-skills)
   - 7.6 [Memory & Persistence](#76-memory--persistence)
   - 7.7 [Configuration System](#77-configuration-system)
   - 7.8 [GitHub Automation Workflows](#78-github-automation-workflows)
   - 7.9 [MCP Skills Server](#79-mcp-skills-server)
   - 7.10 [Self-Improvement Loops](#710-self-improvement-loops)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Data Models](#9-data-models)
10. [API Reference](#10-api-reference)
11. [Security Requirements](#11-security-requirements)
12. [Error Handling & Failure Modes](#12-error-handling--failure-modes)
13. [Testing Requirements](#13-testing-requirements)
14. [Deployment & Operations](#14-deployment--operations)
15. [Roadmap & Queued Goals](#15-roadmap--queued-goals)
16. [Glossary](#16-glossary)

---

## 1. Executive Summary

**AURA** (Autonomous Unified Reasoning Agent) is an open-source, self-improving autonomous software development system that runs entirely on a developer's local machine or in CI/CD environments. AURA accepts natural language goals, decomposes them into executable steps, generates code, validates it in an isolated sandbox, applies it to the filesystem, runs tests, and self-reflects — all without manual intervention.

AURA is designed to:
- **Automate repetitive development tasks** (refactoring, bug fixes, feature additions)
- **Operate continuously** from a goal queue, processing tasks in autonomous cycles
- **Improve over time** through self-reflection, weakness tracking, and adaptive skill weighting
- **Integrate with GitHub** via Actions workflows, Copilot automation, and Codespaces

**Current State:** 332 passing tests, 23 pluggable skills, 10-phase pipeline, live with OpenRouter free-tier models.

---

## 2. Problem Statement

### Pain Points Addressed

| Pain Point | Current Reality | AURA's Solution |
|---|---|---|
| Repetitive coding tasks | Developer manually refactors/fixes | AURA processes goal queue autonomously |
| Context switching overhead | Developer must recall project state | Brain memory + VectorStore persist context |
| Inconsistent code quality | Depends on developer fatigue/time | Critic + Verifier phases enforce quality every cycle |
| Slow feedback loops | Tests run manually, errors discovered late | Sandbox pre-validates code before any file is written |
| Knowledge loss across sessions | Notes/TODOs scattered | MemoryStore persists cycle summaries, weaknesses, decisions |
| CI/CD disconnected from intent | PRs created manually | GitHub Actions loop processes issues as goals automatically |

---

## 3. Product Vision

> *"AURA is the first always-on autonomous development partner that improves your codebase while you sleep — safely, verifiably, and with full Git history."*

### Guiding Principles

1. **Safety first** — code is sandbox-tested before touching the filesystem; all changes are git-stashed and rollback-able
2. **Transparency** — structured JSON logs every decision; memory is queryable; nothing is a black box
3. **Progressive autonomy** — dry-run mode for observation, full mode for action; human-in-loop gate for high-risk changes
4. **Zero lock-in** — pluggable model backends (OpenRouter, OpenAI, Gemini, local); no proprietary cloud dependency
5. **Self-improving** — every failure becomes a weakness; every weakness feeds the next cycle's plan

---

## 4. User Personas

### Persona A: The Solo Developer (Primary)
- Builds personal projects and OSS tools
- Uses Termux on Android or a local Linux machine
- Wants to offload grunt-work (test writing, refactoring, doc generation)
- Comfortable with CLI; not interested in GUI
- **Key need:** Reliable, low-cost autonomous coding that doesn't break their project

### Persona B: The DevOps/Platform Engineer
- Manages CI/CD pipelines and GitHub automation
- Wants AURA running as a scheduled GitHub Action
- Needs confidence that autonomous changes are safe and reviewable
- **Key need:** PR-based workflow with human review gate for risky changes

### Persona C: The AI/ML Researcher
- Experimenting with agentic frameworks and self-improving systems
- Wants to swap in different LLMs, add custom skills, modify the pipeline
- Reads the source code; contributes new agents/skills
- **Key need:** Clean extension points (SkillBase, agent adapters, policy plugins)

---

## 5. Core Concepts & Terminology

| Term | Definition |
|---|---|
| **Goal** | A natural language description of a desired code change or task (e.g., "Add retry logic to the HTTP client") |
| **Goal Queue** | FIFO list of goals waiting to be processed (`memory/goal_queue.json`) |
| **Cycle** | One complete execution of the 10-phase autonomous loop for a single goal |
| **Phase** | A discrete step in the loop pipeline (ingest, plan, critique, synthesize, act, sandbox, apply, verify, reflect) |
| **Agent** | A specialized component that handles one phase of the pipeline |
| **Skill** | A pluggable analysis module (complexity scorer, security scanner, etc.) invoked during skill dispatch |
| **Brain** | The SQLite-backed long-term memory storing recalled context, weaknesses, and vector embeddings |
| **Task Bundle** | The structured artifact passed between phases containing the goal, steps, file targets, and fix hints |
| **AURA_TARGET** | A directive comment (`# AURA_TARGET: path/to/file.py`) the LLM includes in generated code to specify the target file |
| **Sandbox** | An isolated subprocess that executes generated code before it is written to disk |
| **Dry Run** | A mode where the loop runs fully but skips all filesystem writes and memory persistence |
| **Weakness** | A recorded failure pattern (e.g., "tends to hallucinate import paths") used to guide future plans |
| **Skill Weight** | A learned score (EMA) for each skill indicating its reliability for a given goal type |

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Entry Points                     │
│   CLI (main.py)    HTTP API (:8001)    GitHub Actions    │
└──────────┬─────────────────┬────────────────┬───────────┘
           │                 │                │
           ▼                 ▼                ▼
┌─────────────────────────────────────────────────────────┐
│              AURA Runtime (cli_main.create_runtime)      │
│  GoalQueue  ConfigManager  Brain  ModelAdapter  GitTools │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           LoopOrchestrator (core/orchestrator.py)        │
│                                                          │
│  [1] Ingest → [2] Skill Dispatch → [3] Plan             │
│  [4] Critique → [5] Synthesize → [6] Act                │
│  [6a] Sandbox → [7] Apply → [8] Verify → [9] Reflect    │
└──────────────────────────┬──────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Agents     │  │   Skills     │  │   Memory     │
│ PlannerAgent │  │ 23 pluggable │  │ Brain (SQL)  │
│ CoderAgent   │  │ skills via   │  │ MemoryStore  │
│ CriticAgent  │  │ SkillBase    │  │ (JSON tiers) │
│ SandboxAgent │  │              │  │ GoalQueue    │
│ VerifierAgent│  │              │  │ LocalCache   │
└──────────────┘  └──────────────┘  └──────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              Model Adapter (core/model_adapter.py)       │
│   OpenRouter → OpenAI → Gemini CLI → Local Model        │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Gateway | OpenRouter (primary), OpenAI, Gemini CLI |
| Default Model | `google/gemini-2.0-flash-exp:free` (free tier) |
| Persistence | SQLite (Brain), JSON files (queue/archive/store), in-memory TTL dict (cache) |
| HTTP Server | FastAPI (uvicorn) |
| Testing | pytest (332 passing) |
| CI/CD | GitHub Actions |
| Containerization | Docker (Dockerfile included) |
| VCS Integration | GitPython / subprocess git |

---

## 7. Feature Specifications

### 7.1 CLI Interface

**Entry Point:** `python3 main.py [flags]`

#### 7.1.1 Command-Line Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dry-run` | bool | false | Run loop without writing files or persisting memory |
| `--add-goal <text>` | str | — | Add a goal to the queue; can combine with `--run-goals` |
| `--run-goals` | bool | false | Execute all queued goals and exit (non-interactive) |
| `--decompose` | bool | false | Decompose complex goals into hierarchical sub-tasks before queueing |

**Usage Examples:**
```bash
# Add a goal and run immediately (non-interactive)
python3 main.py --add-goal "Add retry logic to HTTP client" --run-goals

# Queue a goal without running
python3 main.py --add-goal "Refactor goal_queue.py to use SQLite"

# Dry run (observe without writing)
python3 main.py --run-goals --dry-run

# Decompose a complex goal
python3 main.py --add-goal "Build user authentication system" --decompose --run-goals
```

#### 7.1.2 Interactive CLI Commands

After startup, AURA enters an interactive `Command >` prompt:

| Command | Description |
|---|---|
| `add <goal text>` | Append a goal to the queue |
| `run` | Execute all queued goals |
| `status` | Display queue contents, archived goals, task hierarchy, loop state |
| `doctor` | Run health checks (Python version, env vars, SQLite, git, pytest) |
| `clear` | Clear terminal screen |
| `help` | List available commands with descriptions |
| `exit` | Gracefully exit AURA |

#### 7.1.3 CLI Runtime Behavior

- Readline history persisted to `memory/.aura_history` (max 1000 entries)
- Structured JSON log lines emitted to stdout (`core/logging_utils.log_json`)
- `AURA_SKIP_CHDIR=1` prevents `os.chdir()` (required for test environments)
- Ctrl+C handled gracefully; in-progress cycles are stashed before exit

---

### 7.2 HTTP API Server

**File:** `aura_cli/server.py`  
**Framework:** FastAPI  
**Default Port:** 8001 (override via `PORT` env var)  
**Default Host:** 127.0.0.1 (override via `AGENT_API_HOST`)

**Start:** `uvicorn aura_cli.server:app --port 8001`

#### 7.2.1 Authentication

All endpoints require `Authorization: Bearer <token>` when `AGENT_API_TOKEN` is set. If unset, auth is disabled (development mode).

#### 7.2.2 Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | Yes | Server health, model provider status, feature flags |
| GET | `/tools` | Yes | List available tools with descriptions |
| POST | `/call` | Yes | Invoke a tool by name |
| GET | `/metrics` | Yes | Runtime metrics (cycle count, skill stats, queue depth) |
| POST | `/run` | Yes | Trigger a full goal cycle (requires `AGENT_API_ENABLE_RUN=1`) |

#### 7.2.3 Tool Invocations (POST /call)

**Request schema:**
```json
{"tool": "ask|run|env|goal", "args": [...]}
```

| Tool | Args | Response | Notes |
|---|---|---|---|
| `ask` | `[question]` | `{"result": "..."}` | LLM question answering |
| `run` | `[shell_cmd]` | SSE stream | Requires `AGENT_API_ENABLE_RUN=1`; commands sanitized |
| `env` | `[]` | `{"TERMUX_*": "..."}` | Environment snapshot |
| `goal` | `[goal_text]` | SSE stream | Full cycle with live events: start→health→cycle→complete |

---

### 7.3 Autonomous Loop Pipeline

**Orchestrator:** `core/orchestrator.py::LoopOrchestrator`

Each `run_cycle()` executes these phases in order:

#### Phase Details

| # | Phase | Agent | Input | Output | On Failure |
|---|---|---|---|---|---|
| 1 | **Ingest** | `IngestAgent` | project_root, goal | context snapshot | Skip (non-fatal) |
| 2 | **Skill Dispatch** | `SkillDispatcher` | goal, context | skill results dict | Skip (non-fatal) |
| 3 | **Plan** | `PlannerAgent` | goal, memory, weaknesses, skill results | `{"steps": [...], "risks": [...]}` | Retry up to 3× |
| 4 | **Critique** | `CriticAgent` | task, plan | `{"issues": [...], "fixes": [...]}` | Skip (non-fatal) |
| 5 | **Synthesize** | `SynthesizerAgent` | plan + critique | task_bundle with file targets | Abort cycle |
| 6 | **Act** | `CoderAgent` | task_bundle | `{"changes": [...]}` | Retry up to 3× |
| 6a | **Sandbox** | `SandboxAgent` | generated code snippet | `{"status": "pass/fail/skip"}` | Inject fix_hints → retry Act |
| 7 | **Apply** | `FileTools` | change set | applied file paths | Route to re-plan or skip |
| 8 | **Verify** | `VerifierAgent` | applied changes | test results + score | Route to retry Act |
| 9 | **Reflect** | `ReflectorAgent` | cycle history | cycle summary | Non-fatal |

#### Stopping Policies

| Policy | Trigger | Config Key |
|---|---|---|
| `SlidingWindowPolicy` | After N cycles (default: 5) | `policy_max_cycles` |
| `TimeBoundPolicy` | After N seconds | `policy_max_seconds` |
| `ResourceBoundPolicy` | After N tokens consumed | (custom) |

#### Failure Routing (`_route_failure`)

Returns one of three actions:
- `"retry_act"` — code generation failed; retry with fix hints
- `"replan"` — fundamental strategy failure; return to plan phase
- `"skip"` — environmental/external issue; abandon this cycle

---

### 7.4 Agent System

#### Base Class: `agents/base.py::AgentBase`
All agents inherit `AgentBase` which provides:
- `run(input_data: dict) -> dict` — public entry point
- `_run(input_data: dict) -> dict` — override in subclasses
- Error swallowing: exceptions are caught and returned as `{"error": "..."}`

#### Pipeline Agents

| Agent | File | Responsibility |
|---|---|---|
| `IngestAgent` | `agents/ingest.py` | Gather project files, git status, recent changes |
| `PlannerAgent` | `agents/planner.py` | LLM-based step-by-step planning |
| `CriticAgent` | `agents/critic.py` | Adversarial plan review |
| `SynthesizerAgent` | `agents/synthesizer.py` | Merge plan + critique into task bundle |
| `CoderAgent` | `agents/coder.py` | LLM code generation (supports `# AURA_TARGET:` directive) |
| `SandboxAgent` | `agents/sandbox.py` | Isolated subprocess code execution |
| `VerifierAgent` | `agents/verifier.py` | Run tests + linters; compute quality score |
| `ReflectorAgent` | `agents/reflector.py` | Cycle summary; update brain memory |
| `DebuggerAgent` | `agents/debugger.py` | LLM-based error diagnosis on failure |
| `RouterAgent` | `agents/router.py` | EMA-weighted model routing by task type |

#### Adapter Pattern
Each agent is wrapped in an **Adapter** class (`agents/registry.py`) that:
1. Normalizes input/output format for the orchestrator
2. Handles schema validation
3. Provides a consistent `run(phase_input) -> phase_output` interface

`ActAdapter` additionally:
- Parses `# AURA_TARGET: path/to/file.py` from generated code
- Falls back to heuristic keyword-based file path scoring
- Generates a new file in `core/` when no target is found

---

### 7.5 Skills System (23 Skills)

**Registry:** `agents/skills/registry.py::all_skills(brain, model) -> Dict[str, SkillBase]`

All skills extend `agents/skills/base.py::SkillBase`:
- `run(input_data: dict) -> dict` — public entry, **never raises**
- `_run(input_data: dict) -> dict` — implement in subclass
- Errors returned as `{"error": "description"}`

#### Complete Skills Catalog

| # | Skill Name | Primary Input | Key Output Fields |
|---|---|---|---|
| 1 | `dependency_analyzer` | `project_root` | `packages`, `conflicts`, `vulnerabilities` |
| 2 | `architecture_validator` | `project_root` | `circular_deps`, `coupling_score` |
| 3 | `complexity_scorer` | `code` / `project_root` | `functions`, `high_risk_count` |
| 4 | `test_coverage_analyzer` | `project_root` | `coverage_pct`, `meets_target` |
| 5 | `doc_generator` | `code` / `project_root` | `generated_docstrings`, `undocumented_count` |
| 6 | `performance_profiler` | `code` | `hotspots`, `antipatterns` |
| 7 | `refactoring_advisor` | `code` / `project_root` | `suggestions`, `smell_count` |
| 8 | `schema_validator` | `schema`, `instance`, `code` | `valid`, `errors`, `pydantic_models` |
| 9 | `security_scanner` | `code` / `project_root` | `findings`, `critical_count` |
| 10 | `type_checker` | `project_root` / `file_path` | `type_errors`, `annotation_coverage_pct` |
| 11 | `linter_enforcer` | `project_root` / `file_path` | `violations`, `naming_violations` |
| 12 | `incremental_differ` | `old_code`, `new_code` | `diff_summary`, `added_symbols`, `removed_symbols` |
| 13 | `tech_debt_quantifier` | `project_root` | `debt_score`, `debt_items` |
| 14 | `api_contract_validator` | `code` / `project_root` | `endpoints`, `breaking_changes` |
| 15 | `generation_quality_checker` | `task`, `generated_code` | `quality_score`, `issues`, `intent_match_score` |
| 16 | `git_history_analyzer` | `project_root` | `hotspot_files`, `patterns` |
| 17 | `skill_composer` | `goal` | `workflow`, `goal_category` |
| 18 | `error_pattern_matcher` | `current_error`, `error_history?` | `matched_pattern`, `suggested_fix`, `fix_steps` |
| 19 | `code_clone_detector` | `project_root` | `exact_clones`, `near_duplicates` |
| 20 | `adaptive_strategy_selector` | `goal`, `record_result?` | `recommended_strategy`, `confidence` |
| 21 | `web_fetcher` | `url` or `query` | `text`, `title`, `source`, `truncated` |
| 22 | `symbol_indexer` | `project_root` | `symbols`, `symbol_count`, `name_index`, `import_graph` |
| 23 | `multi_file_editor` | `goal`, `project_root?`, `symbol_map?` | `change_plan`, `affected_count`, `warnings` |

**Adding a New Skill:**
1. Create `agents/skills/your_skill.py` extending `SkillBase`
2. Set `name = "your_skill"` class attribute
3. Implement `_run(self, input_data: dict) -> dict`
4. Register in `agents/skills/registry.py::all_skills()`

---

### 7.6 Memory & Persistence

#### L1 Cache: LocalCacheAdapter / MomentoAdapter

**File:** `memory/local_cache_adapter.py` / `memory/momento_adapter.py`  
**Selection:** `memory/cache_adapter_factory.py::create_cache_adapter()`

- Returns `MomentoAdapter` if `MOMENTO_API_KEY` is set (cloud cache with sub-ms latency)
- Returns `LocalCacheAdapter` otherwise (always available, in-process, zero dependencies)

| Operation | LocalCacheAdapter | MomentoAdapter |
|---|---|---|
| `get(cache, key)` | dict with TTL | HTTP to Momento cloud |
| `set(cache, key, value, ttl)` | dict + expiry tuple | HTTP to Momento cloud |
| `list_push(cache, key, value)` | dict of lists | Momento list type |
| `list_fetch(cache, key)` | return list | Momento list fetch |
| `publish(topic, data)` | SQLite event log | Momento Topics |
| `subscribe(topic, handler)` | SQLite poll | Momento WebSocket |

#### L2 Brain: SQLite Memory

**File:** `memory/brain.py::Brain`  
**Database:** `memory/brain_v2.db`

```sql
-- Tables:
memory(id INTEGER PRIMARY KEY, content TEXT)
weaknesses(id INTEGER PRIMARY KEY, description TEXT, timestamp DATETIME)
vector_store_data(id INTEGER PRIMARY KEY, content TEXT, embedding BLOB)
response_cache(prompt_hash TEXT PRIMARY KEY, response TEXT, created_at DATETIME)
```

**Key Methods:**

| Method | Description |
|---|---|
| `remember(data)` | Persist arbitrary data to memory table |
| `recall_all() -> List[str]` | Retrieve all memory entries |
| `add_weakness(description)` | Record a failure pattern |
| `recall_weaknesses() -> List[str]` | Get all recorded weaknesses |
| `reflect() -> str` | Generate textual summary of memory state |
| `relate(a, b)` | Add edge to NetworkX concept graph |

#### L3 MemoryStore: Tier-Based JSON

**File:** `memory/store.py::MemoryStore`  
**Root:** `memory/store/`

Each tier is a JSON array file: `memory/store/{tier}.json`

```python
store.put("cycle_summaries", {"goal": "...", "score": 0.9, "ts": "..."})
store.query("cycle_summaries", limit=100) -> List[Dict]
store.append_log({"event": "...", "detail": "..."})
```

#### Goal Queue & Archive

| File | Format | Purpose |
|---|---|---|
| `memory/goal_queue.json` | `["goal1", "goal2"]` | FIFO queue of pending goals |
| `memory/goal_archive.json` | `[["goal", score], ...]` | Completed goals with quality scores |
| `memory/task_hierarchy_v2.json` | Nested tree | Decomposed sub-task hierarchies |
| `memory/strategy_stats.json` | `{strategy: {wins, losses}}` | Adaptive strategy selector weights |

---

### 7.7 Configuration System

**File:** `core/config_manager.py::ConfigManager`  
**Resolution order:** Runtime overrides → `AURA_*` env vars → `aura.config.json` → defaults

#### aura.config.json Schema

```json
{
  "model_name": "google/gemini-2.0-flash-exp:free",
  "api_key": "your-openrouter-api-key",
  "max_retries": 3,
  "dry_run": false,
  "decompose": false,
  "max_iterations": 10,
  "max_cycles": 5,
  "strict_schema": false,
  "policy_name": "sliding_window",
  "policy_max_cycles": 5,
  "policy_max_seconds": 120,
  "model_routing": {
    "code_generation": "google/gemini-2.0-flash-exp:free",
    "planning":        "google/gemini-2.0-flash-exp:free",
    "analysis":        "google/gemini-2.0-flash-exp:free",
    "critique":        "google/gemini-2.0-flash-exp:free",
    "embedding":       "openai/text-embedding-3-small",
    "fast":            "google/gemini-2.0-flash-exp:free",
    "quality":         "google/gemini-2.5-pro"
  }
}
```

#### Environment Variables Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `AURA_API_KEY` | str | — | OpenRouter API key (alias) |
| `OPENROUTER_API_KEY` | str | — | OpenRouter API key (primary) |
| `OPENAI_API_KEY` | str | — | OpenAI API key (fallback) |
| `GEMINI_CLI_PATH` | str | `/usr/bin/gemini` | Path to Gemini CLI binary |
| `AURA_DRY_RUN` | bool | 0 | Set 1 to disable all writes |
| `AURA_SKIP_CHDIR` | bool | 0 | Set 1 to skip os.chdir() (tests) |
| `AURA_MAX_CYCLES` | int | 5 | Max loop cycles per goal |
| `AURA_STRICT_SCHEMA` | bool | 0 | Abort cycle on schema validation failure |
| `AURA_MODEL_ROUTING_<KEY>` | str | see config | Override individual model routing slots |
| `AGENT_API_TOKEN` | str | — | Bearer token for HTTP API |
| `AGENT_API_ENABLE_RUN` | bool | 0 | Enable /call run tool |
| `MCP_API_TOKEN` | str | — | Bearer token for MCP Skills Server |
| `MOMENTO_API_KEY` | str | — | Enables Momento cloud cache (L1) |

---

### 7.8 GitHub Automation Workflows

#### aura-agentic-loop.yml
Runs the AURA loop in CI — daily and on manual trigger.

**Triggers:**
- `schedule`: Daily 02:00 UTC
- `workflow_dispatch`: Manual with `goal`, `dry_run`, `max_cycles` inputs

**3-job pipeline:**

| Job | Purpose | Output |
|---|---|---|
| `aura-loop` | Run goal queue or one-off goal | Commits changes, creates error issues, uploads artifacts |
| `aura-reflect` | Skills analysis + test suite | Step summary with tech debt, coverage, security scores |
| `aura-skill-improve` | Skill composer analysis | Recommended workflows for 3 improvement categories |

#### copilot-autofix.yml
Posts lint + test summary as PR comment using GitHub Copilot branding.

#### copilot-workspace.yml
Converts GitHub issues labeled `aura-goal` into AURA queue entries.  
Auto-labels issue `aura-queued` and posts acknowledgement comment.

#### gemini-code-assist.yml
Runs pylint + AURA skills scan (complexity + security) on changed files in PRs.  
Posts structured review comment.

#### coder-automation.yml
Validates devcontainer config and Python environment on `requirements.txt` changes.  
Auto-creates `.devcontainer/devcontainer.json` if missing.

#### ci.yml
Matrix CI across Python 3.10 and 3.11 on every push/PR to main.

---

### 7.9 MCP Skills Server

**File:** `tools/aura_mcp_skills_server.py`  
**Port:** 8002  
**Start:** `uvicorn tools.aura_mcp_skills_server:app --port 8002`

Exposes all 23 skills as MCP-compatible HTTP endpoints.

| Method | Path | Description |
|---|---|---|
| GET | `/tools` | List all skill tools with input schemas |
| POST | `/call` | `{"tool": "skill_name", "args": {...}}` |
| GET | `/skill/{name}` | Describe a single skill |
| GET | `/health` | Health check |

**MCP Config** (`.vscode/mcp.json` or `~/.config/github-copilot/mcp.json`):
```json
{
  "mcpServers": {
    "aura-skills": {
      "type": "http",
      "url": "http://localhost:8002"
    }
  }
}
```

---

### 7.10 Self-Improvement Loops

AURA includes 7 autonomous improvement loops that fire after each cycle:

| Loop | File | Trigger | Action |
|---|---|---|---|
| `ReflectionLoop` | `core/reflection_loop.py` | Every cycle | Summarize outcomes; update Brain |
| `WeaknessRemediatorLoop` | `core/weakness_remediator.py` | When weaknesses recorded | Generate fix goals; queue them |
| `ConvergenceEscapeLoop` | `core/convergence_escape.py` | Oscillating scores | Switch strategy; vary prompt |
| `HealthMonitor` | `core/health_monitor.py` | Periodic | Check system health; queue alerts |
| `MemoryCompaction` | `core/memory_compaction.py` | Memory threshold | Summarize and compress old memories |
| `PropagationEngine` | `core/propagation_engine.py` | Context changes | Propagate context updates across goals |
| `CASPA-W` | `core/adaptive_pipeline.py` | Every N cycles | Contextually Adaptive Self-Propagating Autonomous Workflow orchestration |

---

## 8. Non-Functional Requirements

### 8.1 Performance
- Cycle latency: < 30 seconds for simple goals with a fast model (gemini-flash)
- Sandbox timeout: 30 seconds per code snippet execution
- Cache L1 hit → < 1ms response (LocalCacheAdapter)
- Memory recall: < 100ms for up to 10,000 Brain entries

### 8.2 Reliability
- All skills: must not raise exceptions (return `{"error": "..."}` instead)
- All agents: wrapped in try/except at adapter layer
- Git stash/pop: protects filesystem before/after every cycle
- Dry run: 100% of functionality exercisable without side effects

### 8.3 Scalability
- Goal queue: supports unlimited entries (JSON file backed)
- Brain: SQLite handles millions of memory entries
- Skills: independently parallelizable; no shared state

### 8.4 Portability
- Runs on: Linux, macOS, Android (Termux), GitHub Actions (ubuntu-latest)
- Python 3.10+ required; no OS-specific APIs
- Docker support via `Dockerfile`
- Codespaces support via `.devcontainer/devcontainer.json`

---

## 9. Data Models

### 9.1 Change Set (ActAdapter output)
```json
{
  "changes": [
    {
      "file_path": "core/goal_queue.py",
      "old_code": "def next(self):\n    ...",
      "new_code": "def next(self) -> Optional[str]:\n    ...",
      "overwrite_file": false
    }
  ]
}
```
- `old_code = ""` + `overwrite_file = true` → full file overwrite
- `old_code` not found → raises `OldCodeNotFoundError`

### 9.2 Task Bundle (Synthesizer output)
```json
{
  "goal": "Add retry logic to HTTP client",
  "tasks": ["1. Locate _make_request", "2. Wrap with retry decorator", "..."],
  "target_files": ["core/model_adapter.py"],
  "fix_hints": [],
  "context": {"skill_results": {...}, "weaknesses": [...]}
}
```

### 9.3 Cycle Summary (Reflector output)
```json
{
  "goal": "...",
  "cycle": 3,
  "score": 0.87,
  "phases_completed": ["ingest","plan","critique","synthesize","act","sandbox","apply","verify","reflect"],
  "changes_applied": ["core/model_adapter.py"],
  "weaknesses_added": [],
  "ts": "2026-02-26T05:32:00Z"
}
```

### 9.4 Skill Result (SkillBase output)
```json
{
  "skill": "security_scanner",
  "critical_count": 0,
  "findings": [],
  "error": null
}
```

---

## 10. API Reference

### POST /call
```bash
curl -X POST http://localhost:8001/call \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool": "goal", "args": ["Add input validation to all API endpoints"]}'
```

**SSE Event stream:**
```
event: start
data: {"goal": "Add input validation..."}

event: health
data: {"model_ok": true, "git_ok": true}

event: cycle
data: {"cycle": 1, "score": 0.0, "phase": "reflect"}

event: complete
data: {"cycles": 3, "final_score": 0.92, "changes": ["core/file_tools.py"]}
```

### POST /call (ask tool)
```bash
curl -X POST http://localhost:8001/call \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -d '{"tool": "ask", "args": ["What files handle goal processing?"]}'
```
```json
{"result": "Goal processing is handled by core/task_handler.py..."}
```

---

## 11. Security Requirements

| Requirement | Implementation |
|---|---|
| API key protection | `.env` is gitignored; never committed |
| Shell command execution | `core/sanitizer.sanitize_command()` allowlist |
| Subprocess isolation | SandboxAgent uses `subprocess` with timeout; no shell=True |
| HTTP API auth | Bearer token via `AGENT_API_TOKEN` |
| MCP server auth | Bearer token via `MCP_API_TOKEN` |
| File path jail | `PROJECT_ROOT` validation prevents writes outside project |
| No eval/exec | Code review enforced; no dynamic execution of untrusted strings |
| Git audit trail | All changes committed with structured messages + Co-authored-by |

---

## 12. Error Handling & Failure Modes

| Error | Cause | Recovery |
|---|---|---|
| `OldCodeNotFoundError` | `old_code` string not found in target file | Route to `retry_act`; inject fix hint with fuzzy match |
| `FileNotFoundError` | LLM hallucinated a target file path | `ActAdapter` falls back to keyword scoring; generates new file |
| `NameResolutionError` | No network (DNS failure) | Model adapter falls back: OpenRouter → OpenAI → Gemini → Local |
| Schema validation failure | Phase output missing required keys | Log warning; continue if `strict_schema=False` |
| Sandbox timeout | Code snippet exceeded 30s | `SandboxResult.timed_out=True`; inject "optimize for speed" hint |
| Sandbox syntax error | Generated code has syntax errors | Inject stderr into fix_hints; retry Act |
| Git stash conflict | Concurrent writes to tracked files | `git checkout --theirs` on pycache/binary conflicts |
| Brain DB corruption | SQLite journal crash | `_load_queue` returns empty deque; fresh start |

---

## 13. Testing Requirements

### 13.1 Current State
- **332 passing tests** across 30+ test files
- Test runner: `pytest -q --tb=short`
- Required env: `AURA_SKIP_CHDIR=1`

### 13.2 Test Organization

| Directory | Coverage |
|---|---|
| `tests/` | Unit tests for all core modules, agents, skills |
| `tests/integration/` | End-to-end orchestrator tests (mock LLM) |
| `tests/fakes/` | Test doubles for Brain, ModelAdapter, etc. |
| `core/test_goal_queue.py` | Legacy SQLite queue tests (excluded from CI) |

### 13.3 Test Requirements per Feature

| Feature | Required Coverage |
|---|---|
| All 23 Skills | `run()` returns dict, never raises, handles bad input |
| All agent adapters | Happy path + failure path |
| File tools | `replace_code`, `_safe_apply_change`, overwrite mode |
| LocalCacheAdapter | TTL expiry, list ops, thread safety, pub/sub |
| HTTP server | All endpoints, auth, SSE streaming |
| CLI commands | Interactive + argparse flags |

---

## 14. Deployment & Operations

### 14.1 Local (Termux/Linux)
```bash
# Install
git clone https://github.com/asshat1981ar/aura-cli
cd aura-cli
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY

# Run
python3 main.py --add-goal "Your goal here" --run-goals
```

### 14.2 GitHub Codespaces
Open repo in Codespaces → `.devcontainer/devcontainer.json` auto-installs all deps.  
Ports 8001 and 8002 are auto-forwarded.

### 14.3 Docker
```bash
docker build -t aura-cli .
docker run -e OPENROUTER_API_KEY=sk-or-... aura-cli python3 main.py --run-goals
```

### 14.4 GitHub Actions (Autonomous)
1. Add `AURA_API_KEY` as repository secret
2. Push goal to `memory/goal_queue.json`
3. Trigger `aura-agentic-loop.yml` manually or wait for daily schedule

### 14.5 Monitoring

| Signal | Location | Meaning |
|---|---|---|
| `cycle_complete` log event | stdout | Cycle finished; check `score` field |
| `apply_change_failed` | stdout | File write error; see `details.error` |
| GitHub issue `aura-loop-error` | GitHub Issues | Loop detected errors in CI; auto-created |
| `memory/store/cycle_summaries.json` | filesystem | Persistent cycle history |
| `/metrics` endpoint | HTTP API | Live skill stats and queue depth |

---

## 15. Roadmap & Queued Goals

The following 15 goals are queued in `memory/goal_queue.json` for autonomous implementation:

| Priority | Goal | Value |
|---|---|---|
| 1 | Atomic multi-file transaction support in ApplicatorAgent | Prevents partial applies |
| 2 | CoderAgent sandbox retry loop (3× with stderr feedback) | Reduces human intervention |
| 3 | VectorStore-based token budget compression | Fixes planning phase token overflow |
| 4 | Circular import detection in architecture_validator | Prevents architectural regressions |
| 5 | Per-skill metrics → /metrics → skill_weight_adapter | Data-driven skill selection |
| 6 | Incremental test coverage (git diff only) | 10× faster verification phase |
| 7 | LLM-based goal type classifier | Smarter skill dispatch |
| 8 | OldCodeNotFoundError recovery via git fuzzy history | Handles stale change sets |
| 9 | End-to-end integration tests for LoopOrchestrator | Catches cross-phase regressions |
| 10 | Human-in-loop gate for security criticals / coverage drops | Safe autonomous operation |
| 11 | Skill chaining (security → remediation goals) | Emergent multi-skill capability |
| 12 | Response cache preloading on startup | Eliminates cold-start latency |
| 13 | `skill_failure_analyzer` skill | Self-healing skill system |
| 14 | Verification-phase oscillation escape | Handles alternating pass/fail |
| 15 | Adaptive context window sizing by goal complexity | Balances speed vs thoroughness |

---

## 16. Glossary

| Term | Definition |
|---|---|
| **AURA** | Autonomous Unified Reasoning Agent |
| **CASPA-W** | Contextually Adaptive Self-Propagating Autonomous Workflow |
| **Brain** | SQLite-backed long-term memory for AURA |
| **EMA** | Exponential Moving Average — used for skill weight updates |
| **MCP** | Model Context Protocol — standard for exposing tools to AI assistants |
| **OpenRouter** | LLM gateway service providing access to 100+ models via one API key |
| **Phase** | One step in the 10-step autonomous loop pipeline |
| **Sandbox** | Isolated subprocess that executes code before it touches the filesystem |
| **Skill** | Pluggable static-analysis module implementing `SkillBase` |
| **Task Bundle** | Structured dict passed between pipeline phases |
| **TTL** | Time-To-Live — expiry time for cache entries |
| **AURA_TARGET** | Comment directive in generated code specifying the target file path |
| **Weakness** | A recorded failure pattern stored in Brain, used to guide future plans |
| **Cycle** | One complete execution of all 9 pipeline phases for a single goal |
| **Stopping Policy** | Rule governing when the loop terminates (by cycle count, time, or resources) |

---

*This document is auto-generated from the AURA CLI codebase and maintained as part of the repository.*  
*Last updated: 2026-02-26*  
*To update: analyze the codebase and regenerate — `python3 main.py --add-goal "Update PRD document in docs/PRD.md"`*
