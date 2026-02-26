# AURA CLI — Product Requirements Document

**Version:** 2.0  
**Status:** Living Document  
**Last Updated:** 2025  
**Repository:** [asshat1981ar/aura-cli](https://github.com/asshat1981ar/aura-cli)  
**Previous Version:** 1.0 (bb32efb) → **This Version covers bb32efb → 5c9f90f (+7 commits, +15 features)**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Product Vision](#3-product-vision)
4. [User Personas](#4-user-personas)
5. [Core Concepts & Terminology](#5-core-concepts--terminology)
6. [System Architecture](#6-system-architecture)
   - 6.1 [High-Level Architecture Diagram](#61-high-level-architecture-diagram)
   - 6.2 [Entry Points & CLI Systems](#62-entry-points--cli-systems)
   - 6.3 [Dependency Hubs](#63-dependency-hubs)
   - 6.4 [Leaf / Foundation Modules](#64-leaf--foundation-modules)
7. [Feature Specifications](#7-feature-specifications)
   - 7.1 [CLI Interface](#71-cli-interface)
   - 7.2 [HTTP API Server](#72-http-api-server)
   - 7.3 [Autonomous Loop Pipeline (10 Phases)](#73-autonomous-loop-pipeline-10-phases)
   - 7.4 [Agent System (16 Agents)](#74-agent-system-16-agents)
   - 7.5 [Skills System (28 Skills)](#75-skills-system-28-skills)
   - 7.6 [Memory Architecture (5 Layers)](#76-memory-architecture-5-layers)
   - 7.7 [Advanced Orchestration Modules](#77-advanced-orchestration-modules)
   - 7.8 [MCP Server Ecosystem (5 Servers)](#78-mcp-server-ecosystem-5-servers)
   - 7.9 [Configuration System](#79-configuration-system)
   - 7.10 [GitHub Automation Workflows](#710-github-automation-workflows)
   - 7.11 [Stopping Policies](#711-stopping-policies)
   - 7.12 [Self-Improvement Loops](#712-self-improvement-loops)
8. [New Features Since v1.0 PRD (15 Implemented)](#8-new-features-since-v10-prd-15-implemented)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Performance Benchmarks](#10-performance-benchmarks)
11. [Data Models](#11-data-models)
12. [API Reference](#12-api-reference)
13. [Security Requirements](#13-security-requirements)
14. [Error Handling & Failure Modes](#14-error-handling--failure-modes)
15. [Testing Requirements & Coverage Gaps](#15-testing-requirements--coverage-gaps)
16. [Known Technical Debt](#16-known-technical-debt)
17. [Deployment & Operations](#17-deployment--operations)
18. [Roadmap & Future Goals](#18-roadmap--future-goals)
19. [Glossary](#19-glossary)

---

## 1. Executive Summary

**AURA** (Autonomous Unified Reasoning Agent) is an open-source, self-improving autonomous software development system that runs entirely on a developer's local machine or in CI/CD environments. AURA accepts natural language goals, decomposes them into executable steps, generates code, validates it in an isolated sandbox, applies it atomically to the filesystem, runs tests, and self-reflects — all without manual intervention.

AURA is designed to:
- **Automate repetitive development tasks** (refactoring, bug fixes, feature additions, documentation)
- **Operate continuously** from a priority goal queue, processing tasks in fully autonomous cycles
- **Improve over time** through deep self-reflection, weakness tracking, oscillation detection, and adaptive skill weighting
- **Integrate with GitHub** via Actions workflows, Copilot automation, Gemini PR review, and Codespaces
- **Expose capabilities as MCP tools** to external agents and LLM clients via 5 distinct MCP servers

**Current State (v2.0):**
- **377 passing tests** (up from 332), 0 failures
- **28 pluggable skills** (up from 23)
- **10-phase pipeline** with atomic apply, 3x sandbox retry, and human gate
- **5 MCP servers** (up from 1 documented)
- **5-layer memory architecture** (L0 in-memory → L4 goal queue JSON)
- **15 roadmap goals from v1.0 PRD: ALL IMPLEMENTED**
- Live with OpenRouter API key on Android/Termux

---

## 2. Problem Statement

Modern software developers face mounting cognitive load: managing backlogs, writing repetitive boilerplate, fixing recurring error patterns, maintaining test coverage, enforcing standards, and documenting changes. Existing AI code assistants (GitHub Copilot, ChatGPT, Cursor) are **reactive** — they require continuous human prompting and cannot autonomously drive a project forward.

**Key gaps AURA fills:**

| Gap | AURA Solution |
|-----|--------------|
| AI needs hand-holding for each step | Autonomous 10-phase pipeline with LLM + sandbox + verify |
| No memory of past decisions | 5-layer memory with SQLite brain, vector store, context graph |
| AI can't run and validate its own code | Isolated subprocess sandbox with 3x retry and stderr fix_hints |
| Unsafe multi-file edits | AtomicChangeSet — all-or-nothing filesystem writes |
| No skill specialization | 28 pluggable skills dispatched in parallel per goal type |
| No self-improvement | ReflectorAgent + DeepReflectionLoop + WeaknessRemediator |
| Can't block dangerous changes | HumanGate intercepts security/coverage regressions |
| Fragile on flaky LLMs | EMA-weighted RouterAgent + response cache preloading |

---

## 3. Product Vision

> **"A tireless software engineer that improves itself while improving your codebase — goal by goal, cycle by cycle."**

AURA's north star is **fully autonomous, safe, continuously improving software development** at the project scale. The system should be able to:

1. Accept a backlog of natural language goals
2. Prioritize, plan, code, test, and apply changes autonomously
3. Learn from failures and weaknesses, generating remediation sub-goals
4. Expose all capabilities as composable MCP tools to the broader AI ecosystem
5. Detect convergence failure (oscillation) and escape loops proactively
6. Eventually self-host: AURA improves its own codebase via the agentic loop workflow

---

## 4. User Personas

### 4.1 Solo Developer (Primary)
- Runs AURA locally on laptop or Android/Termux
- Queues goals via CLI, checks back on completed work
- Needs: reliable goal execution, clear logs, safe apply

### 4.2 Open Source Maintainer
- Uses GitHub Actions workflows to run AURA on PRs and issues
- Labels issues `aura-goal` to trigger automatic queuing
- Needs: CI integration, Copilot + Gemini review, safe PR automation

### 4.3 DevOps / Platform Engineer
- Deploys AURA as a service with HTTP API
- Integrates with MCP-compatible tools (Claude Desktop, VS Code, etc.)
- Needs: MCP servers, structured logging, health monitoring, Docker support

### 4.4 AI/LLM Researcher
- Studies AURA's self-improvement loop and agentic workflow patterns
- Interested in skills system, context graph, propagation engine
- Needs: clean abstractions, documented APIs, test coverage

---

## 5. Core Concepts & Terminology

| Term | Definition |
|------|-----------|
| **Goal** | A natural language task string queued for autonomous execution |
| **Cycle** | One complete pass through the 10-phase pipeline for a single goal |
| **Skill** | A focused analysis module (e.g., `security_scanner`, `linter_enforcer`) implementing `SkillBase` ABC |
| **Agent** | An LLM-backed or rule-based worker implementing `AgentBase` ABC |
| **Brain** | SQLite-backed persistent memory with vector search and weakness tracking (L2) |
| **ContextGraph** | SQLite property graph linking files, goals, skills, and weaknesses (L2) |
| **AtomicChangeSet** | Filesystem transaction: all-or-nothing write of N files with rollback |
| **HumanGate** | Pre-apply blocking gate that intercepts security/coverage regressions |
| **OscillationDetector** | Detects when the loop is cycling on the same failure pattern |
| **Sandbox** | Isolated subprocess that executes generated code with timeout + 3x retry |
| **AURA_TARGET** | Directive comment in LLM-generated code marking the apply insertion point |
| **MCP Server** | Model Context Protocol HTTP server exposing AURA tools to LLM clients |
| **WeaknessRemediator** | Generates fix-goals from Brain-recorded weakness patterns |
| **DeepReflectionLoop** | Reflection triggered every 5 cycles for deeper pattern analysis |
| **SkillChainer** | Chains security scan → auto-remediation goal creation |
| **RouterAgent** | EMA-weighted model selector routing goals to optimal LLM providers |
| **GoalQueue** | Persisted priority queue of pending natural language goals |
| **TaskHierarchy** | Hierarchical sub-task decomposition tree for complex goals |

---

## 6. System Architecture

### 6.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS (3)                             │
│  main.py → cli/cli_main.py    aura_cli/cli_main.py    HTTP Server  │
└────────────────────┬────────────────────────────────────────────────┘
                     │ create_runtime()
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RUNTIME WIRING                                   │
│  GoalQueue → ConfigManager → Brain → ModelAdapter                  │
│  → VectorStore → RouterAgent → DebuggerAgent → PlannerAgent        │
│  → default_agents() → LoopOrchestrator → GitTools → MemoryStore    │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│             LOOP ORCHESTRATOR (core/orchestrator.py, 744L)         │
│  Phase 1: Ingest → 2: Skills → 3: Plan → 4: Critique → 5: Synth   │
│  Phase 6: Act  → 7: Sandbox → 8: Apply → 9: Verify → 10: Reflect  │
└────────────────────────────────────────────────────────────────────┘
         │            │              │              │
         ▼            ▼              ▼              ▼
    AGENTS (16)   SKILLS (28)   MCP SERVERS (5)  MEMORY (5 layers)
```

### 6.2 Entry Points & CLI Systems

> ⚠️ **Architectural Issue:** Three parallel CLI systems exist with overlapping functionality (see §16).

| File | Lines | Role |
|------|-------|------|
| `main.py` | 11 | Thin wrapper → delegates to `cli/cli_main.py` |
| `aura_cli/cli_main.py` | 465 | **Primary runtime** — full arg parsing, `create_runtime()` |
| `cli/cli_main.py` | 129 | Older/partial duplicate, called by main.py |
| `aura-cli/main.py` | 2 | Abandoned stub |

**Canonical entry point:** `aura_cli/cli_main.py` is the primary runtime with full argument parsing and `create_runtime()` wiring. All new features should target this module.

### 6.3 Dependency Hubs

Most-imported internal modules (import frequency):

| Module | Imports | Role |
|--------|---------|------|
| `core.logging_utils` | 85 | Universal structured JSON logging |
| `agents.skills.base` | 29 | `SkillBase` ABC for all 28 skills |
| `core.file_tools` | 17 | `safe_apply`, `AtomicChangeSet` |
| `memory.store` | 10 | `MemoryStore` JSON tier persistence |
| `core.git_tools` | 9 | Git operations wrapper |
| `core.model_adapter` | 9 | LLM provider abstraction |
| `memory.brain` | 8 | SQLite brain + vector store |
| `core.orchestrator` | 6 | `LoopOrchestrator` main loop |
| `core.hybrid_loop` | 6 | Legacy loop (migration pending) |

### 6.4 Leaf / Foundation Modules

Zero local imports — pure foundation layer:

- `agents/applicator.py` — patch application leaf
- `agents/base.py` — `AgentBase` ABC
- `core/logging_utils.py` — JSON structured logging
- `core/schema.py` — shared Pydantic/dataclass schemas
- `core/exceptions.py` — canonical exception hierarchy
- `core/prompts.py` — LLM prompt templates
- `core/types.py` — shared type aliases
- `memory/store.py` — JSON file persistence
- `tools/mcp_server.py` — dev tools MCP server

---

## 7. Feature Specifications

### 7.1 CLI Interface

**Module:** `aura_cli/cli_main.py` (465L), `aura_cli/commands.py`

#### Commands

| Command | Description |
|---------|-------------|
| `aura run <goal>` | Queue and immediately execute a single goal |
| `aura queue add <goal>` | Add goal to persistent GoalQueue |
| `aura queue list` | Display all pending goals with priorities |
| `aura queue clear` | Remove all pending goals |
| `aura loop` | Start autonomous loop — process queue continuously |
| `aura status` | Show current loop state, last cycle summary |
| `aura skills list` | List all 28 registered skills |
| `aura skills run <name> <goal>` | Execute a single skill directly |
| `aura memory show` | Dump recent Brain memories |
| `aura memory clear` | Reset Brain state |
| `aura reflect` | Trigger manual deep reflection cycle |
| `aura config show` | Display effective configuration |
| `aura doctor` | Run `aura_doctor.py` diagnostics |
| `aura server start` | Launch HTTP API server |
| `aura mcp start` | Launch all MCP servers |

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max-cycles N` | 5 | SlidingWindowPolicy limit |
| `--timeout N` | 3600 | TimeBoundPolicy seconds |
| `--model <name>` | config | Override model for this run |
| `--dry-run` | false | Plan only, skip Apply/Verify |
| `--no-sandbox` | false | Skip sandbox phase (unsafe) |
| `--human-gate` | true | Enable HumanGate blocking |
| `--log-level` | INFO | Logging verbosity |
| `--output json\|text` | text | Output format |

### 7.2 HTTP API Server

**Module:** `aura_cli/server.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/goals` | GET | List goal queue |
| `/goals` | POST | Add goal `{"goal": "..."}` |
| `/goals/{id}` | DELETE | Remove goal |
| `/loop/start` | POST | Start autonomous loop |
| `/loop/stop` | POST | Stop loop gracefully |
| `/loop/status` | GET | Current loop state + last cycle |
| `/skills` | GET | List all registered skills |
| `/skills/{name}/run` | POST | Execute skill `{"goal": "..."}` |
| `/metrics` | GET | Per-skill SkillMetrics JSON |
| `/memory/brain` | GET | Recent brain memories |
| `/memory/graph` | GET | Context graph summary |
| `/reflect` | POST | Trigger manual reflection |
| `/config` | GET | Current effective config |

### 7.3 Autonomous Loop Pipeline (10 Phases)

**Module:** `core/orchestrator.py` (744L, `LoopOrchestrator`)

The pipeline executes sequentially for each goal cycle. Each phase has defined inputs, outputs, and failure behaviors.

#### Phase 1: Ingest
- **Agent:** `IngestAgent` (`agents/ingest.py`)
- **Input:** Goal string + project file tree
- **Output:** Project snapshot dict + memory hints from Brain recall
- **Action:** Walks project files, filters by `.gitignore`, recalls relevant memories via `Brain.recall_with_budget()`

#### Phase 2: Skill Dispatch
- **Module:** `core/skill_dispatcher.py`
- **Input:** Goal string + project snapshot
- **Output:** Skill analysis results dict (parallel execution)
- **Action:**
  - Classifies goal type via `classify_goal_llm()` (cached, 1 LLM call per unique goal type)
  - Selects relevant subset of 28 skills via `AdaptiveStrategySelector`
  - Dispatches selected skills in parallel threads
  - Records per-skill `SkillMetrics` (duration, success, token cost)
  - `SkillChainer` triggers remediation goals for security findings

#### Phase 3: Plan
- **Agent:** `PlannerAdapter` → `PlannerAgent` (`agents/planner.py`)
- **Input:** Goal + snapshot + skill results + weakness context from Brain
- **Output:** Ordered list of plan steps
- **Action:** LLM prompt with weakness context; retries up to 3 times on parse failure

#### Phase 4: Critique
- **Agent:** `CriticAdapter` → `CriticAgent` (`agents/critic.py`)
- **Input:** Plan steps + goal + snapshot
- **Output:** Adversarial review dict with identified weaknesses
- **Action:** LLM adversarial review; feeds weaknesses back to Brain for future cycles

#### Phase 5: Synthesize
- **Agent:** `SynthesizerAgent` (`agents/synthesizer.py`)
- **Input:** Plan steps + critique results
- **Output:** `task_bundle` — unified action specification
- **Action:** Merges plan and critique into coherent task bundle; resolves conflicts

#### Phase 6: Act
- **Agent:** `ActAdapter` → `CoderAgent` (`agents/coder.py`)
- **Input:** `task_bundle` + project snapshot
- **Output:** Generated code string with `# AURA_TARGET:` directives
- **Action:** LLM code generation; `MAX_ITERATIONS` loop for quality; embeds target location markers

#### Phase 7: Sandbox
- **Agent:** `SandboxAdapter` → `SandboxAgent` (`agents/sandbox.py`)
- **Input:** Generated code
- **Output:** Execution result (stdout, stderr, exit_code) or fix_hints
- **Action:**
  - Executes code in isolated subprocess with timeout
  - **3x retry loop**: on failure, extracts `fix_hints` from stderr, re-sends to CoderAgent
  - Passes if exit_code == 0 or no critical errors

#### Phase 8: Apply
- **Module:** `core/file_tools.py` → `_safe_apply_change` / `AtomicChangeSet`
- **Input:** Validated code + target file paths
- **Output:** Applied filesystem state or rollback
- **Action:**
  - **HumanGate check**: blocks apply if security or coverage regression detected
  - `AtomicChangeSet`: stages all file writes, commits all-or-nothing
  - `OldCodeNotFoundError` triggers fuzzy git history search via `find_historical_match()`
  - On any write failure: full rollback, error logged to Brain

#### Phase 9: Verify
- **Agent:** `VerifierAgent` (`agents/verifier.py`)
- **Input:** Applied filesystem state
- **Output:** Verification score (0.0–1.0) + test results
- **Action:** Runs `pytest` + linting; parses results; computes quality score; stores in Brain

#### Phase 10: Reflect
- **Agent:** `ReflectorAgent` (`agents/reflector.py`)
- **Input:** Full cycle results (plan, code, verify score, errors)
- **Output:** Cycle summary → Brain memory entry
- **Action:**
  - Summarizes cycle in natural language
  - Records to `Brain` + `MemoryStore.cycle_summaries`
  - Appends to `decision_log.jsonl`
  - Every 5th cycle: triggers `DeepReflectionLoop`
  - `WeaknessRemediator` converts recorded weaknesses to new goals

### 7.4 Agent System (16 Agents)

**Base:** `agents/base.py` — `AgentBase` ABC with error-swallowing `run()` (exceptions logged, never propagated)

| Agent | Module | Lines | Role |
|-------|--------|-------|------|
| `IngestAgent` | `agents/ingest.py` | — | Project file walker + memory recall |
| `PlannerAgent` | `agents/planner.py` | — | LLM step generation with weakness ctx |
| `CriticAgent` | `agents/critic.py` | — | Adversarial plan review |
| `SynthesizerAgent` | `agents/synthesizer.py` | — | Merges plan + critique → task_bundle |
| `CoderAgent` | `agents/coder.py` | — | LLM code gen with AURA_TARGET directive |
| `SandboxAgent` | `agents/sandbox.py` | — | Isolated subprocess + 3x retry |
| `VerifierAgent` | `agents/verifier.py` | — | pytest + lint runner + score |
| `ReflectorAgent` | `agents/reflector.py` | — | Cycle summary → Brain memory |
| `DebuggerAgent` | `agents/debugger.py` | — | LLM error diagnosis |
| `RouterAgent` | `agents/router.py` | 238L, 2 cls | EMA-weighted model routing |
| `MutatorAgent` | `agents/mutator.py` | — | Code mutation/patching |
| `ScaffolderAgent` | `agents/scaffolder.py` | — | New file scaffolding |
| `TesterAgent` | `agents/tester.py` | — | Automated test generation |
| `ContextManagerAgent` | `agents/context_manager.py` | — | Context window management |
| `Applicator` | `agents/applicator.py` | — | Leaf: applies diffs/patches |
| *(Adapters)* | `agents/registry.py` | 463L | `PlannerAdapter`, `CriticAdapter`, `ActAdapter`, `SandboxAdapter`, `default_agents()` |

#### RouterAgent Detail
`agents/router.py` (238L, 2 classes):
- Maintains per-model EMA (Exponential Moving Average) success scores
- Routes goals to OpenRouter / OpenAI / Gemini / Local based on current scores
- Demotes failing models automatically; promotes recovering models
- Exposes routing statistics via `/metrics` endpoint

### 7.5 Skills System (28 Skills)

**Base:** `agents/skills/base.py` — `SkillBase` ABC (29 imports = most-used after logging)  
**Dispatcher:** `core/skill_dispatcher.py`  
**Registry:** All 28 skills self-register on import

#### Original 23 Skills (v1.0)

| Skill | Key Analysis |
|-------|-------------|
| `dependency_analyzer` | Import graph, unused deps, version conflicts |
| `architecture_validator` | Layer violations, **circular import detection** (new) |
| `complexity_scorer` | Cyclomatic + cognitive complexity per function |
| `test_coverage_analyzer` | Coverage % + **incremental from git diff** (new) |
| `doc_generator` | Missing docstrings, stale docs |
| `performance_profiler` | Hot paths, O(n²) patterns, memory leaks |
| `refactoring_advisor` | Extract method, rename, SOLID violations |
| `schema_validator` | JSON/Pydantic schema conformance |
| `security_scanner` | Hardcoded secrets, injection risks, CVEs |
| `type_checker` | mypy-equivalent type error detection |
| `linter_enforcer` | PEP8, flake8, black conformance |
| `incremental_differ` | Git diff → minimal change surface |
| `tech_debt_quantifier` | TODO/FIXME counts, dead code, debt score |
| `api_contract_validator` | REST contract conformance, breaking changes |
| `generation_quality_checker` | LLM output quality scoring |
| `git_history_analyzer` | Commit patterns, hot files, churn |
| `skill_composer` | Combines multiple skill outputs |
| `error_pattern_matcher` | Recurring error pattern detection |
| `code_clone_detector` | Duplicate code blocks (CPD-equivalent) |
| `adaptive_strategy_selector` | Selects skill subset per goal classification |
| `web_fetcher` | Fetches URLs for context (docs, issues) |
| `symbol_indexer` | AST-level symbol table builder |
| `multi_file_editor` | Coordinates edits across N files |

#### 5 New Skills (v2.0 additions)

| Skill | Module | Analysis |
|-------|--------|---------|
| `dockerfile_analyzer` | `agents/skills/dockerfile_analyzer.py` | Image security, layer optimization, best practices |
| `observability_checker` | `agents/skills/observability_checker.py` | Missing logging, metrics, tracing instrumentation |
| `changelog_generator` | `agents/skills/changelog_generator.py` | Auto-generates CHANGELOG from git history |
| `database_query_analyzer` | `agents/skills/database_query_analyzer.py` | SQL query analysis, N+1 detection, index suggestions |
| `skill_failure_analyzer` | `agents/skills/skill_failure_analyzer.py` | Meta-skill: analyzes failed skill runs for patterns |

#### Skill Dispatch Pipeline

```
goal_string
    │
    ▼
classify_goal_llm() ──cache──► goal_type (cached per unique goal type)
    │
    ▼
AdaptiveStrategySelector.select(goal_type, history)
    │
    ▼
parallel_dispatch(selected_skills, goal, snapshot)
    │
    ├── SkillMetrics.record(skill_name, duration, tokens, success)
    │
    └── SkillChainer.check(results) ──security_findings──► new remediation goals
```

#### Per-Skill Metrics (`/metrics` endpoint)

Each skill records:
- `total_runs`, `success_count`, `failure_count`
- `avg_duration_ms`, `p95_duration_ms`
- `total_tokens_used`
- `last_error` string

### 7.6 Memory Architecture (5 Layers)

AURA implements a five-layer memory hierarchy from hot in-process cache to cold JSON files on disk.

```
L0  ┌──────────────────────────────────────────────────────┐
    │  In-Memory Dict Cache                                │
    │  model_adapter._mem_cache (preloaded from DB)        │
    │  Scope: process lifetime | TTL: session              │
L1  ├──────────────────────────────────────────────────────┤
    │  LocalCacheAdapter (memory/local_cache.db)           │
    │  SQLite, TTL-based, pub/sub events                   │
    │  OR MomentoAdapter (cloud) if MOMENTO_API_KEY set    │
    │  Scope: persistent across restarts | TTL: config     │
L2  ├──────────────────────────────────────────────────────┤
    │  Brain (memory/brain_v2.db SQLite)                   │
    │  Tables: memories, weaknesses, vectors, response_cache│
    │  + VectorStore (brain's vector_store_data table)     │
    │  + ContextGraph (memory/context_graph.db SQLite)     │
    │    Property graph: files ↔ goals ↔ skills ↔ weaknesses│
    │  Scope: persistent | Search: vector + keyword        │
L3  ├──────────────────────────────────────────────────────┤
    │  MemoryStore (memory/store/ JSON files)              │
    │  Files: cycle_summaries/, decision_log.jsonl         │
    │         strategy_stats.json                          │
    │  Scope: persistent append-only log                   │
L4  └──────────────────────────────────────────────────────┘
    │  GoalQueue (memory/goal_queue.json)                  │
    │  GoalArchive (memory/goal_archive.json)              │
    │  TaskHierarchy (memory/task_hierarchy_v2.json)       │
    │  StrategyStats (memory/strategy_stats.json)          │
    │  Scope: persistent operational state                 │
```

#### Brain (`memory/brain.py`, `brain_v2.db`)

| Table | Purpose |
|-------|---------|
| `memories` | Cycle summaries, decisions, insights |
| `weaknesses` | Identified code/agent weaknesses with frequency |
| `vectors` | Embedding vectors for semantic recall |
| `response_cache` | Cached LLM responses (deduplication) |
| `vector_store_data` | VectorStore backing (used by Brain.recall()) |

Key operations:
- `Brain.recall_with_budget(goal, max_tokens)` — token-budget-aware semantic recall (compression: 10k entries → 179 kept in 0.32ms)
- `Brain.record_weakness(pattern, context)` — persists weakness for WeaknessRemediator
- `Brain.preload_cache()` — warms L0 in-memory cache on startup

#### ContextGraph (`core/context_graph.py`, 285L, `context_graph.db`)

SQLite-backed property knowledge graph:
- **Nodes:** files, goals, skills, weaknesses, agents
- **Edges:** typed relationships (applies_to, depends_on, causes, fixes)
- **Queries:** `graph.get_related(node, relation)`, `graph.find_path(src, dst)`
- Used by: IngestAgent, ReflectorAgent, AutonomousDiscovery

#### Memory Compaction (`core/memory_compaction.py`)

- Triggered when Brain exceeds configured memory threshold
- Summarizes old memory entries via LLM → replaces N entries with 1 summary
- Preserves weakness records indefinitely

### 7.7 Advanced Orchestration Modules

These modules extend the core pipeline with higher-order capabilities.

#### WorkflowEngine (`core/workflow_engine.py`, 879L, 8 classes, 33 functions)
Reusable agentic workflow execution framework:
- Defines `Workflow`, `WorkflowStep`, `WorkflowContext`, `WorkflowResult` types
- Supports branching, parallel steps, conditional execution
- Used by: `agentic_loop_mcp.py` MCP server

#### AdaptivePipeline (`core/adaptive_pipeline.py`, 228L)
Context-aware phase selection:
- Analyzes goal complexity + historical success rates
- Selects subset of pipeline phases to skip for simple goals
- Tracks phase-level metrics per goal type

#### AutonomousDiscovery (`core/autonomous_discovery.py`, 246L)
Self-propagating work discovery:
- Scans codebase for signals: TODO comments, failing tests, complexity hotspots
- Generates new goals autonomously without human input
- Feeds discovered goals to GoalQueue

#### PropagationEngine (`core/propagation_engine.py`, 315L)
Forward-chaining event→rule→action system:
- Rules: `IF event MATCHES pattern THEN trigger action`
- Events: skill findings, verify failures, weakness detections
- Actions: queue new goals, trigger specific agents, send notifications

#### ConvergenceEscapeLoop (`core/convergence_escape.py`)
- `OscillationDetector`: detects when loop repeats same failure pattern
  - Benchmarked: 1000 events in 0.65ms
- `ConvergenceEscapeLoop`: on oscillation detection, injects goal diversification strategy

#### HealthMonitor (`core/health_monitor.py`, 194L)
Periodic quality drift detection:
- Runs on configurable interval (default: every 10 cycles)
- Checks: test pass rate trend, skill success rate, code complexity growth
- Alerts via logging + generates remediation goals on drift

#### EvolutionLoop (`core/evolution_loop.py`, 144L)
Iterative improvement loop:
- Wraps the main orchestrator with evolutionary selection
- Maintains N candidate code variants, selects best by verify score

#### GoalDecomposer (`core/goal_decomposer.py`)
Hierarchical sub-task decomposition:
- Breaks complex goals into ordered sub-goals
- Stores decomposition tree in `TaskHierarchy` (L4 memory)
- Sub-goals are independently queued and executed

#### WeaknessRemediator (`core/weakness_remediator.py`)
- Reads recorded weaknesses from Brain
- Generates specific fix-goals per weakness pattern
- Priority-queues remediation goals ahead of new user goals

#### DeepReflectionLoop (`core/reflection_loop.py`)
- Triggered every 5 main cycles by ReflectorAgent
- Performs cross-cycle pattern analysis
- Updates StrategyStats and model routing weights

### 7.8 MCP Server Ecosystem (5 Servers)

AURA exposes capabilities as Model Context Protocol HTTP servers.

> ⚠️ **Known Issue:** Port assignments lack a central registry (§16.8).

#### Server 1: Dev Tools MCP (`tools/mcp_server.py`)
- **Port:** 8001 (compatible)
- **Tools:** 28 developer tools
- **Categories:** File I/O, shell execution, git operations, linting
- **Leaf module:** zero local imports (foundation layer)

| Tool Group | Tools |
|-----------|-------|
| File I/O | `read_file`, `write_file`, `list_dir`, `find_files`, `delete_file` |
| Shell | `run_command`, `run_script`, `get_env` |
| Git | `git_status`, `git_diff`, `git_commit`, `git_log`, `git_branch` |
| Lint | `run_pytest`, `run_flake8`, `run_mypy`, `run_black` |

#### Server 2: Skills MCP (`tools/aura_mcp_skills_server.py`, 377L)
- **Port:** 8002
- **Tools:** All 28 skills exposed as HTTP tools
- **Interface:** `POST /skills/{name}` with `{"goal": "...", "context": {...}}`
- **Response:** Skill analysis result JSON
- **Duplicate:** Contains `CallRequest` class (also in `mcp_server.py` — §16.4)

#### Server 3: Control Plane MCP (`tools/aura_control_mcp.py`, 470L)
- **Port:** TBD (not yet registered)
- **Tools:** AURA control plane operations
- **Operations:** Start/stop loop, manage goal queue, trigger reflect, read brain state
- **Audience:** External orchestrators, Claude Desktop, VS Code extensions

#### Server 4: Agentic Loop MCP (`tools/agentic_loop_mcp.py`, 609L)
- **Port:** 8006
- **Tools:** WorkflowEngine + LoopOrchestrator as MCP tools
- **Operations:** Submit goals, run workflows, stream cycle events
- **Backed by:** `core/workflow_engine.py`

#### Server 5: GitHub Copilot MCP (`tools/github_copilot_mcp.py`, 791L)
- **Port:** TBD
- **Tools:** GitHub Copilot integration operations
- **Operations:** PR review, issue → goal conversion, code suggestion application
- **Integrations:** Copilot API, GitHub REST API

#### Bonus: Sequential Thinking MCP (`tools/sequential_thinking_mcp.py`)
- Sequential reasoning chain tool for multi-step problem solving
- Integrates with the PlannerAgent reasoning process

### 7.9 Configuration System

**Module:** `core/config_manager.py`  
**File:** `aura.config.json`

```json
{
  "model": {
    "primary": "openrouter/auto",
    "fallback": "openai/gpt-4o-mini",
    "local": "ollama/codellama",
    "router": "ema_weighted"
  },
  "loop": {
    "max_cycles": 5,
    "timeout_seconds": 3600,
    "sandbox_retries": 3,
    "reflection_interval": 5
  },
  "memory": {
    "brain_db": "memory/brain_v2.db",
    "context_graph_db": "memory/context_graph.db",
    "cache_db": "memory/local_cache.db",
    "memory_compaction_threshold": 10000,
    "goal_queue": "memory/goal_queue.json"
  },
  "skills": {
    "parallel_dispatch": true,
    "max_parallel": 8,
    "timeout_per_skill_seconds": 30
  },
  "human_gate": {
    "enabled": true,
    "block_on_security": true,
    "block_on_coverage_regression": true
  },
  "api_keys": {
    "openrouter": "${OPENROUTER_API_KEY}",
    "openai": "${OPENAI_API_KEY}",
    "gemini": "${GEMINI_API_KEY}",
    "momento": "${MOMENTO_API_KEY}"
  }
}
```

**ConfigManager features:**
- Environment variable substitution (`${VAR}` syntax)
- Layered overrides: defaults → file → env vars → CLI flags
- Hot-reload on SIGHUP

### 7.10 GitHub Automation Workflows

**Location:** `.github/workflows/`

| Workflow | File | Trigger | Jobs |
|----------|------|---------|------|
| CI | `ci.yml` | push, PR | Python 3.10/3.11 matrix test |
| Agentic Loop | `aura-agentic-loop.yml` | daily 02:00 UTC + manual | loop, reflect, skill-improve (3 jobs) |
| Copilot Autofix | `copilot-autofix.yml` | PR | Copilot lint+test PR review |
| Copilot Workspace | `copilot-workspace.yml` | issue label `aura-goal` | queue goal → aura loop |
| Gemini Code Assist | `gemini-code-assist.yml` | PR | Gemini PR review with AURA skills |
| Coder Automation | `coder-automation.yml` | manual | Codespaces devenv validation |

#### Agentic Loop Workflow (`aura-agentic-loop.yml`) — Detail

```
Job 1: loop
  - checkout, setup Python
  - pip install -r requirements.txt
  - aura loop --max-cycles 10
  - upload artifacts (cycle_summaries, logs)

Job 2: reflect (depends: loop)
  - aura reflect --deep
  - push memory updates

Job 3: skill-improve (depends: reflect)
  - aura skills run skill_failure_analyzer "improve failing skills"
  - commit any skill updates
```

### 7.11 Stopping Policies

**Module:** `core/policies/`

| Policy | Class | Parameter | Behavior |
|--------|-------|-----------|---------|
| Sliding Window | `SlidingWindowPolicy` | `max_cycles=5` | Stop after N complete cycles |
| Time Bound | `TimeBoundPolicy` | `max_seconds=3600` | Stop after N wall-clock seconds |
| Resource Bound | `ResourceBoundPolicy` | `max_tokens=100000` | Stop after N total tokens used |

Policies are composable — loop stops when **any** policy triggers. Each policy emits a structured log entry explaining the stop reason.

### 7.12 Self-Improvement Loops

AURA contains multiple nested self-improvement mechanisms:

```
Main Loop (LoopOrchestrator)
  │
  ├── Every cycle: ReflectorAgent → Brain.record()
  │
  ├── Every 5 cycles: DeepReflectionLoop
  │     └── Cross-cycle pattern analysis
  │         └── Update StrategyStats + RouterAgent weights
  │
  ├── Every cycle: WeaknessRemediator
  │     └── Read Brain weaknesses → queue fix-goals
  │
  ├── Continuous: OscillationDetector
  │     └── On oscillation: ConvergenceEscapeLoop → diversify
  │
  ├── Periodic: HealthMonitor
  │     └── On drift: generate health remediation goals
  │
  └── Continuous: AutonomousDiscovery
        └── Scan codebase → queue discovered goals
```

---

## 8. New Features Since v1.0 PRD (15 Implemented)

All 15 roadmap items from PRD v1.0 are **fully implemented** as of commit `5c9f90f`.

| # | Feature | Module | Status |
|---|---------|--------|--------|
| 1 | **AtomicChangeSet** + `apply_atomic()` | `core/file_tools.py` | ✅ Done |
| 2 | **Sandbox retry loop 3x** with stderr `fix_hints` | `core/orchestrator.py` | ✅ Done |
| 3 | **Token budget compression** | `memory/brain.py::recall_with_budget()` | ✅ Done |
| 4 | **Circular import detection** | `agents/skills/architecture_validator.py` | ✅ Done |
| 5 | **Per-skill metrics** (`SkillMetrics` + `/metrics` endpoint) | `core/skill_dispatcher.py` | ✅ Done |
| 6 | **Incremental test coverage** from git diff | `agents/skills/test_coverage_analyzer.py` | ✅ Done |
| 7 | **LLM goal type classifier** (`classify_goal_llm` + cache) | `core/skill_dispatcher.py` | ✅ Done |
| 8 | **`OldCodeNotFoundError` git fuzzy recovery** | `core/file_tools.py::find_historical_match()` | ✅ Done |
| 9 | **E2E integration tests** | `tests/integration/test_orchestrator_e2e.py` (5 tests) | ✅ Done |
| 10 | **HumanGate** (security/coverage blocking) | `core/human_gate.py` | ✅ Done |
| 11 | **SkillChainer** (security→remediation goals) | `core/skill_dispatcher.py` | ✅ Done |
| 12 | **Response cache preloading** | `core/model_adapter.py::preload_cache()` | ✅ Done |
| 13 | **`skill_failure_analyzer` skill** | `agents/skills/skill_failure_analyzer.py` | ✅ Done |
| 14 | **OscillationDetector** | `core/convergence_escape.py` | ✅ Done |
| 15 | **Adaptive context window sizing** | `core/model_adapter.py::estimate_context_budget()` | ✅ Done |

---

## 9. Non-Functional Requirements

### 9.1 Performance

| Requirement | Target | Measured |
|-------------|--------|---------|
| Single orchestrator cycle (mock LLM) | < 1s | ~640ms ✅ |
| Cache miss (10k entries) | < 100ms | 36ms ✅ |
| Cache hit (1k entries) | < 50ms | 15ms ✅ |
| Token compression (10k → 179) | < 5ms | 0.32ms ✅ |
| AtomicChangeSet (10 files) | < 200ms | 51ms ✅ |
| AtomicChangeSet (50 files) | < 500ms | 248ms ✅ |
| OscillationDetector (1000 events) | < 5ms | 0.65ms ✅ |
| SkillMetrics (10k records) | < 500ms | 131ms ✅ |
| HumanGate decision (10k calls) | < 100ms | 35ms ✅ |
| Keyword classifier (1000 calls) | < 200ms | 44ms ✅ |
| LLM classifier cache (50 calls) | 1 LLM call | 1 call ✅ |

### 9.2 Reliability
- All orchestrator phases: exceptions caught, logged, never propagate to crash loop
- `AgentBase.run()` error-swallowing pattern across all 16 agents
- Sandbox 3x retry before marking goal as failed
- AtomicChangeSet full rollback on partial write failure
- HumanGate prevents unsafe changes from reaching filesystem

### 9.3 Portability
- Target platform: Python 3.10+ on Linux (primary), macOS, Android/Termux
- No GUI dependencies; all CLI/HTTP
- Docker support via `Dockerfile`
- SQLite for all persistence (no external database required by default)

### 9.4 Observability
- Structured JSON logging via `core/logging_utils.py` (85 imports = universal)
- Per-skill `SkillMetrics` via `/metrics` HTTP endpoint
- Cycle summaries in `memory/store/cycle_summaries/`
- `decision_log.jsonl` for decision audit trail
- `HealthMonitor` periodic quality drift reports

### 9.5 Security
- `HumanGate` blocks security regressions from applying
- `security_scanner` skill runs on every cycle
- `SkillChainer` auto-queues remediation for findings
- No secrets in source code (env var substitution in config)
- Sandbox execution isolated from main process

---

## 10. Performance Benchmarks

*Source: `tests/test_performance_simulations.py` (48 tests, all passing), measured on Android/Termux*

### Memory & Caching

| Operation | N | Time |
|-----------|---|------|
| Cache miss lookups | 10,000 | 36ms total |
| Cache hit lookups | 1,000 | 15ms total |
| Cache writes | 5,000 | 104ms total |
| Token compression | 10,000 → 179 | 0.32ms |
| Response cache preload | baseline | < 1ms |

### File Operations

| Operation | N Files | Time |
|-----------|---------|------|
| AtomicChangeSet apply | 1 | 4ms |
| AtomicChangeSet apply | 10 | 51ms |
| AtomicChangeSet apply | 50 | 248ms |

### Agent & Skill Operations

| Operation | N | Time |
|-----------|---|------|
| OscillationDetector events | 1,000 | 0.65ms |
| SkillMetrics records (thread-safe) | 10,000 | 131ms (0.013ms each) |
| Keyword classifier calls | 1,000 | 44ms |
| LLM classifier unique types | 50 calls | 1 LLM call (cached) |
| HumanGate decisions | 10,000 | 35ms (0.004ms each) |

### End-to-End

| Scenario | Time |
|----------|------|
| Single orchestrator cycle (mock LLM) | ~640ms |
| Full 5-cycle loop (mock LLM) | ~3.2s |

---

## 11. Data Models

### Goal

```python
@dataclass
class Goal:
    id: str                    # UUID
    text: str                  # Natural language goal string
    priority: int              # 1 (highest) – 10 (lowest)
    status: str                # pending | in_progress | done | failed
    created_at: datetime
    parent_id: Optional[str]   # For sub-goals from GoalDecomposer
    metadata: Dict[str, Any]   # Arbitrary context
```

### TaskBundle (Phase 5 → Phase 6 output)

```python
@dataclass
class TaskBundle:
    goal: str
    plan_steps: List[str]
    critique_points: List[str]
    target_files: List[str]
    context_snapshot: Dict
    skill_results: Dict[str, Any]
    weakness_hints: List[str]
```

### CycleSummary (Phase 10 output)

```python
@dataclass
class CycleSummary:
    cycle_id: str
    goal: str
    phases_completed: List[str]
    verify_score: float         # 0.0 – 1.0
    files_changed: List[str]
    tests_passed: int
    tests_failed: int
    weaknesses_found: List[str]
    duration_seconds: float
    model_used: str
    tokens_used: int
    timestamp: datetime
```

### SkillResult

```python
@dataclass
class SkillResult:
    skill_name: str
    success: bool
    findings: List[Dict]        # Structured findings
    score: Optional[float]      # 0.0 – 1.0 where applicable
    duration_ms: float
    tokens_used: int
    error: Optional[str]
```

### Memory Entry (Brain)

```python
@dataclass
class MemoryEntry:
    id: str
    content: str               # Natural language summary
    embedding: Optional[List[float]]
    source: str                # cycle_id | skill_name | user
    importance: float          # 0.0 – 1.0
    created_at: datetime
    accessed_count: int
```

### Weakness Record (Brain)

```python
@dataclass
class WeaknessRecord:
    id: str
    pattern: str               # Weakness description
    context: str               # When/where it occurs
    frequency: int             # Times observed
    last_seen: datetime
    remediation_goal: Optional[str]  # Generated fix goal
```

---

## 12. API Reference

### REST API (aura_cli/server.py)

#### `POST /goals`
```json
Request:  {"goal": "Add type hints to all functions in core/", "priority": 3}
Response: {"id": "uuid-...", "status": "pending", "position": 4}
```

#### `GET /loop/status`
```json
{
  "running": true,
  "current_goal": "Add type hints...",
  "cycle": 3,
  "phase": "verify",
  "last_verify_score": 0.87,
  "goals_completed_today": 5
}
```

#### `GET /metrics`
```json
{
  "skills": {
    "security_scanner": {
      "total_runs": 42, "success_count": 40, "failure_count": 2,
      "avg_duration_ms": 230.5, "p95_duration_ms": 890.0,
      "total_tokens_used": 15420
    }
  },
  "router": {
    "openrouter/auto": {"ema_score": 0.91, "total_calls": 128},
    "openai/gpt-4o-mini": {"ema_score": 0.73, "total_calls": 22}
  }
}
```

### MCP Tool Interface

All MCP servers use the Model Context Protocol JSON-RPC format:

```json
Request:
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "security_scanner",
    "arguments": {"goal": "scan for hardcoded secrets", "context": {}}
  },
  "id": 1
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "content": [{"type": "text", "text": "{\"findings\": [...]}"}]
  },
  "id": 1
}
```

---

## 13. Security Requirements

### 13.1 HumanGate (`core/human_gate.py`)
- Intercepts Phase 8 (Apply) before any filesystem write
- **Blocking conditions:**
  - `security_scanner` finding with severity >= HIGH
  - Test coverage regression > configured threshold (default: 5%)
  - Explicit `BLOCK` directive in skill results
- **Action on block:** Logs detailed reason, skips apply, records to Brain, queues remediation goal
- **Performance:** 10k decisions in 35ms (0.004ms each)

### 13.2 API Keys
- All API keys via environment variables, never hardcoded
- `aura.config.json` uses `${VAR}` substitution syntax
- `.gitignore` excludes `.env` and `*_secrets*` files

### 13.3 Sandbox Isolation
- Generated code runs in subprocess with:
  - Configurable execution timeout
  - Captured stdout/stderr (never executed in AURA's process)
  - `OldCodeNotFoundError` cannot escape sandbox
- 3x retry does not loosen sandbox constraints

### 13.4 MCP Server Access
- MCP servers bind to `localhost` by default
- No authentication in v2.0 (planned: API key auth in roadmap)
- Port 8001/8002/8006 should be firewalled in production

---

## 14. Error Handling & Failure Modes

### 14.1 Phase-Level Failures

| Phase | Failure | Behavior |
|-------|---------|---------|
| Ingest | File read error | Log warning, continue with partial snapshot |
| Skill Dispatch | Skill exception | Log error, mark skill failed in SkillMetrics, continue |
| Plan | 3x LLM parse failure | Skip cycle, record weakness, queue retry |
| Critique | LLM error | Use empty critique, log warning |
| Synthesize | Merge conflict | Use plan-only task_bundle, log warning |
| Act | LLM error | Skip cycle, record to Brain |
| Sandbox | 3x subprocess failure | Skip apply, record fix_hints to Brain |
| Apply | Write failure | Full rollback via AtomicChangeSet |
| Apply | HumanGate block | Skip apply, queue remediation |
| Verify | pytest crash | Score = 0.0, record weakness |
| Reflect | LLM error | Use template summary, continue |

### 14.2 Known Error Classes

```python
# core/exceptions.py (canonical)
class AuraError(Exception): ...
class GitError(AuraError): ...           # ⚠️ Also in core/git_tools.py (duplicate)
class SandboxError(AuraError): ...
class ApplyError(AuraError): ...
class OldCodeNotFoundError(ApplyError): ...  # Triggers fuzzy git recovery
class HumanGateBlockError(AuraError): ...
class SkillError(AuraError): ...
class BrainError(AuraError): ...
class ConfigError(AuraError): ...
```

### 14.3 OscillationDetector & Convergence Escape
- Detects repeated identical failure patterns across N consecutive cycles
- On detection: `ConvergenceEscapeLoop` injects goal diversification
  - Changes model routing weights
  - Injects a "break pattern" sub-goal
  - Reduces max_cycles for this goal to prevent infinite loops

---

## 15. Testing Requirements & Coverage Gaps

### 15.1 Current Test State

| Metric | Value |
|--------|-------|
| Total test functions | 379 |
| Test files | 20 |
| **Tests passing** | **377** |
| Tests failing | 0 |
| Performance simulation tests | 48 (all passing) |
| E2E integration tests | 5 (all passing) |

### 15.2 Test File Inventory

| File | Tests | Coverage Area |
|------|-------|--------------|
| `tests/test_skills.py` | ~50 | Skills registry (all 28, surface-level) |
| `tests/test_performance_simulations.py` | 48 | Performance benchmarks (all modules) |
| `tests/integration/test_orchestrator_e2e.py` | 5 | Full pipeline (fake LLM/FS) |
| `tests/test_brain.py` | ~20 | Brain memory operations |
| `tests/test_file_tools.py` | ~25 | AtomicChangeSet, safe_apply |
| `tests/test_goal_queue.py` | ~15 | GoalQueue operations |
| `tests/test_router.py` | ~15 | RouterAgent EMA logic |
| `tests/test_human_gate.py` | ~12 | HumanGate blocking logic |
| `tests/test_convergence.py` | ~10 | OscillationDetector |
| `tests/test_skill_dispatcher.py` | ~20 | Skill dispatch + chainer |
| `test_aura_doctor_root.py` | ~10 | Doctor diagnostics |
| *(other test files)* | ~149 | Mixed |

### 15.3 Critical Coverage Gaps (75+ Untested Modules)

The following production modules have **no dedicated unit tests**:

#### Agents (no unit tests)
- `agents/coder.py` — CoderAgent LLM code generation
- `agents/planner.py` — PlannerAgent LLM step generation
- `agents/critic.py` — CriticAgent adversarial review
- `agents/synthesizer.py` — SynthesizerAgent merge logic
- `agents/verifier.py` — VerifierAgent pytest runner
- `agents/reflector.py` — ReflectorAgent summary generation
- `agents/debugger.py` — DebuggerAgent error diagnosis
- `agents/ingest.py` — IngestAgent file walker
- `agents/mutator.py` — MutatorAgent code patching
- `agents/scaffolder.py` — ScaffolderAgent new file creation
- `agents/tester.py` — TesterAgent test generation
- `agents/context_manager.py` — Context window management
- `agents/registry.py` — Adapter wrappers + default_agents()

#### CLI & Server (no unit tests)
- `aura_cli/server.py` — HTTP API server
- `aura_cli/cli_main.py` — Primary CLI (465L)
- `aura_cli/commands.py` — Command implementations

#### Core Orchestration (no unit tests)
- `core/orchestrator.py` (744L) — Only covered by E2E fakes
- `core/config_manager.py` — Configuration loading/override
- `core/adaptive_pipeline.py` (228L) — Phase selection logic

#### Advanced Modules (no unit tests)
- `core/workflow_engine.py` (879L) — Entire workflow engine
- `core/autonomous_discovery.py` (246L) — Goal discovery
- `core/context_graph.py` (285L) — Property graph operations
- `core/propagation_engine.py` (315L) — Forward-chaining rules
- `core/health_monitor.py` (194L) — Quality drift detection
- `core/evolution_loop.py` (144L) — Evolutionary improvement
- `core/reflection_loop.py` — DeepReflectionLoop
- `core/weakness_remediator.py` — Weakness → goal generation
- `core/memory_compaction.py` — Memory compression
- `core/goal_decomposer.py` — Hierarchical decomposition

#### MCP Servers (no unit tests)
- `tools/aura_mcp_skills_server.py` (377L)
- `tools/agentic_loop_mcp.py` (609L)
- `tools/aura_control_mcp.py` (470L)
- `tools/github_copilot_mcp.py` (791L)
- `tools/sequential_thinking_mcp.py`

#### Skills (registry-only, no per-skill unit tests)
All 28 individual skill implementations are tested only via `test_skills.py` registry check (confirms they load), not for correctness of analysis output.

### 15.4 Testing Roadmap Priority

| Priority | Target | Type | Effort |
|----------|--------|------|--------|
| P0 | `core/orchestrator.py` | Unit (mock LLM) | High |
| P0 | `agents/coder.py`, `agents/planner.py` | Unit (mock LLM) | Medium |
| P1 | `core/workflow_engine.py` | Unit | High |
| P1 | `core/context_graph.py` | Unit | Medium |
| P1 | `aura_cli/server.py` | Integration | Medium |
| P2 | All 28 skills individually | Unit per skill | High (bulk) |
| P2 | `core/adaptive_pipeline.py` | Unit | Medium |
| P3 | MCP servers | Integration | High |
| P3 | `core/autonomous_discovery.py` | Unit | Medium |

---

## 16. Known Technical Debt

These issues are identified, documented, and have defined remediation paths.

### 16.1 Three Parallel CLI Entry Points
**Severity:** High  
**Files:** `main.py`, `aura_cli/cli_main.py`, `cli/cli_main.py`, `aura-cli/main.py`  
**Problem:** Overlapping argument parsing, duplicate command definitions, unclear which is canonical.  
**Remediation:**
1. Designate `aura_cli/cli_main.py` (465L) as the **sole canonical entry point**
2. Rewrite `main.py` as a 3-line shim: `from aura_cli.cli_main import main; main()`
3. Archive `cli/cli_main.py` to `archive/` with deprecation note
4. Delete `aura-cli/main.py` stub
5. Update all documentation and workflow references

### 16.2 Duplicated GitError Exception Class
**Severity:** Medium  
**Files:** `core/git_tools.py` AND `core/exceptions.py`  
**Problem:** Two definitions of `GitError` — importing from wrong module causes silent type mismatch in `except` clauses.  
**Remediation:**
1. Keep canonical definition in `core/exceptions.py` only
2. In `core/git_tools.py`: `from core.exceptions import GitError`
3. Add `__all__` to `exceptions.py` to prevent future re-definition

### 16.3 Duplicated Task/TaskManager
**Severity:** Medium  
**Files:** `task_manager.py` (root) AND `core/task_manager.py`  
**Problem:** Two `TaskManager` implementations; unclear which is used by orchestrator.  
**Remediation:**
1. Audit imports to determine which is actually used
2. Merge into `core/task_manager.py` (keep in `core/`)
3. Delete root `task_manager.py`
4. Update all imports

### 16.4 CallRequest / ToolCallRequest Duplication
**Severity:** Medium  
**Files:** `tools/mcp_server.py`, `tools/aura_mcp_skills_server.py`, and 2 other MCP servers (4 total)  
**Problem:** `ToolCallRequest` dataclass defined 4 times with potential schema divergence.  
**Remediation:**
1. Extract to `core/schema.py` or `tools/mcp_types.py`
2. All MCP servers import from the shared location
3. Enforce via `ruff` rule or import checker in CI

### 16.5 HybridClosedLoop vs LoopOrchestrator (Legacy vs Modern)
**Severity:** Medium  
**Files:** `core/hybrid_loop.py` (6 imports) vs `core/orchestrator.py`  
**Problem:** `HybridClosedLoop` coexists with `LoopOrchestrator` with no documented migration path. 6 files still import the legacy module.  
**Remediation:**
1. Document `HybridClosedLoop` as deprecated in docstring
2. Create migration guide: which features exist only in `HybridClosedLoop`
3. Port any missing features to `LoopOrchestrator`
4. Migrate the 6 imports, remove `hybrid_loop.py`

### 16.6 No Unit Tests for Agent Implementations
**Severity:** High  
**Scope:** 13 agent files, 0 unit tests  
**Problem:** All agent logic is only tested via E2E fakes or not at all. Regression risk on refactors.  
**Remediation:**
1. Create `tests/agents/` directory
2. Use `unittest.mock.AsyncMock` for LLM calls
3. Target P0 agents first: `coder.py`, `planner.py`, `verifier.py`
4. Add to CI requirements (min coverage threshold per module)

### 16.7 goal_queue.json vs goal_queue_v2.json Path Confusion
**Severity:** Low  
**Files:** `memory/goal_queue.json` (configured), potential stale `goal_queue_v2.json` references  
**Problem:** Version suffix in filename creates confusion; config and code may reference different paths.  
**Remediation:**
1. Standardize on `goal_queue.json` (no version suffix)
2. Add migration script: if `goal_queue_v2.json` exists, merge into `goal_queue.json`
3. Remove all `_v2` references from codebase

### 16.8 MCP Server Port Proliferation Without Registry
**Severity:** Low**  
**Ports:** 8001, 8002, 8006 (+ 2 TBD)  
**Problem:** No central port registry; hardcoded port numbers scattered in 5 files; port conflicts on multi-server startup.  
**Remediation:**
1. Add `[mcp_servers]` section to `aura.config.json`:
   ```json
   "mcp_servers": {
     "dev_tools": {"port": 8001},
     "skills": {"port": 8002},
     "control": {"port": 8003},
     "agentic_loop": {"port": 8006},
     "copilot": {"port": 8007}
   }
   ```
2. All MCP servers read port from config
3. Add `aura mcp list` command showing server status/port

---

## 17. Deployment & Operations

### 17.1 Local Development

```bash
# Prerequisites
python3.10+ | pip | git

# Setup
git clone https://github.com/asshat1981ar/aura-cli
cd aura-cli
pip install -r requirements.txt

# Configure
cp aura.config.json.example aura.config.json
export OPENROUTER_API_KEY="sk-or-..."

# Verify
python -m pytest tests/ -q  # 377 tests should pass

# Run
python main.py loop --max-cycles 5
```

### 17.2 Docker

```dockerfile
# Dockerfile provided
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py", "loop"]
```

```bash
docker build -t aura-cli .
docker run -e OPENROUTER_API_KEY=$KEY -v ./memory:/app/memory aura-cli
```

### 17.3 Android/Termux

AURA is tested and benchmarked on Android/Termux:
- All performance benchmarks above measured on Termux
- SQLite works natively
- No Android-specific code paths required
- `pip install termux-compat` may be needed for some deps

### 17.4 CI/CD

GitHub Actions matrix: Python 3.10 + 3.11 on push and PR to `main`.

```yaml
# ci.yml
strategy:
  matrix:
    python-version: ["3.10", "3.11"]
steps:
  - uses: actions/setup-python@v4
  - run: pip install -r requirements.txt
  - run: python -m pytest tests/ -q --tb=short
```

### 17.5 Monitoring & Health

- **Logs:** `logs/aura.log` (JSON structured, rotated daily)
- **Metrics:** `GET /metrics` on HTTP API server
- **Cycle summaries:** `memory/store/cycle_summaries/YYYY-MM-DD/`
- **Decision log:** `memory/store/decision_log.jsonl`
- **Health monitor:** Auto-generates goals on quality drift

### 17.6 Secrets Management

| Secret | Env Var | Required |
|--------|---------|----------|
| OpenRouter API key | `OPENROUTER_API_KEY` | Yes (primary) |
| OpenAI API key | `OPENAI_API_KEY` | No (fallback) |
| Gemini API key | `GEMINI_API_KEY` | No (review) |
| Momento API key | `MOMENTO_API_KEY` | No (cloud cache) |
| GitHub token | `GITHUB_TOKEN` | For workflows |

---

## 18. Roadmap & Future Goals

### 18.1 Priority Queue (P0 — Critical)

| # | Goal | Rationale |
|---|------|-----------|
| R1 | **Consolidate CLI entry points** (§16.1) | Eliminates confusing dual-entry architecture |
| R2 | **Unit tests for all 16 agents** | P0 coverage gap; regression risk |
| R3 | **Deduplicate GitError + TaskManager** (§16.2, §16.3) | Silent bugs in production |
| R4 | **Unit tests for WorkflowEngine** (879L, 0 tests) | Largest untested module |
| R5 | **MCP server port registry** (§16.8) | Prevents startup conflicts |

### 18.2 High Priority (P1 — Next Sprint)

| # | Goal | Rationale |
|---|------|-----------|
| R6 | **Per-skill unit tests for all 28 skills** | Only registry-tested currently |
| R7 | **Migrate HybridClosedLoop users to LoopOrchestrator** (§16.5) | Remove legacy |
| R8 | **MCP server authentication (API key)** | Security for production deployment |
| R9 | **Integration tests for HTTP API server** | Critical path untested |
| R10 | **ContextGraph query optimization** | Potential N+1 on large graphs |

### 18.3 Medium Priority (P2 — Upcoming)

| # | Goal | Rationale |
|---|------|-----------|
| R11 | **Async pipeline execution** | 5 sequential LLM calls → parallel |
| R12 | **Streaming cycle output** (WebSocket/SSE) | Real-time UX for long cycles |
| R13 | **Goal prioritization ML model** | Replace manual priority with learned weights |
| R14 | **Multi-repo support** | AURA currently single-repo only |
| R15 | **Plugin SDK for custom skills** | Enable community skill packages |
| R16 | **ContextGraph visualization endpoint** | Graph viz for debugging |
| R17 | **Coverage enforcement in CI** (`--cov-fail-under 80`) | Prevent coverage regression |

### 18.4 Future Exploration (P3)

| # | Goal | Rationale |
|---|------|-----------|
| R18 | **AURA self-hosts: improves own codebase via CI loop** | North star goal |
| R19 | **Distributed goal execution across N machines** | Scale for large codebases |
| R20 | **Fine-tuned local model for AURA_TARGET parsing** | Reduce OpenRouter dependency |
| R21 | **Voice goal input** | Accessibility / mobile UX |
| R22 | **VS Code extension with MCP integration** | Developer adoption |
| R23 | **Benchmark suite vs Copilot/Cursor on real tasks** | Competitive positioning |
| R24 | **ToolCallRequest deduplication** (§16.4) | Code quality |

---

## 19. Glossary

| Term | Definition |
|------|-----------|
| **AURA** | Autonomous Unified Reasoning Agent |
| **ABC** | Abstract Base Class — defines interface contract |
| **AgentBase** | Base class for all 16 agents; error-swallowing `run()` |
| **AtomicChangeSet** | All-or-nothing filesystem write transaction |
| **AURA_TARGET** | Comment directive marking code insertion point |
| **Brain** | SQLite-backed persistent memory (L2) |
| **ContextGraph** | SQLite property graph (files/goals/skills/weaknesses) |
| **ConvergenceEscapeLoop** | Anti-oscillation mechanism |
| **CycleSummary** | Structured output of a complete pipeline run |
| **DeepReflectionLoop** | Cross-cycle pattern analysis (every 5 cycles) |
| **EMA** | Exponential Moving Average — used in RouterAgent scoring |
| **GoalQueue** | Priority queue of pending natural language goals |
| **HumanGate** | Pre-apply blocking gate for security/coverage |
| **LoopOrchestrator** | Main 10-phase pipeline engine (744L) |
| **MCP** | Model Context Protocol — JSON-RPC for LLM tool use |
| **OscillationDetector** | Detects repeated failure patterns |
| **PropagationEngine** | Forward-chaining event→rule→action system |
| **RouterAgent** | EMA-weighted LLM provider selector |
| **Sandbox** | Isolated subprocess for generated code execution |
| **SkillBase** | ABC for all 28 analysis skills |
| **SkillChainer** | Chains security findings → remediation goals |
| **SkillMetrics** | Per-skill performance telemetry |
| **TaskBundle** | Unified action spec output of Phase 5 |
| **TaskHierarchy** | Sub-task decomposition tree |
| **VectorStore** | Embedding-based semantic search (in Brain) |
| **WeaknessRemediator** | Converts Brain weaknesses → fix goals |
| **WorkflowEngine** | Reusable agentic workflow framework (879L) |

---

*Document generated from deep semantic analysis of AURA CLI codebase at commit `5c9f90f`.*  
*156 Python files | 26,428 LOC | 250 classes | 1,180 functions | 379 tests passing*  
*PRD v2.0 — supersedes v1.0 (bb32efb)*
