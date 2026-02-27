# AURA CLI — Final Development Guide

**Status**: Active Reference  
**Last Updated**: February 27, 2026  
**Codebase Snapshot**: 210 Python files · 39,788 LOC · 1,165 tests passing  
**Purpose**: Systematic audit of implementation status and prioritised completion checklist for the final development stages.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Inventory](#2-component-inventory)
3. [PRD Completion Status](#3-prd-completion-status)
4. [Test Infrastructure Status](#4-test-infrastructure-status)
5. [Known Issues & Remaining Work](#5-known-issues--remaining-work)
6. [Completion Checklist](#6-completion-checklist)
7. [Dependency Reference](#7-dependency-reference)
8. [Development Workflow](#8-development-workflow)
9. [Deployment Readiness Checklist](#9-deployment-readiness-checklist)

---

## 1. Architecture Overview

AURA is an **autonomous AI development loop** that ingests goals, plans changes, generates code, applies them, verifies the result, and reflects — repeatedly — until each goal is resolved.

### Runtime Initialization (`aura_cli/cli_main.py::create_runtime()`)

```
create_runtime()
 ├── GoalQueue          (memory/goal_queue_v2.json)
 ├── ModelAdapter       (core/model_adapter.py)
 ├── Brain              (memory/brain.py → memory/brain_v2.db)
 ├── VectorStore        (memory/vector_store.py)
 ├── VectorStoreV2      (memory/vector_store_v2.py)  ← ASCM v2
 ├── EmbeddingProvider  (memory/embedding_provider.py)
 ├── RouterAgent        (agents/router.py)
 ├── DebuggerAgent      (agents/debugger.py)
 ├── PlannerAgent       (agents/planner.py)
 ├── AdaptivePipeline   (core/adaptive_pipeline.py)
 ├── ContextGraph       (core/context_graph.py)
 ├── LoopOrchestrator   (core/orchestrator.py)  ← PRIMARY ORCHESTRATOR
 └── GitTools           (core/git_tools.py)
```

### Orchestration Pipeline (`core/orchestrator.py::LoopOrchestrator`)

Each `run_cycle()` call executes 10 phases in order:

| Phase | Module | Description |
|-------|--------|-------------|
| 1. ingest | `agents/ingest.py` | Gather project context and memory hints |
| 2. skill_dispatch | `core/skill_dispatcher.py` | Run adaptive static-analysis skills |
| 3. plan | `agents/planner.py` | Generate step-by-step implementation plan (with retries) |
| 4. critique | `agents/critic.py` | Adversarially review the plan |
| 5. synthesize | `agents/synthesizer.py` | Merge plan + critique into actionable task bundle |
| 6. act | `agents/coder.py` | Generate code changes (with retries on failure) |
| 7. sandbox | `agents/sandbox.py` | Execute snippet in isolated subprocess |
| 8. apply | `core/file_tools.py` | Write file changes atomically |
| 9. verify | `agents/verifier.py` | Run tests/linters against applied changes |
| 10. reflect | `agents/reflector.py` | Summarise outcomes and update skill weights |

### Memory Architecture

| Store | Location | Contents | Access Pattern |
|-------|----------|----------|----------------|
| `Brain` | `memory/brain_v2.db` | General memories, weaknesses, embeddings, response cache | `recall_with_budget()`, `recall_recent()`, `count_memories()` |
| `VectorStoreV2` | `memory/brain_v2.db` (shared) | Provenance-tagged semantic memory records | `search()`, `add()`, `migrate_from_v1()` |
| `EmbeddingProvider` | In-memory / API | TF-IDF (local) or OpenAI text-embedding-3-small | `embed()`, `available()` |
| `ContextBudgetManager` | `core/context_budget.py` | Token-aware context assembly | `assemble()` |
| `MemoryStore` | `memory/store/` | Per-cycle summaries (ReflectorAgent) | File-based append |
| `GoalQueue` | `memory/goal_queue_v2.json` | Pending and in-progress goals | `add()`, `next()`, `batch_add()` |

### HTTP APIs

| Server | Port | File | Auth |
|--------|------|------|------|
| AURA HTTP API | 8001 | `aura_cli/server.py` | `AGENT_API_TOKEN` env var |
| MCP Skills Server | 8002 | `tools/aura_mcp_skills_server.py` | `MCP_API_TOKEN` env var |

---

## 2. Component Inventory

### Agents (`agents/` — 14 modules)

| Module | Role | Status |
|--------|------|--------|
| `base.py` | `AgentBase` abstract class | ✅ Complete |
| `coder.py` | Code generation; targets files via `# AURA_TARGET:` | ✅ Complete |
| `critic.py` | Adversarial plan review | ✅ Complete |
| `debugger.py` | Fix invalid change-sets (retried by orchestrator) | ✅ Complete |
| `ingest.py` | Project context ingestion (uses ASCM v2 if available) | ✅ Complete |
| `mutator.py` | Introduce code mutations for exploration | ✅ Complete |
| `planner.py` | Step-by-step plan generation | ✅ Complete |
| `reflector.py` | Cycle-summary persistence + skill weight update | ✅ Complete |
| `registry.py` | `default_agents()` wiring for orchestrator | ✅ Complete |
| `router.py` | EMA-ranked model routing; persists stats keyed by `__router_stats__` | ✅ Complete |
| `sandbox.py` | Isolated code execution subprocess | ✅ Complete |
| `scaffolder.py` | Boilerplate generation for new project types | ✅ Complete |
| `synthesizer.py` | Merges plan + critique into `task_bundle` | ✅ Complete |
| `tester.py` | Test suite execution agent | ✅ Complete |
| `verifier.py` | Post-apply verification (lint + tests) | ✅ Complete |
| `applicator.py` | Change application bridge | ✅ Complete |

### Skills (`agents/skills/` — 30 modules)

All skills extend `SkillBase`, implement `_run(input_data) -> dict`, and **never raise** (errors returned as `{"error": "..."}`).

| # | Skill | Key Inputs | Key Outputs |
|---|-------|-----------|-------------|
| 1 | `dependency_analyzer` | `project_root` | `packages`, `vulnerabilities` |
| 2 | `architecture_validator` | `project_root` | `circular_deps`, `coupling_score` |
| 3 | `complexity_scorer` | `code`/`project_root` | `functions`, `high_risk_count` |
| 4 | `test_coverage_analyzer` | `project_root` | `coverage_pct`, `meets_target` |
| 5 | `doc_generator` | `code`/`project_root` | `generated_docstrings` |
| 6 | `performance_profiler` | `code` | `hotspots`, `antipatterns` |
| 7 | `refactoring_advisor` | `code`/`project_root` | `suggestions`, `smell_count` |
| 8 | `schema_validator` | `schema`, `instance`, `code` | `valid`, `errors` |
| 9 | `security_scanner` | `code`/`project_root` | `findings`, `critical_count` |
| 10 | `type_checker` | `project_root`/`file_path` | `type_errors`, `annotation_coverage_pct` |
| 11 | `linter_enforcer` | `project_root`/`file_path` | `violations`, `naming_violations` |
| 12 | `incremental_differ` | `old_code`, `new_code` | `diff_summary`, `added_symbols` |
| 13 | `tech_debt_quantifier` | `project_root` | `debt_score`, `debt_items` |
| 14 | `api_contract_validator` | `code`/`project_root` | `endpoints`, `breaking_changes` |
| 15 | `generation_quality_checker` | `task`, `generated_code` | `quality_score`, `intent_match_score` |
| 16 | `git_history_analyzer` | `project_root` | `hotspot_files`, `patterns` |
| 17 | `skill_composer` | `goal` | `workflow`, `goal_category` |
| 18 | `error_pattern_matcher` | `current_error`, `error_history?` | `matched_pattern`, `fix_steps` |
| 19 | `code_clone_detector` | `project_root` | `exact_clones`, `near_duplicates` |
| 20 | `adaptive_strategy_selector` | `goal`, `record_result?` | `recommended_strategy`, `confidence` |
| 21 | `web_fetcher` | `url` or `query` | `text`, `title`, `source` |
| 22 | `symbol_indexer` | `project_root` | `symbols`, `import_graph` |
| 23 | `multi_file_editor` | `goal`, `project_root?` | `change_plan`, `affected_count` |
| 24 | `dockerfile_analyzer` | `project_root` | `issues`, `recommendations` |
| 25 | `observability_checker` | `project_root` | `findings`, `score` |
| 26 | `changelog_generator` | `project_root` | `changelog`, `entries` |
| 27 | `database_query_analyzer` | `code`/`project_root` | `queries`, `issues` |
| 28 | `skill_failure_analyzer` | `failures` | `patterns`, `recommendations` |
| 29 | `security_hardener` | `code`/`project_root` | `hardening_suggestions` |
| 30 | `structural_analyzer` | `project_root` | `structure`, `coupling_map` |

### Core Infrastructure (`core/` — 43 modules)

| Module | Role | Status |
|--------|------|--------|
| `orchestrator.py` | `LoopOrchestrator` — primary 10-phase pipeline | ✅ Complete |
| `adaptive_pipeline.py` | `AdaptivePipeline` / `PipelineConfig` — intensity-aware config | ✅ Complete (confidence field added) |
| `file_tools.py` | Atomic code application with overwrite-safety policy | ✅ Complete |
| `config_manager.py` | Priority-resolved config (env > json > defaults) | ✅ Complete |
| `model_adapter.py` | OpenRouter API + Gemini CLI fallback; response cache (1hr TTL) | ✅ Complete |
| `cycle_outcome.py` | `CycleOutcome` dataclass + serialization | ✅ Complete |
| `quality_snapshot.py` | Fast post-cycle quality metrics (< 500ms) | ✅ Complete |
| `context_budget.py` | `ContextBudgetManager` — token-aware greedy context fill | ✅ Complete |
| `context_graph.py` | SQLite-backed knowledge graph for goal/skill correlation | ✅ Complete |
| `skill_dispatcher.py` | Classifies goals, dispatches appropriate skills | ✅ Complete |
| `schema.py` | `validate_phase_output()` per-phase JSON schema validation | ✅ Complete |
| `hybrid_loop.py` | `HybridClosedLoop` — **DEPRECATED**, issues `DeprecationWarning` | ✅ Deprecated |
| `evolution_loop.py` | Mutation-based improvement loop | ✅ Complete |
| `autonomous_discovery.py` | Scans for improvement opportunities; auto-enqueues goals | ✅ Complete |
| `workflow_engine.py` | `WorkflowEngine` — 8 classes, DAG-based step execution | ✅ Complete |
| `convergence_escape.py` | Oscillation detection and escape | ✅ Complete |
| `human_gate.py` | `HumanGate` — human-in-the-loop approval for risky changes | ✅ Complete |
| `goal_queue.py` | JSON-backed persistent goal queue | ✅ Complete |
| `goal_archive.py` | Completed goal archive | ✅ Complete |
| `goal_decomposer.py` | Breaks complex goals into sub-goals | ✅ Complete |
| `git_tools.py` | `GitTools` — commit, rollback, stash via gitpython | ✅ Complete |
| `health_monitor.py` | System health monitoring | ✅ Complete |
| `logging_utils.py` | Structured JSON logging (`log_json()`) | ✅ Complete |
| `memory_compaction.py` | Prune and compact Brain memory | ✅ Complete |
| `policy.py` | Stopping condition evaluator | ✅ Complete |
| `propagation_engine.py` | Change propagation across related files | ✅ Complete |
| `reflection_loop.py` | Persistent learning from past cycles | ✅ Complete |
| `sanitizer.py` | Input/output sanitization | ✅ Complete |
| `skill_weight_adapter.py` | Dynamic skill weight adjustment | ✅ Complete |
| `vector_store.py` | VectorStore v1 (legacy; migration path to v2 available) | ✅ Legacy |
| `weakness_remediator.py` | Targeted weakness fixing | ✅ Complete |

### Memory Layer (`memory/` — 10 modules)

| Module | Role | Status |
|--------|------|--------|
| `brain.py` | `Brain` — SQLite-backed general memory, `recall_with_budget()`, `count_memories()` | ✅ Complete |
| `vector_store_v2.py` | `VectorStoreV2` / `MemoryRecord` / `SearchHit` — ASCM v2 backend | ✅ Complete |
| `embedding_provider.py` | `LocalEmbeddingProvider` (TF-IDF/SVD, offline) + `OpenAIEmbeddingProvider` | ✅ Complete |
| `store.py` | `MemoryStore` — file-based per-cycle summary persistence | ✅ Complete |
| `controller.py` | Memory lifecycle management | ✅ Complete |
| `brain.py` (momento) | `MomentoBrain` — cloud-backed variant | ✅ Complete |
| `cache_adapter_factory.py` | Factory for local vs. Momento cache | ✅ Complete |
| `local_cache_adapter.py` | In-process TTL cache | ✅ Complete |
| `momento_adapter.py` | Momento serverless cache adapter | ✅ Complete |

### TUI / Observability (`aura_cli/tui/`)

| File | Role | Status |
|------|------|--------|
| `app.py` | Main TUI application with rich Live display | ✅ Complete |
| `panels/cycle_panel.py` | Phase progress with spinners | ✅ Complete |
| `panels/queue_panel.py` | GoalQueue live view | ✅ Complete |
| `panels/memory_panel.py` | Brain entries browser | ✅ Complete |
| `panels/metrics_panel.py` | Performance sparklines | ✅ Complete |
| `log_streamer.py` | JSON log → rich colorized output | ✅ Complete |

---

## 3. PRD Completion Status

### PRD-001: R4 Agent Recall Optimization ✅ COMPLETE

**Goal**: Replace all `recall_all()` calls with budget-aware alternatives.

**Verified state** (grep `recall_all()` in `agents/` and `core/`):
- `core/hybrid_loop.py` — ✅ Fixed (now uses `brain.count_memories()`)
- All other agent/core files — ✅ No remaining `recall_all()` calls

**Acceptance criteria met**:
- [x] No agent prompt embeds more than 4,000 tokens of memory context
- [x] `hybrid_loop.py` uses `count_memories()` instead of `len(recall_all())`
- [x] 44/44 R4 optimization tests passing (`tests/test_optimization_r4.py`)

---

### PRD-002: ASCM v2 — Semantic Context Manager ✅ COMPLETE

**Files delivered**:
- `memory/vector_store_v2.py` — `VectorStoreV2`, `MemoryRecord`, `SearchHit`
- `memory/embedding_provider.py` — `LocalEmbeddingProvider` (offline TF-IDF/SVD) + `OpenAIEmbeddingProvider`
- `core/context_budget.py` — `ContextBudgetManager` (greedy token fill)
- `tests/test_ascm_v2.py` — 69 tests, all passing

**Acceptance criteria met**:
- [x] `VectorStoreV2` stores and retrieves with metadata filters
- [x] `LocalEmbeddingProvider` works fully offline (no API key required)
- [x] Deduplication via SHA-256 content hash
- [x] Token budget respected via `ContextBudgetManager`
- [x] `migrate_from_v1()` available
- [x] 69/69 ASCM v2 tests passing

---

### PRD-003: Autonomous Learning Loop ✅ COMPLETE

**Files delivered**:
- `core/cycle_outcome.py` — `CycleOutcome` dataclass + `to_json()` / `from_json()`
- `core/quality_snapshot.py` — `run_quality_snapshot()` (< 500ms)
- `core/hybrid_loop.py` — `DeprecationWarning` on instantiation
- `agents/router.py` — keyed `__router_stats__` lookup (not `recall_all()`)
- `core/adaptive_pipeline.py` — `record_outcome()` for strategy persistence

**Acceptance criteria met**:
- [x] `CycleOutcome` schema matches PRD spec (all fields present)
- [x] `HybridClosedLoop` issues `DeprecationWarning` on instantiation
- [x] `RouterAgent` uses keyed lookup, not `recall_all()`
- [x] 38/38 learning loop tests passing (`tests/test_learning_loop.py`)
- [x] 25/25 cycle outcome tests passing (`tests/test_cycle_outcome.py`)

---

### PRD-004: Full Test Coverage & Quality ✅ SUBSTANTIALLY COMPLETE

**Target**: ≥ 930 tests total.  
**Actual**: **1,158 tests passing** (74 subtests).

| Test file | Tests | Status |
|-----------|-------|--------|
| `tests/test_optimization_r4.py` | 44 | ✅ Pass |
| `tests/test_ascm_v2.py` | 69 | ✅ Pass |
| `tests/test_learning_loop.py` | 38 | ✅ Pass |
| `tests/test_cycle_outcome.py` | 25 | ✅ Pass |
| `tests/test_tui.py` | 45 | ✅ Pass |
| `tests/test_skills.py` | see below | ✅ Pass |
| `tests/test_all_skills.py` | see below | ✅ Pass |
| `tests/test_skills_comprehensive.py` | see below | ✅ Pass |
| `tests/test_workflow_engine.py` | see below | ✅ Pass |
| `tests/test_workflow_engine_full.py` | see below | ✅ Pass |
| `tests/test_agents_unit.py` | see below | ✅ Pass |
| `tests/integration/test_orchestrator_e2e.py` | see below | ✅ Pass |

---

### PRD-005: AURA Studio — TUI & Observability ✅ SUBSTANTIALLY COMPLETE

**Files delivered**:
- `aura_cli/tui/__init__.py`
- `aura_cli/tui/app.py`
- `aura_cli/tui/panels/cycle_panel.py`
- `aura_cli/tui/panels/queue_panel.py`
- `aura_cli/tui/panels/memory_panel.py`
- `aura_cli/tui/panels/metrics_panel.py`
- `aura_cli/tui/log_streamer.py`
- `tests/test_tui.py` — 45 tests, all passing

**CLI commands available**:
```bash
python3 main.py watch           # Launch TUI dashboard
python3 main.py studio          # Alias for watch
python3 main.py logs --tail 50  # Colorized log stream
python3 main.py doctor          # Health check
python3 main.py goal status     # Goal queue list
python3 main.py diag            # MCP diagnostics
```

**Acceptance criteria met**:
- [x] TUI panels implemented (cycle, queue, memory, metrics)
- [x] Log streamer with colorized JSON output
- [x] `aura doctor` check structure in place
- [x] 45/45 TUI tests passing

---

## 4. Test Infrastructure Status

### Overall

```
Total:    1,165 passing  (+ 74 subtests)
Failing:      0  (all fixed with complete requirements installation)
Erroring:     1  (test_aura_doctor_root.py — root-level file, not in tests/)
```

> **Note**: The 7 previously-failing tests were caused by missing optional dependencies
> (`networkx`, `textblob`, `uvicorn`, `python-dotenv`). Installing all packages from
> `requirements.txt` (or `tools/requirements.txt`) resolves all failures.

### Failing Tests (0)

All tests pass when all packages from `requirements.txt` are installed.

### Previously-failing Tests (now resolved)

| Test | Cause | Resolution |
|------|-------|-----------|
| `test_optimization_r2.py::TestBrainLazyNetworkx` (4 tests) | `networkx` not installed | Added to `requirements.txt` |
| `test_optimization_simulations.py::TestBrainLazyTextblob` | `textblob` not installed | Added to `requirements.txt` |
| `test_aura_doctor.py` (2 tests) | `uvicorn`, `python-dotenv` not installed | Added to `requirements.txt` |
| `tests/integration/` + others (14 tests) | `PipelineConfig` missing `confidence` field | Fixed: field added to `core/adaptive_pipeline.py` |

### Test Coverage by Module Area

| Area | Test Files | Estimated Coverage |
|------|-----------|-------------------|
| Agents | `test_agents_unit.py`, `test_agents_sandbox.py` | ~65% |
| Skills (all 30) | `test_skills.py`, `test_all_skills.py`, `test_skills_comprehensive.py`, `test_new_skills.py` | ~80% |
| Core orchestrator | `tests/integration/test_orchestrator_e2e.py`, `test_performance_simulations.py` | ~65% |
| `WorkflowEngine` | `test_workflow_engine.py`, `test_workflow_engine_full.py` | ~75% |
| ASCM v2 / Memory | `test_ascm_v2.py`, `test_semantic_memory.py`, `test_retrieval_quality.py` | ~80% |
| File tools | `test_file_tools.py`, `test_atomic_change_set_policy.py` | ~90% |
| CLI dispatch | `test_cli_main_dispatch.py`, `test_cli_contract.py`, `test_cli_options.py` | ~70% |
| Server API | `test_server_api.py`, `test_server_sse.py`, `test_mcp_tools.py` | ~60% |
| TUI | `test_tui.py` | ~60% |
| Learning loop | `test_learning_loop.py`, `test_cycle_outcome.py` | ~85% |

---

## 5. Known Issues & Remaining Work

### 🔴 High Priority

#### Issue 1: ~~Root-level `requirements.txt` missing~~ ✅ RESOLVED

**Resolution**: `requirements.txt` created at repo root including all packages from `tools/requirements.txt` plus dev/test dependencies (`pytest`, `httpx`, `anyio`).

#### Issue 2: ~~`PipelineConfig` missing `confidence` field~~ ✅ RESOLVED

**Resolution**: `confidence: float = 0.0` field added to `PipelineConfig` in `core/adaptive_pipeline.py`. This fixed 14 test failures across integration tests, performance simulations, and dry-run workflow tests.

---

### 🟠 Medium Priority

#### Issue 3: `HybridClosedLoop` callers not fully migrated

`core/hybrid_loop.py` issues `DeprecationWarning`, but some CLI wiring paths still instantiate it:
- `aura_cli/cli_main.py` — check if `create_runtime()` still creates `HybridClosedLoop`
- `tests/test_learning_loop.py`, `tests/test_cli_main_dispatch.py`, `tests/test_refactor_debug_fixes.py` — still import it for testing

**Fix**: Migrate `create_runtime()` to use `LoopOrchestrator` directly. Update tests to use the non-deprecated path.

#### Issue 4: `aura_doctor.py` at root vs `aura_cli/doctor.py`

There are two doctor implementations:
- `aura_doctor.py` (root) — standalone script, ~180 lines
- `aura_cli/doctor.py` — CLI-integrated version (canonical)

PRD-005 called for consolidation into `aura_cli/doctor.py`. The root file remains but the `test_aura_doctor_root.py` test also remains. The consolidated version is the canonical one; the root file should either be removed or reduced to a shim.

---

### 🟡 Low Priority

#### Issue 5: `core/test_goal_queue.py` — outdated SQLite-based test

`core/test_goal_queue.py` tests the old SQLite-backed `GoalQueue`. The current implementation uses JSON (`memory/goal_queue_v2.json`). This file is documented as outdated in `docs/INTEGRATION_MAP.md` and is ignored by CI.

**Fix**: Either delete it or update it to test the current JSON-backed implementation.

#### Issue 6: `fix_*.sh` / `patch_*.sh` scripts at repo root

Twelve shell scripts (`fix_all_aura_issues.sh`, `patch_main_resilient_apply_v4.sh`, etc.) at the repo root are leftover from earlier debugging sessions. They should be removed or moved to a `scripts/archive/` folder.

#### Issue 7: Archive file

`archive.zip` at the repo root should be removed or documented (gitignored).

---

## 6. Completion Checklist

### Pre-release Required

- [x] **Create `requirements.txt` at repo root** ✅ DONE — includes all packages + dev tools
- [x] **Fix `PipelineConfig.confidence` attribute error** ✅ DONE — `confidence: float = 0.0` added
- [x] **Fix zero `recall_all()` calls in `agents/` and `core/`** ✅ DONE — last usage in `hybrid_loop.py` replaced with `count_memories()`
- [x] **Confirm all PRD acceptance criteria green** ✅ DONE (see Section 3)
- [x] **Confirm CI test count ≥ 1,158** ✅ DONE — 1,165 passing

### Post-release Polish

- [ ] Migrate `create_runtime()` in `aura_cli/cli_main.py` to use `LoopOrchestrator` directly (remove `HybridClosedLoop` instantiation)
- [ ] Consolidate `aura_doctor.py` (root) into `aura_cli/doctor.py`; delete root file
- [ ] Remove or archive leftover `fix_*.sh` / `patch_*.sh` scripts at repo root
- [ ] Remove `archive.zip` from repo root
- [ ] Add `core/test_goal_queue.py` to `.gitignore` exclusions or update it for the JSON-backed queue
- [ ] Validate `aura watch` launches without errors after `pip install rich`
- [ ] Validate `aura doctor` passes all 10+ checks in a clean environment

---

## 7. Dependency Reference

### Runtime (production)

From `tools/requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | `0.129.0` | HTTP API servers (ports 8001, 8002) |
| `uvicorn` | `0.40.0` | ASGI server |
| `requests` | `2.32.3` | HTTP client (OpenRouter, MCP calls) |
| `pydantic` | `2.12.5` | Schema validation |
| `python-dotenv` | latest | `.env` file loading |
| `numpy` | latest | Vector math for embeddings |
| `gitpython` | latest | `GitTools` — git operations |
| `rich` | latest | TUI rendering (aura_cli/tui/) |
| `textblob` | latest | NLP for brain lazy-load optimization |
| `networkx` | latest | Brain knowledge graph (`brain.relate()`) |

### Dev / Test

```
pytest>=7.0
httpx>=0.24       # FastAPI TestClient dependency
anyio>=4.0        # async test support
```

### Optional / Environment

| Env Variable | Default | Purpose |
|--------------|---------|---------|
| `AURA_API_KEY` | — | OpenRouter API key (required for LLM calls) |
| `AURA_DRY_RUN=1` | off | No file or memory writes |
| `AURA_SKIP_CHDIR=1` | off | Required in tests to prevent `os.chdir()` |
| `AURA_STRICT_SCHEMA=1` | off | Abort cycle on schema validation failure |
| `GEMINI_CLI_PATH` | — | Path to `gemini` binary (model fallback) |
| `AGENT_API_TOKEN` | — | Auth token for HTTP API (port 8001) |
| `MCP_API_TOKEN` | — | Auth token for MCP Skills Server (port 8002) |
| `AGENT_API_ENABLE_RUN=1` | off | Enable `/run` endpoint on HTTP API |

---

## 8. Development Workflow

### Quick Start

```bash
# Install all dependencies
pip install -r tools/requirements.txt pytest httpx anyio

# Run all tests
python3 -m pytest

# Run targeted tests
python3 -m pytest tests/test_file_tools.py
python3 -m pytest tests/test_ascm_v2.py
python3 -m pytest tests/integration/

# Start a one-off goal run
AURA_SKIP_CHDIR=1 python3 main.py goal once "Refactor core/model_adapter.py" --dry-run

# Run the full loop on a queued goal
python3 main.py goal add "Improve test coverage" && python3 main.py goal run

# Launch the TUI dashboard
python3 main.py watch

# Run the HTTP API server
uvicorn aura_cli.server:app --port 8001

# Run the MCP Skills server
uvicorn tools.aura_mcp_skills_server:app --port 8002
```

### Code Change Conventions

1. **Structured logging**: always use `core/logging_utils.log_json(level, event_name, **kwargs)`. Never use `print()` or `logging.*` in production paths.
2. **File targeting**: the first line of any generated code block must be `# AURA_TARGET: path/to/file.py`.
3. **Overwrite policy**: apply changes through `apply_change_with_explicit_overwrite_policy()` in `core/file_tools.py`. Never call raw `replace_code()` from orchestration paths.
4. **Schema validation**: all phase outputs must pass `core/schema.validate_phase_output(phase_name, output)`.
5. **Skills never raise**: every skill's `_run()` must return `{"error": "..."}` on failure, not raise an exception.
6. **Tests use `AURA_SKIP_CHDIR=1`**: set this env var to prevent `os.chdir()` from changing the working directory during test runs.

### Adding a New Skill

1. Create `agents/skills/your_skill.py` extending `SkillBase`, set `name`, implement `_run()`.
2. Register in `agents/skills/registry.py` — add import and entry to the returned dict.
3. Add 5 tests to `tests/test_skills_comprehensive.py` (or a new file):
   - `test_{skill}_run_minimal_input`
   - `test_{skill}_run_returns_dict`
   - `test_{skill}_never_raises`
   - `test_{skill}_error_key_on_bad_input`
   - `test_{skill}_run_with_project_root`
4. Expose the skill via the MCP Skills Server (it will auto-register via `all_skills()`).

### Adding a New Agent

1. Extend `agents/base.AgentBase`, implement `run(input_data: dict) -> dict`.
2. Register in `agents/registry.default_agents()`.
3. Wire into `core/orchestrator.LoopOrchestrator` if it should be a pipeline phase.
4. Add tests to `tests/test_agents_unit.py`:
   - instantiation test
   - `run()` with mock adapter
   - error handling (adapter failure)
   - output schema validation

---

## 9. Deployment Readiness Checklist

### Environment

- [ ] `AURA_API_KEY` configured (OpenRouter)
- [ ] `AGENT_API_TOKEN` set for HTTP API
- [ ] `MCP_API_TOKEN` set for MCP Skills Server
- [ ] All packages from `tools/requirements.txt` installed
- [ ] `memory/` directory writable (Brain DB, goal queue, store)
- [ ] `aura.config.json` valid JSON with no real secrets committed

### Functional Checks

```bash
# Doctor check — all green
python3 main.py doctor

# Config bootstrap (first-time setup)
python3 main.py bootstrap

# Smoke test: one-off dry run
python3 main.py goal once "Hello world" --dry-run

# Health endpoint
curl http://localhost:8001/health

# Skills endpoint
curl http://localhost:8002/tools | python3 -m json.tool | head -20
```

### Security

- [ ] `aura.config.json` does NOT contain the real API key (use `AURA_API_KEY` env var)
- [ ] `memory/*.db` and `memory/*.json` are gitignored (check `.gitignore`)
- [ ] HTTP API endpoints protected with `AGENT_API_TOKEN` header
- [ ] `AGENT_API_ENABLE_RUN=1` only set in trusted environments

### Performance Baselines

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Memory recall latency | < 50ms | `python3 -m pytest tests/test_optimization_r4.py -v` |
| Full cycle time (mock LLM) | < 2s overhead | `python3 -m pytest tests/test_performance_simulations.py -v` |
| ASCM retrieval (1k records) | < 100ms | `python3 -m pytest tests/test_ascm_v2.py -v` |
| Quality snapshot | < 500ms | `python3 -m pytest tests/test_cycle_outcome.py -v` |

---

*This document was generated by systematic codebase inspection on 2026-02-27. Update it whenever a PRD acceptance criterion is closed or a new architectural decision is made.*
