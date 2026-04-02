# Semantic Codebase Scanner — SADD Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the remaining 7 tasks from the semantic scanner implementation plan using SADD parallel workstreams, maximizing concurrent execution across independent tasks.

**Architecture:** Three SADD waves with 3 independent workstreams in Wave 1, 2 in Wave 2, then sequential integration/testing in Waves 3-4. Each workstream maps to a subagent running the full TDD cycle. All detailed step-by-step code lives in the master plan — this document defines the execution DAG.

**Tech Stack:** SADD (`core/sadd/`), n8n dispatcher workflows (WF-0 through WF-6), existing agent-sdk test infrastructure.

**Master Plan:** `docs/superpowers/plans/2026-04-02-semantic-scanner.md` (contains all code, tests, and detailed steps)
**Spec:** `docs/superpowers/specs/2026-04-02-semantic-scanner-design.md`

---

## Completion Status

| Task | Description | Status | Workstream |
|------|-------------|--------|------------|
| 1 | Config scan fields | NOT STARTED | WS-A (Wave 1) |
| 2 | SQLite schema + DAL | **DONE** | — |
| 3 | AST scanner (Layers 1+2+3) | NOT STARTED | WS-B (Wave 1) |
| 4 | Semantic querier | NOT STARTED | WS-C (Wave 1) |
| 5 | Context builder integration | NOT STARTED | WS-D (Wave 2) |
| 6 | Tool registry + controller wiring | NOT STARTED | WS-E (Wave 3) |
| 7 | CLI scan command | NOT STARTED | WS-D (Wave 2) |
| 8 | Integration test | NOT STARTED | WS-F (Wave 4) |

---

## File Structure

**Already exists (Task 2 — complete):**

| File | Status |
|------|--------|
| `core/agent_sdk/semantic_schema.py` | DONE — 436 lines, full CRUD + FTS5 + WAL |

**To create:**

| File | Responsibility | Workstream |
|------|---------------|------------|
| `core/agent_sdk/semantic_scanner.py` | Three-layer pipeline: AST extraction, relationship analysis, LLM intent | WS-B |
| `core/agent_sdk/semantic_querier.py` | 7 query methods + architecture_overview | WS-C |
| `tests/test_semantic_schema.py` | Schema CRUD tests (Task 2 tests — not yet written) | WS-A |
| `tests/test_semantic_scanner.py` | Scanner pipeline tests | WS-B |
| `tests/test_semantic_querier.py` | Querier tests for all 7 query types | WS-C |
| `tests/test_semantic_integration.py` | End-to-end integration test | WS-F |

**To modify:**

| File | Change | Workstream |
|------|--------|------------|
| `core/agent_sdk/config.py` | Add 7 scan config fields + from_aura_config mappings | WS-A |
| `core/agent_sdk/context_builder.py` | Add semantic_querier param + _get_codebase_context() + prompt rendering | WS-D |
| `core/agent_sdk/controller.py` | Create querier in __init__, use self.context_builder in _build_prompt, pass querier to MCP server | WS-E |
| `core/agent_sdk/tool_registry.py` | Add query_codebase tool + _handle_query_codebase handler + semantic_querier dep | WS-E |
| `core/agent_sdk/cli_integration.py` | Add handle_agent_scan() + format_scan_stats() | WS-D |

---

## Wave 1: Independent Foundation (3 parallel workstreams)

All three workstreams depend only on the already-complete `semantic_schema.py`. No inter-dependencies. Run all three concurrently.

### Workstream A: Config Fields + Schema Tests (Tasks 1 + 2 tests)

**Dependencies:** None
**Files:** `core/agent_sdk/config.py`, `tests/test_agent_sdk_config.py`, `tests/test_semantic_schema.py`
**Estimated time:** ~10 min

#### Task 2 Tests (schema already implemented, tests missing)

- [ ] **Step 0a: Write schema tests** — Create `tests/test_semantic_schema.py` with `TestSemanticSchema` class (10 tests: init_creates_tables, upsert_file, insert_symbol, insert_import, insert_call_site, upsert_relationship, delete_file_cascades, record_scan_meta, fts_search, get_all_files). See master plan Task 2, Step 1 for exact test code.

**Important:** Adapt the tests to match the actual `semantic_schema.py` API:
  - `record_scan()` requires 7 args including `scan_time`
  - `insert_import()` takes `(file_id, imported_module, imported_name, is_from_import)`
  - No `clear_symbols_for_file` — use `clear_file_data(file_id)`

- [ ] **Step 0b: Run schema tests to verify they pass**

Run: `python3 -m pytest tests/test_semantic_schema.py -v`
Expected: All 10 tests PASS (implementation already exists)

- [ ] **Step 0c: Commit**

```bash
git add tests/test_semantic_schema.py
git commit -m "test: add semantic schema CRUD and FTS5 tests"
```

#### Task 1: Config Fields

- [ ] **Step 1: Write failing tests** — Append `TestAgentSDKConfigScan` to `tests/test_agent_sdk_config.py` (see master plan Task 1, Step 1)

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_agent_sdk_config.py::TestAgentSDKConfigScan -v`
Expected: FAIL — `AttributeError: 'AgentSDKConfig' object has no attribute 'semantic_index_path'`

- [ ] **Step 3: Add 7 fields to AgentSDKConfig dataclass**

In `core/agent_sdk/config.py`, add after existing `skill_weight_floor` field:

```python
    # Semantic scanner config
    semantic_index_path: Path = field(default_factory=lambda: Path("memory/semantic_index.db"))
    scan_llm_budget_usd: float = 0.50
    scan_llm_model: str = "claude-haiku-4-5"
    scan_exclude_patterns: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv", "*.egg-info",
    ])
    scan_min_function_lines: int = 10
    scan_min_file_lines: int = 5
    scan_batch_size: int = 10
```

Add corresponding fields to `from_aura_config()` cls(...) call:

```python
            semantic_index_path=Path(sdk_section.get("semantic_index_path", "memory/semantic_index.db")),
            scan_llm_budget_usd=sdk_section.get("scan_llm_budget_usd", 0.50),
            scan_llm_model=sdk_section.get("scan_llm_model", "claude-haiku-4-5"),
            scan_exclude_patterns=sdk_section.get("scan_exclude_patterns", [
                ".git", "__pycache__", "node_modules", ".venv", "venv", "*.egg-info",
            ]),
            scan_min_function_lines=sdk_section.get("scan_min_function_lines", 10),
            scan_min_file_lines=sdk_section.get("scan_min_file_lines", 5),
            scan_batch_size=sdk_section.get("scan_batch_size", 10),
```

- [ ] **Step 4: Run ALL config tests**

Run: `python3 -m pytest tests/test_agent_sdk_config.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/config.py tests/test_agent_sdk_config.py
git commit -m "feat: add semantic scanner config fields"
```

---

### Workstream B: AST Scanner Pipeline (Task 3)

**Dependencies:** `semantic_schema.py` (complete)
**Files:** `core/agent_sdk/semantic_scanner.py`, `tests/test_semantic_scanner.py`
**Estimated time:** ~15 min

- [ ] **Step 1: Write failing tests** — Create `tests/test_semantic_scanner.py` with 4 test classes: `TestASTExtraction` (5 tests), `TestRelationshipAnalysis` (2 tests), `TestFullScan` (2 tests), `TestIncrementalScan` (3 tests). See master plan Task 3, Step 1 for exact test code.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semantic_scanner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.agent_sdk.semantic_scanner'`

- [ ] **Step 3: Write implementation** — Create `core/agent_sdk/semantic_scanner.py` with:
  - `extract_symbols(file_path)` — AST walker for functions, classes, methods, decorators, base classes
  - `extract_imports(file_path)` — Import/ImportFrom node extraction
  - `extract_call_sites(file_path, symbol)` — Call node extraction within symbol line range
  - `build_relationships(db)` — File-level import relationships from module resolution
  - `compute_coupling_scores(db)` — `(inbound + outbound) / total_files` normalized to 0-1
  - `generate_module_summary(db, file_id, file_path, model_adapter)` — LLM module summary
  - `generate_function_intents(symbols, file_path, model_adapter, batch_size)` — Batched LLM intent summaries
  - `class SemanticScanner` — Pipeline orchestrator with `scan_full()`, `scan_incremental()`, `refresh_if_needed()`

See master plan Task 3, Step 3 for exact implementation code.

**Key implementation note:** The existing `semantic_schema.py` has a slightly different API than the plan's version. The actual file uses:
  - `insert_import(file_id, imported_module, imported_name, is_from_import)` — 4 positional args
  - `record_scan(scan_sha, files_scanned, symbols_found, llm_calls_made, llm_cost_usd, scan_type, scan_time)` — requires `scan_time` parameter
  - No `clear_symbols_for_file()` or `clear_imports_for_file()` — use `clear_file_data(file_id)` instead

The scanner implementation must match the ACTUAL schema API, not the plan's proposed schema. Read `core/agent_sdk/semantic_schema.py` before implementing.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_semantic_scanner.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/semantic_scanner.py tests/test_semantic_scanner.py
git commit -m "feat: add three-layer semantic scanner with AST extraction and relationships"
```

---

### Workstream C: Semantic Querier (Task 4)

**Dependencies:** `semantic_schema.py` (complete)
**Files:** `core/agent_sdk/semantic_querier.py`, `tests/test_semantic_querier.py`
**Estimated time:** ~10 min

- [ ] **Step 1: Write failing tests** — Create `tests/test_semantic_querier.py` with `_populate_test_db()` helper and `TestSemanticQuerier` class (8 tests: what_calls, what_depends_on, what_changes_break, summarize_file, summarize_symbol, find_similar, architecture_overview, recent_changes). See master plan Task 4, Step 1.

**Important:** The `_populate_test_db` helper must use the actual `semantic_schema.py` API:
  - `insert_import(file_id, imported_module, imported_name, is_from_import)` — boolean not int
  - `record_scan(scan_sha, files_scanned, symbols_found, llm_calls_made, llm_cost_usd, scan_type, scan_time)` — needs `scan_time` arg

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_semantic_querier.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation** — Create `core/agent_sdk/semantic_querier.py` with `SemanticQuerier` class implementing all 7 query methods. See master plan Task 4, Step 3.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_semantic_querier.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/semantic_querier.py tests/test_semantic_querier.py
git commit -m "feat: add semantic querier with 7 query types"
```

---

## Wave 2: Integration Layer (2 parallel workstreams)

**Gate:** Wave 1 must be complete. Specifically:
- WS-D needs: Querier (WS-C)
- WS-D (CLI) needs: Scanner (WS-B)

### Workstream D: Context Builder + CLI (Tasks 5 + 7)

**Dependencies:** WS-B (scanner), WS-C (querier)
**Files:** `core/agent_sdk/context_builder.py`, `core/agent_sdk/cli_integration.py`, `tests/test_agent_sdk_context_builder.py`, `tests/test_agent_sdk_cli_integration.py`

#### Task 5: Context Builder Integration

- [ ] **Step 1: Write failing tests** — Append `TestContextBuilderSemantic` (3 tests) to `tests/test_agent_sdk_context_builder.py`. See master plan Task 5, Step 1.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py::TestContextBuilderSemantic -v`

- [ ] **Step 3: Update context_builder.py** — 4 changes:
  1. Add `semantic_querier` kwarg to `__init__`
  2. Add `_get_codebase_context(goal)` method
  3. Merge codebase context into `build()` return dict
  4. Add codebase rendering to `build_system_prompt()`

See master plan Task 5, Step 3 for exact code.

- [ ] **Step 4: Run ALL context builder tests**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/context_builder.py tests/test_agent_sdk_context_builder.py
git commit -m "feat: integrate semantic querier into context builder"
```

#### Task 7: CLI Scan Command

- [ ] **Step 6: Write failing tests** — Append `TestAgentScanCLI` (2 tests) to `tests/test_agent_sdk_cli_integration.py`. See master plan Task 7, Step 1.

- [ ] **Step 7: Run tests to verify failure**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py::TestAgentScanCLI -v`

- [ ] **Step 8: Add handle_agent_scan() and format_scan_stats()** — Two functions in `core/agent_sdk/cli_integration.py`. See master plan Task 7, Step 3.

- [ ] **Step 9: Run tests**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add core/agent_sdk/cli_integration.py tests/test_agent_sdk_cli_integration.py
git commit -m "feat: add agent scan CLI command"
```

---

## Wave 3: Controller Wiring (Task 6)

**Gate:** Wave 2 WS-D (context builder changes) must be complete.

### Workstream E: Tool Registry + Controller (Task 6)

**Dependencies:** WS-C (querier), WS-D (context builder)
**Files:** `core/agent_sdk/tool_registry.py`, `core/agent_sdk/controller.py`, `tests/test_agent_sdk_tool_registry.py`

- [ ] **Step 1: Write failing tests** — Append `TestQueryCodebaseTool` (3 tests) to `tests/test_agent_sdk_tool_registry.py`. See master plan Task 6, Step 1.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_agent_sdk_tool_registry.py::TestQueryCodebaseTool -v`

- [ ] **Step 3: Add _handle_query_codebase and tool definition** — In `tool_registry.py`:
  1. Add `_handle_query_codebase(args, *, semantic_querier=None, **_)` function
  2. Add `query_codebase` entry to tool definitions list
  3. Add `semantic_querier` to `create_aura_tools` signature and deps

See master plan Task 6, Step 3.

- [ ] **Step 4: Update controller.py** — 3 changes:
  1. In `__init__`, create querier and store as `self._semantic_querier`, pass to `ContextBuilder`
  2. Update `_build_prompt` to use `self.context_builder`
  3. Pass `semantic_querier` to `create_aura_tools` in `_build_mcp_server`

See master plan Task 6, Step 4.

- [ ] **Step 5: Run ALL tool_registry and controller tests**

Run: `python3 -m pytest tests/test_agent_sdk_tool_registry.py tests/test_agent_sdk_controller.py tests/test_agent_sdk_controller_v2.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add core/agent_sdk/tool_registry.py core/agent_sdk/controller.py tests/test_agent_sdk_tool_registry.py
git commit -m "feat: add query_codebase tool and wire semantic querier into controller"
```

---

## Wave 4: End-to-End Verification (Task 8)

**Gate:** All previous waves complete.

### Workstream F: Integration Test (Task 8)

**Dependencies:** All workstreams (WS-A through WS-E)
**Files:** `tests/test_semantic_integration.py`

- [ ] **Step 1: Write integration test** — Create `tests/test_semantic_integration.py` with `TestSemanticEndToEnd` (2 tests: scan_then_query, context_builder_with_semantic_index). See master plan Task 8, Step 1.

- [ ] **Step 2: Run integration test**

Run: `python3 -m pytest tests/test_semantic_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run complete test suite**

Run: `python3 -m pytest tests/test_agent_sdk_*.py tests/test_semantic_*.py tests/integration/test_agent_sdk_integration.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_semantic_integration.py
git commit -m "test: add semantic scanner end-to-end integration tests"
```

---

## SADD Execution DAG

```
Wave 1 (parallel):
  WS-A: Config Fields (Task 1)        ──┐
  WS-B: AST Scanner (Task 3)          ──┼── Gate 1
  WS-C: Semantic Querier (Task 4)     ──┘
                                         │
Wave 2 (after Gate 1):                   │
  WS-D: Context Builder + CLI (T5+T7) ──┼── Gate 2
                                         │
Wave 3 (after Gate 2):                   │
  WS-E: Tool Registry + Controller (T6)─┤── Gate 3
                                         │
Wave 4 (after Gate 3):                   │
  WS-F: Integration Test (Task 8)     ──┘── Done
```

**Parallelism:** Wave 1 runs 3 workstreams concurrently. Wave 2 is a single workstream. Waves 3-4 are sequential. Total critical path: 4 waves.

## n8n Workflow Mapping

| Wave | n8n Trigger | Workstreams | Auto-gate |
|------|-------------|-------------|-----------|
| 1 | Manual or `agent run` | WS-A, WS-B, WS-C | pytest pass on all 3 |
| 2 | Wave 1 gate pass | WS-D | pytest pass |
| 3 | Wave 2 gate pass | WS-E | pytest pass |
| 4 | Wave 3 gate pass | WS-F | full suite pass |

Use `sadd run --spec docs/superpowers/plans/2026-04-02-semantic-scanner-execution.md` to execute via SADD, or dispatch manually via n8n WF-0 (dispatcher).
