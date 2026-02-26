# AURA CLI — Roadmap PRD Series
**Status**: Active Planning Document  
**Synthesized from**: 4 sessions, 10 checkpoints, 619 tests, 30,193 lines of Python  
**Current baseline**: 619 tests passing, R1–R3 performance optimizations complete

---

## Index

| # | Title | Priority | Scope | Dependency |
|---|-------|----------|-------|------------|
| [PRD-001](#prd-001-r4-agent-recall-optimization) | R4 Agent Recall Optimization | 🔴 P0 | 9 files, ~60 lines | None |
| [PRD-002](#prd-002-ascm-v2--semantic-context-manager) | ASCM v2 — Semantic Context Manager | 🔴 P0 | New module + wiring | PRD-001 |
| [PRD-003](#prd-003-autonomous-learning-loop) | Autonomous Learning Loop | 🟠 P1 | 6 core files | PRD-001 |
| [PRD-004](#prd-004-full-test-coverage--quality) | Full Test Coverage & Quality | 🟠 P1 | 110 untested files | None |
| [PRD-005](#prd-005-aura-studio--tui--observability) | AURA Studio — TUI & Observability | 🟡 P2 | New aura_cli/tui/ | PRD-003 |

---

## PRD-001: R4 Agent Recall Optimization

### Problem

R1–R3 optimizations fixed `Brain` itself. But **14 call sites across 9 files** still call `brain.recall_all()` which loads all 30,419 entries (7.9MB) into Python on every agent invocation:

```
agents/coder.py:51      — in f-string prompt (embeds entire DB into every prompt!)
agents/critic.py:45,82  — same pattern, called twice per cycle
agents/scaffolder.py:87 — same
agents/tester.py:46     — same
agents/ingest.py:33     — uses MemoryTier.SESSION (API mismatch, silently loads all)
agents/router.py:222    — reversed(brain.recall_all()) to find router stats
core/evolution_loop.py:52 — "\n".join(brain.recall_all()) — worst offender
core/closed_loop.py:29  — len(brain.recall_all()) — just counting
tools/aura_control_mcp.py:249,319,427 — 3 HTTP handlers, no inter-request cache
```

`core/evolution_loop.py:52` generates a prompt containing ALL 30,419 memory entries as a single newline-joined string — the entire 7.9MB database embedded in one LLM call. This is both a performance catastrophe and a token budget violation.

### Goals

1. Replace every `recall_all()` prompt embed with `recall_with_budget(max_tokens=N)` or `recall_recent(limit=N)`
2. Replace counting patterns with efficient `COUNT(*)` SQL
3. Replace router stat lookup with a direct keyed query
4. Add per-request memory cache to MCP HTTP handlers
5. Fix `agents/ingest.py` API mismatch (`MemoryTier.SESSION` arg that Brain ignores)

### Success Metrics

- No agent prompt embeds more than 4,000 tokens of memory context
- `core/evolution_loop.py` prompt assembly: < 10ms (was ~57ms on 30k entries)
- Full cycle time with real Brain: < 2s memory overhead total
- All 14 call sites replaced; verified by grep in CI

### Implementation Plan

**File: `agents/coder.py:51`**
```python
# Before
{self.brain.recall_all()}
# After  
{self.brain.recall_with_budget(max_tokens=2000)}
```

**File: `agents/critic.py:45,82`** — same pattern, budget 1500 tokens each  
**File: `agents/scaffolder.py:87`** — same, budget 2000  
**File: `agents/tester.py:46`** — same, budget 1500

**File: `agents/ingest.py:33`**
```python
# Before — wrong API, MemoryTier.SESSION silently ignored
memory_entries = self.brain.recall_all(MemoryTier.SESSION)
# After
memory_entries = self.brain.recall_recent(limit=50)
```

**File: `agents/router.py:222`**
```python
# Before — scans ALL entries reversed to find __router_stats__
for entry in reversed(self.brain.recall_all()):
# After — keyed lookup (router stats stored with known prefix)
entries = [e for e in self.brain.recall_recent(limit=200) 
           if e.startswith("__router_stats__")]
```

**File: `core/evolution_loop.py:52`**
```python
# Before — 7.9MB string, catastrophic
memory_snapshot = "\n".join(self.brain.recall_all())
# After
memory_snapshot = self.brain.recall_with_budget(max_tokens=3000)
```

**File: `core/closed_loop.py:29`**
```python
# Before — loads 30k rows to count them
memory_count = len(self.brain.recall_all())
# After — direct SQL COUNT
memory_count = self.brain.count_memories()
```

**File: `tools/aura_control_mcp.py`** — add module-level `_memory_cache` with 30s TTL:
```python
_memory_cache: Dict[str, Tuple[Any, float]] = {}

def _get_memories_cached(brain, ttl=30.0):
    now = time.time()
    if "memories" in _memory_cache and now - _memory_cache["memories"][1] < ttl:
        return _memory_cache["memories"][0]
    result = brain.recall_with_budget(max_tokens=4000)
    _memory_cache["memories"] = (result, now)
    return result
```

**New method: `Brain.count_memories()`**
```python
def count_memories(self) -> int:
    row = self.db.execute("SELECT COUNT(*) FROM memory").fetchone()
    return row[0] if row else 0
```

### New Test: `tests/test_optimization_r4.py`
- 30 tests covering all 9 files
- Regression: no file calls `recall_all()` in a prompt context
- Performance: evolution_loop memory assembly < 10ms
- Correctness: agent prompts contain memory context < 4000 tokens

### Acceptance Criteria
- [ ] `grep -r "recall_all()" agents/ core/evolution_loop.py` returns 0 results
- [ ] All 619 existing tests still pass
- [ ] R4 test suite: 30/30 passing

---

## PRD-002: ASCM v2 — Semantic Context Manager

### Problem

The existing `VectorStore` (333L) and `ContextManager` (150L) have critical production gaps documented in `docs/PRD_ASCM_V2.md`:

1. **No provenance** — retrieved snippets have no source file/line reference; agents hallucinate attribution
2. **No metadata filtering** — can't filter by `goal_id`, `source_type`, `agent_name`, or `tags`
3. **Fixed-count retrieval** — "top 5 results" ignores token budget; causes prompt overflow on large files
4. **No embedding model versioning** — changing models causes dimension mismatch, corrupts DB silently
5. **No deduplication** — same content re-embedded multiple times, bloating the vector store
6. **No evaluation harness** — retrieval quality unmeasured, regressions undetectable

### Goals

1. Implement `MemoryRecord` dataclass with full metadata schema
2. Implement `EmbeddingProvider` protocol + `LocalEmbeddingProvider` (offline/fallback)
3. Implement `VectorStoreV2` with SQLite backend, model versioning, and filter support
4. Implement token-aware `ContextBudgetManager` replacing fixed-count retrieval
5. Implement retrieval quality evaluation harness with recall@k metrics
6. Wire ASCM v2 into `IngestAgent` and `LoopOrchestrator`

### Architecture

```
IngestAgent
    └── ASCM v2
         ├── EmbeddingProvider (protocol)
         │   ├── OpenAIEmbeddingProvider (online)
         │   └── LocalEmbeddingProvider (sentence-transformers, offline)
         ├── VectorStoreV2 (memory/vector_store_v2.py)
         │   └── SQLite: memory_records + embeddings tables
         ├── ContextBudgetManager (core/context_budget.py)
         │   └── Greedy fill by similarity score * importance
         └── RetrievalEvaluator (tests/eval_ascm.py)
             └── recall@k, precision@k, MRR
```

### Data Model

```python
# memory/vector_store_v2.py

@dataclass
class MemoryRecord:
    id: str                          # UUID
    content: str                     # The actual text
    source_type: str                 # 'file' | 'memory' | 'goal' | 'output' | 'skill'
    source_ref: str                  # 'core/orchestrator.py:45' or 'goal:fix-tests'
    created_at: float                # Unix timestamp
    goal_id: Optional[str] = None   # Which goal produced this
    agent_name: Optional[str] = None # Which agent stored it
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0         # 0.0–2.0; default 1.0
    token_count: int = 0            # Pre-computed
    embedding_model: str = ""       # 'text-embedding-3-small' | 'local'
    content_hash: str = ""          # SHA256 for deduplication

@dataclass  
class RetrievalQuery:
    query_text: str
    budget_tokens: int = 4000
    min_score: float = 0.65
    filters: Dict[str, Any] = field(default_factory=dict)  # e.g. {"source_type": "file"}
    recency_bias: float = 0.1       # 0=pure semantic, 1=pure recency
    dedupe: bool = True

@dataclass
class SearchHit:
    record: MemoryRecord
    score: float                    # Cosine similarity 0.0–1.0
    explanation: str                # 'semantic:0.87 + recency_bias:0.05'
```

### SQL Schema (VectorStoreV2)

```sql
CREATE TABLE memory_records (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '',
    goal_id TEXT,
    agent_name TEXT,
    tags TEXT DEFAULT '[]',       -- JSON array
    importance REAL DEFAULT 1.0,
    token_count INTEGER DEFAULT 0,
    embedding_model TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX idx_records_goal ON memory_records(goal_id);
CREATE INDEX idx_records_source ON memory_records(source_type);
CREATE INDEX idx_records_hash ON memory_records(content_hash);
CREATE INDEX idx_records_created ON memory_records(created_at DESC);

CREATE TABLE embeddings (
    record_id TEXT PRIMARY KEY REFERENCES memory_records(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dims INTEGER NOT NULL,
    data BLOB NOT NULL             -- numpy float32 array, little-endian
);
```

### EmbeddingProvider Protocol

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]) -> List[np.ndarray]: ...
    def model_id(self) -> str: ...
    def dimensions(self) -> int: ...
    def available(self) -> bool: ...

class LocalEmbeddingProvider:
    """Uses TF-IDF + SVD for offline embedding. No external API needed."""
    # Installed via: pip install scikit-learn (already in environment)
    # 50-dim dense vectors; fast, no API calls, works on Termux
    
class OpenAIEmbeddingProvider:
    """Uses text-embedding-3-small via OpenRouter."""
    # Falls back to LocalEmbeddingProvider if API unavailable
```

### ContextBudgetManager

```python
class ContextBudgetManager:
    def assemble(
        self,
        hits: List[SearchHit],
        budget_tokens: int,
        format: str = "markdown"  # or "plain" or "json"
    ) -> str:
        """
        Greedy fill: sort by score * importance, add until token budget exhausted.
        Returns formatted context string with source attribution.
        """
```

### Migration Path from VectorStoreV1

```python
class VectorStoreV2:
    def migrate_from_v1(self, v1_store: VectorStore) -> int:
        """Import all v1 embeddings, assign source_type='legacy', return count."""
```

### Wire-in Points

1. `agents/ingest.py` — replace `brain.recall_all()` with `ascm.retrieve(query, budget=3000)`
2. `core/orchestrator.py._retrieve_hints()` — use ASCM for hint assembly
3. `aura_cli/cli_main.py.create_runtime()` — instantiate ASCM v2, attach to orchestrator

### New Files

```
memory/vector_store_v2.py      — VectorStoreV2, MemoryRecord, SearchHit
memory/embedding_provider.py   — EmbeddingProvider protocol + both implementations
core/context_budget.py         — ContextBudgetManager
tests/test_ascm_v2.py          — 40+ tests
tests/eval_ascm.py             — Retrieval quality evaluation harness
```

### Acceptance Criteria
- [ ] `VectorStoreV2` stores and retrieves with metadata filters
- [ ] `LocalEmbeddingProvider` works fully offline (no API key required)
- [ ] Deduplication: indexing same content twice stores 1 record
- [ ] Token budget respected: assembled context never exceeds budget
- [ ] Migration: 100% of v1 records imported with `source_type='legacy'`
- [ ] Retrieval recall@5 >= 0.80 on synthetic test set
- [ ] All 40+ tests passing

---

## PRD-003: Autonomous Learning Loop

### Problem

AURA has the *skeleton* of a learning system but **zero actual feedback-driven adaptation**:

- `ConvergenceEscapeLoop` detects oscillation but doesn't learn which strategies worked
- `AdaptivePipeline` chooses intensity but forgets between sessions  
- `AutonomousDiscovery` scans for missing tests but doesn't enqueue them
- `EvolutionLoop` exists but isn't wired into `LoopOrchestrator`
- `RouterAgent` tracks EMA scores but the persistence key (`__router_stats__`) is scanned via `recall_all()` (broken after 30k entries)
- No mechanism to measure: "did applying this cycle's changes actually improve the codebase?"

The system processes goals but doesn't **learn from outcomes**. Every cycle starts from the same prior.

### Goals

1. **Outcome measurement** — after each cycle, run a lightweight quality snapshot (test count, lint score, coverage %, cyclomatic complexity) and store delta
2. **Strategy memory** — `AdaptivePipeline` persists winning strategies to Brain keyed by `goal_type:context_hash`
3. **Fix `AutonomousDiscovery` wiring** — scan results auto-enqueue as goals with priority
4. **Fix `RouterAgent` persistence** — direct keyed lookup, not `recall_all()` scan
5. **Deprecate `HybridClosedLoop`** — migrate all callers to `LoopOrchestrator`, add deprecation warning
6. **Wire `EvolutionLoop` into `LoopOrchestrator`** as phase 8: `evolve`
7. **Implement `CycleOutcome` dataclass** — structured result of each cycle for learning

### CycleOutcome Schema

```python
@dataclass
class CycleOutcome:
    cycle_id: str                    # UUID
    goal: str
    goal_type: str                   # from classify_goal()
    started_at: float
    completed_at: float
    phases_completed: List[str]      # which phases ran
    changes_applied: int             # files modified
    tests_before: int
    tests_after: int
    tests_delta: int                 # positive = added tests
    lint_score_before: float         # 0-10
    lint_score_after: float
    strategy_used: str               # AdaptivePipeline strategy name
    success: bool
    failure_phase: Optional[str]     # if failed, which phase
    failure_reason: Optional[str]
    brain_entries_added: int
```

### Feedback Loop Architecture

```
run_cycle()
    ├── [phases 1-7: existing pipeline]
    ├── Phase 8: measure()          ← NEW
    │   └── run_quality_snapshot() → CycleOutcome
    ├── Phase 9: learn()            ← NEW
    │   ├── AdaptivePipeline.record_outcome(outcome)
    │   ├── RouterAgent.update_ema(model, success)
    │   └── Brain.remember(f"outcome:{cycle_id}", outcome.to_json())
    └── Phase 10: discover()        ← NEW (was AutonomousDiscovery.on_cycle_complete)
        └── scan findings → GoalQueue.batch_add(new_goals)
```

### `AdaptivePipeline` Persistence Fix

```python
class AdaptivePipeline:
    def record_outcome(self, goal_type: str, strategy: str, success: bool):
        """Store win/loss record for this (goal_type, strategy) pair."""
        key = f"__strategy_stats__:{goal_type}:{strategy}"
        existing = self._load_stats(key) or {"wins": 0, "losses": 0}
        if success:
            existing["wins"] += 1
        else:
            existing["losses"] += 1
        self.brain.remember(key, json.dumps(existing))
    
    def _choose_intensity(self, goal_type: str, consecutive_fails: int) -> str:
        """Now uses persisted win rates instead of just consecutive_fails."""
```

### `RouterAgent` Fix

```python
# Before (broken for 30k+ entries)
for entry in reversed(self.brain.recall_all()):
    if "__router_stats__" in entry:

# After (O(1) keyed lookup)  
raw = next((e for e in self.brain.recall_recent(limit=50) 
            if e.startswith("__router_stats__")), None)
```

### `AutonomousDiscovery` Wiring

```python
# In LoopOrchestrator.run_cycle() — new discover phase
discovery = AutonomousDiscovery(brain=self.brain, git_tools=self.git_tools)
findings = discovery.run_scan()
if findings.get("suggestions"):
    new_goals = [f["suggested_goal"] for f in findings["suggestions"]]
    self.goal_queue.batch_add(new_goals[:3])  # cap at 3 per cycle
```

### Quality Snapshot (lightweight, < 500ms)

```python
def run_quality_snapshot(project_root: Path) -> Dict:
    """Fast quality metrics — runs after every cycle."""
    import subprocess
    # 1. Test count: grep -r "def test_" tests/ | wc -l  (~10ms)
    # 2. Syntax errors: python3 -m py_compile on changed files (~50ms)
    # 3. Import health: try importing each changed module (~100ms)
    # NOT: full pytest run (too slow for every cycle)
```

### HybridClosedLoop Deprecation

```python
class HybridClosedLoop:
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "HybridClosedLoop is deprecated. Use LoopOrchestrator instead.",
            DeprecationWarning, stacklevel=2
        )
        # ... existing __init__
```

### New Files

```
core/cycle_outcome.py          — CycleOutcome dataclass + serialization
core/quality_snapshot.py       — run_quality_snapshot() 
tests/test_learning_loop.py    — 35+ tests
tests/test_cycle_outcome.py    — 20+ tests
```

### Acceptance Criteria
- [ ] After each cycle, `CycleOutcome` written to Brain
- [ ] `AdaptivePipeline.win_rate(goal_type, strategy)` returns correct ratio from persisted data
- [ ] `RouterAgent._load_stats()` uses keyed lookup, not `recall_all()`
- [ ] `AutonomousDiscovery` findings auto-enqueue (end-to-end test)
- [ ] `HybridClosedLoop` issues `DeprecationWarning` on instantiation
- [ ] Zero calls to `recall_all()` in learning loop components
- [ ] All 55+ new tests passing, all 619 existing tests still passing

---

## PRD-004: Full Test Coverage & Quality

### Problem

**110 source files have zero test coverage.** This includes every core agent:

```
agents/coder.py     agents/critic.py    agents/planner.py
agents/verifier.py  agents/reflector.py agents/ingest.py
agents/router.py    agents/mutator.py   agents/scaffolder.py
agents/tester.py    agents/base.py      agents/applicator.py
agents/registry.py
```

And all 28 skills (only integration-level tested, no unit tests):
```
agents/skills/adaptive_strategy_selector.py
agents/skills/api_contract_validator.py
... (26 more)
```

And core infrastructure:
```
core/workflow_engine.py   (879L, 8 classes — most complex file, zero dedicated tests)
core/context_graph.py     
core/health_monitor.py
core/explain.py
aura_cli/server.py        (HTTP API — only basic endpoint tests)
```

### Goals

1. **Agent unit tests** — every agent gets: instantiation, `run()` with mock adapter, error handling, output schema validation
2. **Skill unit tests** — each of 28 skills: `run()` with minimal input, error path returns `{"error": ...}` not raises, output has expected keys
3. **WorkflowEngine tests** — 8 classes, 33 functions → comprehensive test suite
4. **Server API tests** — auth failure, SSE streaming, `/metrics` format, `/health` format
5. **Integration tests** — full cycle with mock LLM; verify all 7 phases run, output saved

### Test Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| `agents/*.py` (13 files) | ~0% | ≥70% |
| `agents/skills/*.py` (28 files) | ~30% (integration) | ≥80% |
| `core/workflow_engine.py` | 0% | ≥75% |
| `core/orchestrator.py` | ~15% | ≥65% |
| `aura_cli/server.py` | ~25% | ≥70% |
| **Overall** | ~35% | ≥65% |

### Agent Test Pattern (replicable for all 13 agents)

```python
# tests/test_agents.py — all 13 agents in one file

class TestCoderAgent(unittest.TestCase):
    def setUp(self):
        self.mock_adapter = MagicMock()
        self.mock_brain = MagicMock()
        self.mock_brain.recall_with_budget.return_value = "context"
        self.agent = CoderAgent(self.mock_adapter, self.mock_brain)
    
    def test_instantiation(self):
        self.assertIsNotNone(self.agent)
    
    def test_run_returns_code_block(self):
        self.mock_adapter.respond.return_value = "# AURA_TARGET: foo.py\ndef foo(): pass"
        result = self.agent.run({"task": "write foo", "plan": []})
        self.assertIn("file_path", result)
    
    def test_run_handles_adapter_failure(self):
        self.mock_adapter.respond.side_effect = Exception("API down")
        result = self.agent.run({"task": "write foo"})
        # Should not raise; should return error dict
        self.assertIsInstance(result, dict)
    
    def test_output_schema_valid(self):
        from core.schema import validate_phase_output
        self.mock_adapter.respond.return_value = "# AURA_TARGET: foo.py\ncode"
        result = self.agent.run({"task": "x"})
        # Should not raise
        validate_phase_output("act", result)
```

### WorkflowEngine Test Coverage

`core/workflow_engine.py` has 8 classes and 33 functions — no dedicated tests exist. Priorities:

```python
# tests/test_workflow_engine_comprehensive.py

class TestRetryPolicy(TestCase): ...
class TestWorkflowStep(TestCase): ...
class TestWorkflowDefinition(TestCase): ...
class TestWorkflowEngine_Define(TestCase): ...
class TestWorkflowEngine_Run(TestCase): ...
class TestWorkflowEngine_Pause(TestCase): ...
class TestAgenticLoop(TestCase): ...  # The loop primitive
class TestWorkflowEngine_Integration(TestCase): ...  # define → run → pause → resume
```

### Skill Test Matrix

Each skill needs exactly 5 tests:
1. `test_{skill}_run_minimal_input` — smallest valid input, check output has expected keys
2. `test_{skill}_run_returns_dict` — output is always dict
3. `test_{skill}_never_raises` — bad input returns `{"error": "..."}` not exception
4. `test_{skill}_error_key_on_bad_input` — `"error"` key present when input invalid
5. `test_{skill}_run_with_project_root` — if skill takes `project_root`, test with real tmpdir

### New Test Files

```
tests/test_agents.py                   — 65+ tests (13 agents × 5)
tests/test_workflow_engine_full.py     — 45+ tests (8 classes × 5+)
tests/test_skills_comprehensive.py    — 140+ tests (28 skills × 5)
tests/test_server_comprehensive.py    — 30+ tests (auth, SSE, endpoints)
tests/test_orchestrator_phases.py     — 35+ tests (10 phases individually)
```

Total new tests: ~315

### Acceptance Criteria
- [ ] `python3 -m pytest tests/test_agents.py` — all passing
- [ ] Every skill file: at least 5 tests, `never_raises` test green
- [ ] `WorkflowEngine`: ≥ 75% line coverage measured with `coverage.py`
- [ ] CI total test count ≥ 930 (619 + 315 new)

---

## PRD-005: AURA Studio — TUI & Observability

### Problem

AURA's developer experience is terminal-only argparse. Monitoring a running cycle requires reading raw JSON logs. There is no way to:

- Watch the 7-phase pipeline progress in real-time
- See which goals are queued, in-progress, or completed
- Browse memory entries interactively
- View skill execution history
- Visualize performance metrics over time

### Goals

1. **Live cycle monitor** — real-time phase progress display using `rich` library
2. **Goal queue dashboard** — interactive list of pending/active/done goals
3. **Memory browser** — paginated view of Brain entries with search
4. **Metrics dashboard** — cycle times, success rates, skill stats over time
5. **AURA Doctor enhancement** — extend `aura_cli/doctor.py` with all checks from `aura_doctor.py` (root) into one canonical tool
6. **Log streamer** — tail structured JSON logs with colorized output

### Architecture

```
aura_cli/tui/
├── __init__.py
├── app.py              — Main TUI application (rich Live display)
├── panels/
│   ├── cycle_panel.py  — Phase progress (8 phases with spinners)
│   ├── queue_panel.py  — GoalQueue live view
│   ├── memory_panel.py — Brain entries browser
│   └── metrics_panel.py — Performance charts (sparklines)
└── log_streamer.py     — JSON log → rich colorized output
```

### CLI Integration

```bash
# New commands:
aura watch          # Launch TUI dashboard (requires `rich`)
aura logs [--tail N] [--level debug|info|warn|error]
aura queue [list|add|clear]
aura memory [search QUERY] [--limit N]
aura doctor         # Health check (consolidated)
aura metrics        # Print cycle stats table
```

### Rich TUI Layout

```
┌─ AURA Studio ────────────────────────────────────────────────┐
│ Goal: "Fix test failures in test_skills.py"    [Cycle #42]   │
├──────────────────────┬───────────────────────────────────────┤
│ Pipeline             │ Goal Queue                            │
│ ✅ ingest    0.12s   │ ▶ Fix test failures (ACTIVE)          │
│ ✅ plan      1.43s   │ · Add docstrings to router.py         │
│ ✅ critique  0.89s   │ · Refactor config_manager.py          │
│ ✅ synthesize 0.34s  │ · (8 more...)                         │
│ ⟳ act       3.21s…  ├───────────────────────────────────────┤
│ ○ verify             │ Brain Memory: 30,419 entries           │
│ ○ reflect            │ Last write: 0.8s ago                  │
├──────────────────────┤ Cache hit rate: 94.2%                 │
│ Recent Memories      ├───────────────────────────────────────┤
│ > Applied change to  │ Perf (last 10 cycles)                 │
│   tests/test_x.py    │ avg: 4.2s  p95: 8.1s  success: 80%  │
│ > Plan: 3 steps      │ ▁▂▃▂▄▅▃▄▅▄ (sparkline)              │
└──────────────────────┴───────────────────────────────────────┘
```

### Log Streamer

```bash
$ aura logs --tail 50 --level info
[10:15:33] INFO  ingest      context_built  tokens=1243 sources=7
[10:15:34] INFO  plan        steps_produced count=4
[10:15:35] WARN  act         old_code_miss  file=core/foo.py fallback=overwrite
[10:15:36] INFO  verify      passed         checks=8
```

### Consolidated Doctor

Merge `aura_doctor.py` (root, 180L) and `aura_cli/doctor.py` into one canonical `aura_cli/doctor.py`:

```
AURA Doctor v2 Checks:
  [✓] Config file valid JSON          aura.config.json
  [✓] API key configured              AURA_API_KEY
  [✓] OpenRouter reachable            https://openrouter.ai/api/v1
  [✓] Brain DB accessible             memory/brain.db (30,419 entries)
  [✓] Brain WAL mode                  PRAGMA journal_mode = wal
  [✓] Goal queue exists               memory/goal_queue.json (11 goals)
  [✓] MCP servers                     8001 (HTTP), 8002 (skills)
  [✗] rich not installed              pip install rich (for TUI)
  [✓] 619 tests passing               python3 -m pytest --tb=no -q
```

### Dependencies

```
rich>=13.0.0     # TUI, tables, spinners, progress bars — pip install rich
```
`rich` has no binary dependencies, works on Termux/Android.

### New Files

```
aura_cli/tui/__init__.py
aura_cli/tui/app.py
aura_cli/tui/panels/cycle_panel.py
aura_cli/tui/panels/queue_panel.py
aura_cli/tui/panels/memory_panel.py
aura_cli/tui/panels/metrics_panel.py
aura_cli/tui/log_streamer.py
tests/test_tui.py              — 25+ tests (no display, test data models)
```

### Acceptance Criteria
- [ ] `pip install rich && aura watch` launches TUI without errors
- [ ] `aura logs` streams colorized output from structured JSON logs
- [ ] `aura doctor` reports all 10 checks with pass/fail
- [ ] `aura queue list` shows current goal queue
- [ ] `aura memory search "keyword"` returns paginated results
- [ ] TUI works on Termux (no curses dependency, pure rich)

---

## Execution Order & Dependencies

```
PRD-001 (R4 Recall)
    ↓
PRD-002 (ASCM v2)  ←── can start after PRD-001
    ↓
PRD-003 (Learning) ←── requires PRD-001 for router fix
    ↓
PRD-004 (Tests)    ←── independent, can run in parallel with PRD-002/003
    ↓
PRD-005 (TUI)      ←── requires PRD-003 (needs CycleOutcome data to display)
```

### Estimated Test Count Progression

| After PRD | Tests |
|-----------|-------|
| Current baseline | 619 |
| + PRD-001 (R4) | ~649 |
| + PRD-002 (ASCM v2) | ~689 |
| + PRD-003 (Learning) | ~744 |
| + PRD-004 (Coverage) | ~1,059 |
| + PRD-005 (TUI) | ~1,084 |

### The Finished State

When all 5 PRDs are complete, AURA will be:

1. **Self-optimizing**: Every cycle measures its own quality delta and adapts strategy for next cycle
2. **Memory-efficient**: 205× faster memory retrieval, token-budget-aware context, deduplicated storage
3. **Semantically intelligent**: Provenance-tracked, metadata-filtered, token-aware context assembly
4. **Fully tested**: 1,084 tests, ≥65% line coverage across all modules
5. **Observable**: Real-time TUI dashboard, structured logs, health monitoring
6. **Autonomous**: `AutonomousDiscovery` continuously finds improvement opportunities, enqueues them as goals, the loop processes them and learns from outcomes — true recursive self-improvement

---

*Generated from: 4 sessions · 10 checkpoints · 619 tests · 30,193 LOC · 30,419 brain entries*
