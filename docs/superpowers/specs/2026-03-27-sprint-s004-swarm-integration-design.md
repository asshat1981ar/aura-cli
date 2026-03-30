# Sprint S004: Hierarchical Swarm Coordinator Activation

**Date:** 2026-03-27  
**Status:** Approved (autonomous — revised per spec-reviewer + innovation-synthesizer)  
**Branch:** `feat/sprint-s004`

---

## Problem Statement

~630 LOC of production code forming a **Hierarchical Swarm Coordination** pipeline exists in the working tree but has never been committed or wired. This sprint activates it safely.

### File Manifest (in-scope)

| File | LOC | Purpose | Tracked? |
|------|-----|---------|---------|
| `core/swarm_models.py` | 147 | `AgentRole`, `SwarmTask`, `TaskResult`, `CycleLesson`, `CycleReport`, `PRGateDecision`, `SupervisorConfig` dataclasses | No |
| `agents/hierarchical_coordinator.py` | 227 | Architect→Coder→Tester pipeline; `async execute(task, context) → TaskResult` | No |
| `agents/sdlc_debugger.py` | 204 | 5-lens failure classifier (`REQUIREMENTS/DESIGN/IMPLEMENTATION/TESTING/OPERATIONS`) | No |
| `core/swarm_supervisor.py` | 52 | `install_swarm_runtime(orchestrator, registry)` wiring entry point | No |
| `memory/learning_loop.py` | 52 | `LessonStore` — JSONL persistence of `CycleLesson` records | No |
| `aura_cli/runtime_factory.py` | 352 | `RuntimeFactory.create()` with `_WeaknessRemediatorLoop` + `_ConvergenceEscapeLoop` hooks | No |
| `tests/test_swarm_supervisor.py` | ~80 | 4 existing tests for swarm contracts | No |

### Deferred to Sprint S005

- `aura_cli/mcp_swarm_bridge.py` — HTTP bridge to Node.js swarm MCP (ops complexity)
- `aura_cli/interactive_shell.py` + `aura_cli/entrypoint.py` — REPL shell (UX, not capability)
- `agents/skills/mcp_semantic_discovery.py` — requires bridge running

---

## Scope Decision (Innovation-Synthesizer Ranked Output)

**Queue now**: Hierarchical Coordinator activation (#1) + Swarm test expansion (#4)  
**Queue next** (S005): LessonStore injection into main orchestrator cycle loop  
**Defer**: MCP Bridge, Interactive Shell  

Rationale: The coordinator transforms every autonomous sprint's quality ceiling. 630 LOC built, ~40 LOC to wire. Highest ROI.

---

## Architecture

```
aura_cli/cli_main.py  (unchanged call signature)
  └── create_runtime()   [existing — kept, not replaced]
        │
        └── install_swarm_runtime(orchestrator, registry)   ← NEW wiring
              │  (called only when AURA_ENABLE_SWARM=1)
              │
              ▼
        HierarchicalCoordinator  (agents/hierarchical_coordinator.py)
          ├── SwarmTask → WorkerProtocol.execute(task, context) → TaskResult
          ├── SDLCDebuggerAgent.classify(error) → SDLCFinding (5 lenses)
          ├── CycleLesson → LessonStore (JSONL: memory/swarm_lessons.jsonl)
          └── PRGateDecision  [disabled by default — AURA_SWARM_PR_GATE=0]
```

**Key decision: `create_runtime()` is NOT replaced.** Instead, `install_swarm_runtime()` is called as a post-construction hook, gated by `AURA_ENABLE_SWARM=1` (defaults to `0`). This eliminates the highest-risk item from the sprint (call-site inventory, deprecation/dual-run period) while still delivering the capability.

---

## WorkerProtocol Behavioral Contract

```python
@runtime_checkable
class WorkerProtocol(Protocol):
    async def execute(self, task: SwarmTask, context: Dict[str, Any]) -> TaskResult:
        """Execute a task. Must not raise — return TaskResult(status="failed") on error."""
```

- **Timeout**: 120s per task (configurable via `SupervisorConfig.task_timeout_s`)
- **Failure contract**: workers return `TaskResult(status="failed", error=str(e))`; never raise
- **Discovery**: workers registered in `agents/registry.py::default_agents()` dict by role name

---

## LessonStore Backend

- **Format**: JSONL — one `CycleLesson` JSON object per line
- **Path**: `memory/swarm_lessons.jsonl` (gitignored)
- **Write failure**: log warning and continue — never abort cycle
- **Read**: `injectable_lessons(limit=5)` returns most recent 5 lessons
- **Tests**: use `tmp_path` fixture to avoid filesystem side-effects

---

## Async/Sync Bridge

`HierarchicalCoordinator.execute()` is `async`. The main `LoopOrchestrator` is sync. Bridge pattern:

```python
# In install_swarm_runtime() or the caller:
import asyncio
import concurrent.futures

def _run_coroutine_sync(coro):
    """Run a coroutine from a sync context, safe when a loop is already running."""
    try:
        asyncio.get_running_loop()
        # Already inside a running loop (e.g. pytest-asyncio): submit to a new thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly
        return asyncio.run(coro)
```

Note: `loop.run_until_complete()` is **not** used — it raises `RuntimeError: This event loop is already running` when called from inside a running loop. The `ThreadPoolExecutor` pattern avoids `nest_asyncio` as a dependency.

---

## PRGateDecision Safety Gate

`PRGateDecision` (push branch + open PR) is **disabled by default**.  
Enable via env: `AURA_SWARM_PR_GATE=1`.  
In `runtime_factory.py`: `supervisor_config.github_delivery_enabled = bool(int(os.getenv("AURA_SWARM_PR_GATE", "0")))`.  
Tests must assert `github_delivery_enabled` defaults to `False`.

---

## Testing Strategy

### Net-new tests this sprint (target: +20 tests)

| File | Tests | Coverage |
|------|-------|---------|
| `tests/test_swarm_supervisor.py` | 4 → 15 | SDLCDebugger all 5 lenses (parametrized), PRGateDecision boundary, lesson injection at cycle 5/10 vs 1-4, dependency chain ordering, `github_delivery_enabled` default |
| `tests/test_swarm_models.py` | 0 → 8 | Dataclass validation, `CycleLesson` field contracts, `SupervisorConfig` defaults |
| `tests/test_runtime_factory.py` | 0 → 8 | `RuntimeFactory.create()` smoke, `AURA_ENABLE_SWARM=0` skips wiring, weakness loop fires every 5 cycles, convergence hook fires every cycle |

### `swarm run --dry-run` expected behavior
Exit code: `0`  
Stdout contains: `"SWARM"` header line + workstream summary + `"Mode: dry-run"`  
No files written, no GitHub API calls, `PRGateDecision` never evaluated  
Verified by: `test_swarm_run_dry_run_stdout` in `tests/test_swarm_supervisor.py`

---

## Safety Considerations

1. **Feature flag**: `AURA_ENABLE_SWARM=1` (default `0`) — no behavior change without opt-in
2. **PR gate off by default**: `AURA_SWARM_PR_GATE=0` — no GitHub API calls in CI or default runs
3. **Coordinator errors**: `try/except Exception as e: log_json("ERROR", "swarm_error", error=str(e))` — never propagate
4. **LessonStore write failure**: log warning, `missing_ok=True` on JSONL append
5. **Async bridge**: use `asyncio.run()` only when no event loop is running; guard against `nest_asyncio` scenario

---

## Eval Rubric (same as S001–S003)

| Dimension | Weight | Pass threshold |
|-----------|--------|---------------|
| correctness | 0.40 | All new tests pass; dry-run output matches spec |
| safety | 0.25 | Feature flags correct; no bare `except`; PR gate defaults off |
| completeness | 0.25 | All 5 in-scope files committed + tested; +20 net-new tests |
| style | 0.10 | `log_json` not `print()` in core/; type hints; docstrings |

Overall threshold: **≥ 0.85**

---

## Acceptance Criteria

- [ ] 6 untracked files committed: `core/swarm_models.py`, `agents/hierarchical_coordinator.py`, `agents/sdlc_debugger.py`, `core/swarm_supervisor.py`, `memory/learning_loop.py`, `aura_cli/runtime_factory.py`
- [ ] `tests/test_swarm_supervisor.py` expanded from 4 → 15 tests, all pass
- [ ] `tests/test_swarm_models.py` created with 8 tests
- [ ] `tests/test_runtime_factory.py` created with 8 tests
- [ ] `+20 net-new tests` added this sprint (not total suite count)
- [ ] `AURA_ENABLE_SWARM=0` (default): existing `goal run` behavior unchanged
- [ ] `AURA_SWARM_PR_GATE` defaults to `False` in `SupervisorConfig`
- [ ] `aura swarm run --dry-run` exits 0, prints `SWARM` header + `Mode: dry-run`, writes no files, makes no GitHub API calls
- [ ] Code-review eval score ≥ 0.85
- [ ] CHANGELOG updated; PR open on `feat/sprint-s004`

---

## Out of Scope (Sprint S005)

- `aura_cli/mcp_swarm_bridge.py` HTTP bridge (Node.js ops complexity)
- `aura_cli/interactive_shell.py` + `aura_cli/entrypoint.py`
- `agents/skills/mcp_semantic_discovery.py`
- LessonStore injection into main `LoopOrchestrator` cycle loop (#2 from innovation ranking)
