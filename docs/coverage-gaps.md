# Coverage Gaps Analysis

Generated from: `pytest --cov=core --cov=agents --cov-report=json` against
`tests/test_file_tools.py`, `tests/test_agents_unit.py`, `tests/test_agents_sandbox.py`.

---

## Overall Coverage

| Metric | Value |
|---|---|
| **Total measured lines** | 24,490 |
| **Covered lines** | 2,114 |
| **Overall coverage** | **8.6%** |
| **Files < 80%** | 266 |
| **Files < 50%** | 252 |

> **Note:** Coverage is measured against the three existing test files that import
> the core modules. The vast majority of the `agents/` and `core/` sub-packages have
> *zero* imports in the current test suite — this inflates the 0 % count.

---

## Priority Focus: Key Agent & Core Modules

| Module | Coverage % | Covered / Total | Missing Lines |
|---|---|---|---|
| `core/orchestrator.py` | **0.0%** | 0 / 893 | 893 |
| `agents/planner.py` | **52.4%** | 54 / 103 | 49 |
| `core/file_tools.py` | **55.5%** | 111 / 200 | 89 |
| `agents/coder.py` | **67.5%** | 77 / 114 | 37 |
| `agents/sandbox.py` | **71.4%** | 80 / 112 | 32 |
| `agents/reflector.py` | 80.3% | 49 / 61 | 12 |
| `agents/applicator.py` | 87.5% | 63 / 72 | 9 |

---

## Critical Gaps (< 50%)

These modules have zero or near-zero test coverage and represent the highest
deployment risk.

| Module | Coverage % | Missing Lines | Risk |
|---|---|---|---|
| `core/orchestrator.py` | 0.0% | 893 | 🔴 **CRITICAL** — entire orchestration loop untested |
| `agents/adversarial/agent.py` | 0.0% | 175 | 🔴 **CRITICAL** — adversarial critique path untested |
| `core/voting/engine.py` | 0.0% | 228 | 🔴 **CRITICAL** — consensus voting untested |
| `core/simulation/engine.py` | 0.0% | 136 | 🔴 HIGH |
| `agents/multi_agent_workflow.py` | 0.0% | 241 | 🔴 HIGH |
| `core/workflow_engine.py` | 0.0% | 447 | 🔴 HIGH |
| `core/task_handler.py` | 0.0% | 299 | 🔴 HIGH |
| `agents/planner.py` | 52.4% | 49 | 🟠 MEDIUM |
| `core/file_tools.py` | 55.5% | 89 | 🟠 MEDIUM |
| `agents/coder.py` | 67.5% | 37 | 🟡 LOW-MEDIUM |

### `core/orchestrator.py` — 0% (893 missing lines)

The entire orchestration loop — `run_cycle`, `_dispatch_task`, `run_simulation`,
`attach_enhanced_features` — is not exercised at all. This is the most critical
gap because it is the integration seam between every other module.

**Missing regions (sample):**
- Lines 21–119: class definition, `__init__`, `attach_enhanced_features`
- Lines 158–265: `run_cycle` main body
- Lines 411–537: `_dispatch_task`, async dispatch
- Lines 617–720: simulation phase orchestration
- Lines 1047–1990: advanced planning, reflection, RSI integration

### `agents/planner.py` — 52.4% (49 missing lines)

**Missing regions:**
- Lines 12–13: module-level imports / init guard
- Lines 34–65: `plan()` error-handling branches and `_build_prompt()`
- Lines 95–114: `_parse_response()` malformed-JSON path
- Lines 134–175: `backfill()` and `prioritize()` methods
- Lines 195–205: async teardown / cleanup

### `core/file_tools.py` — 55.5% (89 missing lines)

**Missing regions:**
- Lines 80–148: `patch_file()` — hunks, offset calculation, context matching
- Lines 157–193: `validate_patch()` — all validation branches
- Lines 221–234: `create_file()` with existing-file guard
- Lines 361–378: `list_files()` gitignore filtering
- Lines 398–490: `diff_files()`, `backup_file()`, `restore_backup()`

### `agents/coder.py` — 67.5% (37 missing lines)

**Missing regions:**
- Lines 12–13: import guard
- Lines 41–44: constructor error path
- Lines 64–82: `code()` LLM error-handling branches
- Lines 93–103: retry logic
- Lines 123–132: response-parsing edge cases
- Lines 152–180: multi-file edit path
- Lines 220–241: async cleanup / streaming fallback

### `agents/sandbox.py` — 71.4% (32 missing lines)

**Missing regions:**
- Lines 186–188: Docker sandbox teardown on error
- Lines 214–215: timeout enforcement branch
- Lines 250–263: subprocess result parsing edge cases
- Lines 294–330: `run_tests()` failure aggregation
- Lines 350: final cleanup guard

---

## Modules 50–80% Coverage (Need Attention)

| Module | Coverage % | Missing Lines |
|---|---|---|
| `agents/coder.py` | 67.5% | 37 |
| `agents/sandbox.py` | 71.4% | 32 |
| `agents/reflector.py` | 80.3% | 12 |
| `core/sanitizer.py` | 76.4% | 9 |

---

## Prioritized Fix List (Highest Risk First)

1. **`core/orchestrator.py`** — 0% → needs full unit-test scaffold
   - Blocking: any integration test of the aura loop
   - Strategy: mock all agent dependencies, test `run_cycle()` state machine

2. **`agents/planner.py`** — 52% → error branches untested
   - Quick wins: malformed LLM response, empty goal, retry exhaustion

3. **`core/file_tools.py`** — 55% → patch/diff/backup untested
   - Quick wins: `patch_file()` with golden input/output pairs

4. **`agents/coder.py`** — 67% → LLM error paths untested
   - Quick wins: mock `requests.post` to return 4xx, 5xx, malformed JSON

5. **`agents/sandbox.py`** — 71% → failure aggregation untested
   - Quick wins: subprocess timeout, non-zero exit code from test runner

6. **`agents/adversarial/agent.py`** — 0% → critique path never exercised
   - Strategy: unit test `_run_strategies()` with mock strategy registry

7. **`core/voting/engine.py`** — 0% → consensus never exercised
   - Strategy: unit test `_collect_votes()` + `_analyze_consensus()` with mock adapters

---

## Recommended Test Strategy per Module

### `core/orchestrator.py`
```python
# Mock all agent dependencies; test state machine transitions
@patch('core.orchestrator.PlannerAgent')
@patch('core.orchestrator.CoderAgent')
async def test_run_cycle_happy_path(mock_coder, mock_planner):
    orch = Orchestrator(config=mock_config)
    result = await orch.run_cycle(goal="add docstrings")
    assert result.status == "completed"
```

### `core/file_tools.py`
```python
# Use tmp_path to test real filesystem operations
def test_patch_file_applies_hunk(tmp_path):
    f = tmp_path / "foo.py"
    f.write_text("def old(): pass\n")
    patch_file(str(f), old="def old(): pass", new="def new(): pass")
    assert f.read_text() == "def new(): pass\n"
```

### `agents/planner.py`
```python
# Mock LLM call; test all response shapes
@patch('agents.planner.call_llm')
def test_plan_malformed_json(mock_llm):
    mock_llm.return_value = "not json"
    planner = PlannerAgent()
    result = planner.plan("some goal")
    assert result is None or result == []  # graceful degradation
```

### `agents/coder.py`
```python
@patch('requests.post')
def test_coder_handles_500(mock_post):
    mock_post.return_value.status_code = 500
    coder = CoderAgent()
    with pytest.raises(RuntimeError):
        coder.code("write a function")
```

---

## pyproject.toml Coverage Configuration

Current `pyproject.toml` has `fail_under = 40`. Once coverage improves past 80%,
add `--cov-fail-under=80` to `addopts`.

```toml
# TODO: Uncomment once overall coverage reaches 80%:
# --cov-fail-under=80
```
