# AURA CLI Codebase Audit & Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematically analyze every module in the AURA CLI codebase, identify and fix bugs, refactor oversized files, improve error handling, add missing tests, enhance UX, and bring the codebase to production quality.

**Architecture:** Five-phase approach: (1) Static analysis sweep using all 15 skills, (2) Critical bug fixes and error handling, (3) Refactor oversized modules, (4) Test coverage push to 40%, (5) UX and documentation improvements.

**Tech Stack:** Python 3.10+, pytest, Pydantic, rich (TUI), pathlib, structured logging

---

## Audit Summary (Evidence-Based)

| Metric | Current | Target |
|--------|---------|--------|
| Test coverage | 17.85% | 40%+ |
| Files >500 lines | 13 | 0 |
| Functions >50 lines | 181 | <20 |
| Broad exception handlers | 27+ | 0 |
| Type hint coverage (core/) | ~60% | 95%+ |
| Untested modules | 96 (57%) | <20% |
| Documentation gaps | 5 major | 0 |

---

## Phase 1: Static Analysis Sweep & Triage

### Task 1.1: Run Full Skill Battery

**Files:**
- Create: `reports/audit_2026-04-10.json`

- [ ] **Step 1: Run all 15 analysis skills via AURA**

```bash
AURA_SKIP_CHDIR=1 python3 main.py goal once \
  "Run full static analysis: symbol_indexer, architecture_validator, api_contract_validator, complexity_scorer, dependency_analyzer, linter_enforcer, type_checker, structural_analyzer, test_coverage_analyzer, code_clone_detector, security_scanner, tech_debt_quantifier, doc_generator, dockerfile_analyzer, performance_profiler" \
  --max-cycles 1 --dry-run --json > reports/audit_2026-04-10.json
```

- [ ] **Step 2: Run TechnicalDebtAgent hotspot analysis**

```bash
python3 -c "
from agents.technical_debt_agent import TechnicalDebtAgent
agent = TechnicalDebtAgent()
# Feed heatmap data from complexity scorer output
import json
with open('reports/audit_2026-04-10.json') as f:
    data = json.load(f)
hotspots = agent.prioritize_hotspots(data.get('skill_outputs', {}).get('complexity_scorer', {}).get('high_risk', []))
print(json.dumps(hotspots, indent=2))
"
```

- [ ] **Step 3: Run DuplicateCodeReducer analysis**

```bash
python3 -c "
from agents.code_refactor_agent import DuplicateCodeReducer
reducer = DuplicateCodeReducer(base_path='.')
dupes = reducer.analyze_codebase()
abstractions = reducer.propose_abstractions()
import json
print(json.dumps({'duplicates': len(dupes), 'abstractions': abstractions}, indent=2, default=str))
"
```

- [ ] **Step 4: Commit audit report**

```bash
git add reports/
git commit -m "chore: add full codebase audit report"
```

---

## Phase 2: Critical Bug Fixes & Error Handling

### Task 2.1: Replace Broad Exception Handlers (27 instances)

**Files:**
- Modify: `core/sadd/session_coordinator.py` (5 instances)
- Modify: `core/agent_sdk/semantic_scanner.py` (5 instances)
- Modify: `core/agent_sdk/context_builder.py` (2 instances)
- Modify: `core/agent_sdk/cli_integration.py` (3 instances)
- Modify: `core/agent_sdk/controller.py` (1 instance)
- Modify: `core/agent_sdk/resilience.py` (1 instance)
- Modify: `core/agent_sdk/feedback.py` (1 instance)
- Test: `tests/test_exception_handling.py`

- [ ] **Step 1: Write test for specific exception types**

```python
# tests/test_exception_handling.py
import pytest
from core.exceptions import FileToolsError, OldCodeNotFoundError, MismatchOverwriteBlockedError

def test_session_coordinator_catches_specific_exceptions():
    """Verify session coordinator doesn't swallow unexpected errors."""
    # Each broad except should be narrowed to expected exception types
    from core.sadd.session_coordinator import SessionCoordinator
    # Test that AttributeError propagates (not caught by broad handler)
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exception_handling.py -v`

- [ ] **Step 3: Narrow each broad handler to specific types**

For each file, replace `except Exception:` with specific types:
- `except (OSError, json.JSONDecodeError):` for file/parse operations
- `except (KeyError, AttributeError, TypeError):` for data access
- `except (ConnectionError, TimeoutError):` for network calls

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/ --no-cov -x -q --timeout=60`

- [ ] **Step 5: Commit**

```bash
git commit -m "fix: narrow 27 broad exception handlers to specific types"
```

### Task 2.2: Fix Config Path Resolution in SADD Orchestrators

**Files:**
- Modify: `core/orchestrator.py:439-447`
- Test: `tests/test_orchestrator_config.py`

- [ ] **Step 1: Write test for config loading**

```python
def test_load_config_file_with_absolute_path(tmp_path):
    """Config loads correctly when project_root is absolute."""
    config_file = tmp_path / "aura.config.json"
    config_file.write_text('{"n8n_connector": {"enabled": true}}')
    orch = LoopOrchestrator(agents={}, project_root=tmp_path)
    result = orch._load_config_file()
    assert result["n8n_connector"]["enabled"] is True
```

- [ ] **Step 2: Run test, verify fail**
- [ ] **Step 3: Add debug logging to _load_config_file**

```python
def _load_config_file(self) -> dict:
    config_path = self.project_root / "aura.config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log_json("WARN", "config_file_load_failed", details={"path": str(config_path), "error": str(exc)})
    return {}
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

### Task 2.3: Add Missing Type Hints to Critical Modules

**Files:**
- Modify: `core/policy.py` (0% → 100%)
- Modify: `core/policies/base.py` (0% → 100%)
- Modify: `core/logging_configuration.py` (0% → 100%)
- Modify: `core/performance_monitor.py` (0% → 100%)
- Modify: `agents/coder.py` (33% → 90%)
- Modify: `agents/technical_debt_agent.py` (37% → 100%)

- [ ] **Step 1: Add return types and parameter types to each module**
- [ ] **Step 2: Run mypy check**

```bash
python3 -m mypy core/policy.py core/policies/base.py --ignore-missing-imports
```

- [ ] **Step 3: Commit**

---

## Phase 3: Refactor Oversized Modules

### Task 3.1: Split `core/orchestrator.py` (2,007 → 5 files)

**Files:**
- Create: `core/orchestrator_phases.py` — plan/critique/synthesize/act phase methods
- Create: `core/orchestrator_verify.py` — verify/apply/restore logic
- Create: `core/orchestrator_learn.py` — reflect/learn/n8n feedback/quality trends
- Create: `core/orchestrator_capabilities.py` — MCP discovery, capability management
- Modify: `core/orchestrator.py` — slim to ~400 lines (init + run_cycle + run_loop)
- Test: All existing tests must pass unchanged

- [ ] **Step 1: Run baseline tests**

```bash
python3 -m pytest tests/ --no-cov -q --timeout=120 2>&1 | tail -5
```
Record pass count.

- [ ] **Step 2: Extract phase methods to `orchestrator_phases.py`**

Move `_execute_plan_critique_synthesize`, `_run_plan_loop`, `_run_act_loop`, `_execute_act_verify_attempt` into a mixin or standalone module. Import back into `LoopOrchestrator`.

- [ ] **Step 3: Run tests — must match baseline count**
- [ ] **Step 4: Extract verify/apply to `orchestrator_verify.py`**
- [ ] **Step 5: Run tests**
- [ ] **Step 6: Extract learn/reflect/n8n to `orchestrator_learn.py`**
- [ ] **Step 7: Run tests**
- [ ] **Step 8: Extract capability/MCP to `orchestrator_capabilities.py`**
- [ ] **Step 9: Run full test suite — same pass count as baseline**
- [ ] **Step 10: Commit**

```bash
git commit -m "refactor: split orchestrator.py (2007 lines) into 5 focused modules"
```

### Task 3.2: Split `core/model_adapter.py` (1,137 → 3 files)

**Files:**
- Create: `core/model_providers.py` — individual provider call methods (openai, gemini, anthropic, openrouter, local)
- Create: `core/model_cache.py` — caching logic (L0/L1/Momento)
- Modify: `core/model_adapter.py` — slim to ~300 lines (routing + respond + respond_for_role)

- [ ] **Step 1-5: Same extract-test-extract-test pattern as Task 3.1**
- [ ] **Step 6: Commit**

### Task 3.3: Decompose 10 Largest Functions (>100 lines)

**Files:**
- Modify: `agents/multi_agent_workflow.py::compile_summary` (225 lines)
- Modify: `core/evolution_loop.py::run` (197 lines)
- Modify: `core/orchestrator.py::run_cycle` (155 lines)
- Modify: `agents/skills/dockerfile_analyzer.py::_analyse` (156 lines)
- Modify: `core/orchestrator.py::_record_cycle_outcome` (145 lines)

For each function:
- [ ] **Step 1: Identify logical sections within the function**
- [ ] **Step 2: Extract each section into a named helper method**
- [ ] **Step 3: Run tests after each extraction**
- [ ] **Step 4: Commit**

---

## Phase 4: Test Coverage Push (17.85% → 40%)

### Task 4.1: Test Critical Untested Core Modules

**Files:**
- Create: `tests/test_model_adapter_providers.py`
- Create: `tests/test_evolution_loop.py`
- Create: `tests/test_async_orchestrator.py`
- Create: `tests/test_improvement_loop.py`
- Create: `tests/test_context_budget.py`
- Create: `tests/test_autonomous_discovery.py`
- Create: `tests/test_mcp_client.py`

For each module:
- [ ] **Step 1: Use TesterAgent to generate test skeletons**

```bash
AURA_SKIP_CHDIR=1 python3 -c "
from agents.tester import TesterAgent
from unittest.mock import MagicMock
tester = TesterAgent(brain=MagicMock(), model=MagicMock(), sandbox=MagicMock())
# Generate tests for each module
"
```

- [ ] **Step 2: Write focused unit tests for public API methods**
- [ ] **Step 3: Run and verify**
- [ ] **Step 4: Commit after each test file**

### Task 4.2: Add Integration Tests for SADD Pipeline

**Files:**
- Create: `tests/integration/test_sadd_full_pipeline.py`

- [ ] **Step 1: Write test for parse → graph → coordinate → report flow**
- [ ] **Step 2: Mock LLM calls, verify DAG execution order**
- [ ] **Step 3: Verify n8n webhook fires (mock the HTTP call)**
- [ ] **Step 4: Commit**

### Task 4.3: Add Snapshot Tests for New CLI Commands

**Files:**
- Create: `tests/snapshots/sadd_run_help.json`
- Create: `tests/snapshots/innovate_help.json`
- Create: `tests/snapshots/agent_help.json`

- [ ] **Step 1: Generate snapshots for each new command**
- [ ] **Step 2: Add to CI snapshot contract**
- [ ] **Step 3: Commit**

---

## Phase 5: UX & Documentation

### Task 5.1: Create Interactive `aura init` Wizard

**Files:**
- Create: `aura_cli/init_wizard.py`
- Modify: `aura_cli/options.py` — add `init` command
- Modify: `aura_cli/cli_main.py` — register dispatch
- Test: `tests/test_init_wizard.py`

- [ ] **Step 1: Write failing test**

```python
def test_init_creates_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from aura_cli.init_wizard import run_init
    run_init(project_root=tmp_path, interactive=False)
    config = json.loads((tmp_path / "aura.config.json").read_text())
    assert "model_routing" in config
```

- [ ] **Step 2: Implement wizard** — detect env vars, create minimal config, validate connectivity
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

### Task 5.2: Add Progress Indicators

**Files:**
- Modify: `core/orchestrator.py` — emit phase progress events
- Create: `aura_cli/progress.py` — rich-based progress display
- Modify: `aura_cli/dispatch.py` — attach progress display to goal/sadd runs

- [ ] **Step 1: Add phase timing events to orchestrator UI callbacks**
- [ ] **Step 2: Create rich progress bar that renders phase names**
- [ ] **Step 3: Wire into dispatch for `goal once` and `sadd run`**
- [ ] **Step 4: Test manually, commit**

### Task 5.3: Complete Getting-Started Documentation

**Files:**
- Modify: `docs/getting-started/installation.md`
- Create: `docs/getting-started/quickstart.md`
- Create: `docs/getting-started/configuration.md`
- Create: `docs/tutorials/sadd-tutorial.md`
- Create: `docs/tutorials/innovation-tutorial.md`
- Create: `aura.config.example.json`

- [ ] **Step 1: Write quickstart with 5-minute walkthrough**
- [ ] **Step 2: Create commented example config**
- [ ] **Step 3: Write SADD tutorial with sample design spec**
- [ ] **Step 4: Write Innovation Catalyst tutorial**
- [ ] **Step 5: Update installation docs with correct paths and providers**
- [ ] **Step 6: Commit**

### Task 5.4: Improve Error Messages

**Files:**
- Modify: `aura_cli/dispatch.py` — context-aware error messages
- Modify: `aura_cli/runtime_factory.py` — helpful provider setup hints
- Create: `aura_cli/error_hints.py` — remediation suggestions

- [ ] **Step 1: Catalog all `print(..., file=sys.stderr)` calls**
- [ ] **Step 2: Replace generic messages with actionable hints**

```python
# Before:
print(f"Error: {exc}", file=sys.stderr)

# After:
print(f"Error: {exc}\n  Hint: Run 'aura doctor' to diagnose, or check docs/getting-started/configuration.md", file=sys.stderr)
```

- [ ] **Step 3: Test error paths manually**
- [ ] **Step 4: Commit**

---

## Agents & Skills Used Per Phase

| Phase | Agents/Skills |
|-------|--------------|
| 1. Analysis | All 15 skills, TechnicalDebtAgent, DuplicateCodeReducer, InvestigationAgent |
| 2. Bug Fixes | DebuggerAgent, SelfCorrectionAgent, RootCauseAnalysisAgent |
| 3. Refactor | DuplicateCodeReducer, CodeSearchAgent, VerifierAgent |
| 4. Testing | TesterAgent, SandboxAgent, VerifierAgent |
| 5. UX | ScaffolderAgent, InnovationSwarm (for init wizard UX), MetaConductor |

## Estimated Scope

| Phase | Tasks | Est. Effort |
|-------|-------|-------------|
| 1. Analysis | 1 task | 30 min |
| 2. Bug Fixes | 3 tasks | 2-3 hours |
| 3. Refactor | 3 tasks | 4-6 hours |
| 4. Testing | 3 tasks | 4-6 hours |
| 5. UX/Docs | 4 tasks | 3-4 hours |
| **Total** | **14 tasks** | **~15-20 hours** |
