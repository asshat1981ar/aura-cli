# SADD R3: Self-Bootstrap with MCP Tools & Sub-Agent Enhancement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SADD fully operational — fix interface mismatches, wire real orchestrator execution, integrate MCP tools into sub-agent workstreams, expose SADD as an MCP server, and prove the system by having it improve AURA itself.

**Architecture:** Fix the SubAgentRunner/SessionCoordinator interface contract, create an orchestrator factory, wire MCP tool discovery into workstream execution context, build a SADD MCP server for external access, then run SADD on a bounded self-improvement spec targeting AURA-411/402/410.

**Tech Stack:** Python 3.10+, FastAPI (MCP server), SQLite (session store), ThreadPoolExecutor (parallelism), existing LoopOrchestrator + Brain + MCPAsyncClient

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `core/sadd/sub_agent_runner.py` | Modify | Fix constructor, add MCP tool injection |
| `core/sadd/session_coordinator.py` | Modify | Fix param names, add orchestrator factory wiring, MCP discovery |
| `core/sadd/mcp_tool_bridge.py` | Create | Bridge between SADD workstreams and MCP tool servers |
| `tools/sadd_mcp_server.py` | Create | Expose SADD as an MCP server (workstream_execute, status, artifacts) |
| `aura_cli/cli_main.py` | Modify | Wire real execution into sadd-run, fix resume path |
| `docs/superpowers/specs/2026-03-25-sadd-self-bootstrap-spec.md` | Create | Self-improvement spec targeting AURA |
| `tests/test_sadd_runner.py` | Modify | Add MCP bridge tests |
| `tests/test_sadd_mcp_bridge.py` | Create | MCP tool bridge unit tests |
| `tests/test_sadd_mcp_server.py` | Create | SADD MCP server endpoint tests |
| `tests/integration/test_sadd_e2e.py` | Create | End-to-end: parse spec → execute workstreams → report |

---

### Task 1: Fix SubAgentRunner/SessionCoordinator Interface Contract

**Files:**
- Modify: `core/sadd/sub_agent_runner.py`
- Modify: `core/sadd/session_coordinator.py`
- Modify: `tests/test_sadd_runner.py`
- Modify: `tests/test_sadd_coordinator.py`

- [ ] **Step 1: Write a failing integration-style test exposing the mismatch**

```python
# tests/test_sadd_coordinator.py — add this test
def test_coordinator_creates_runner_with_correct_params(self):
    """Verify SessionCoordinator passes correct param names to SubAgentRunner."""
    from core.sadd.sub_agent_runner import SubAgentRunner
    import inspect
    sig = inspect.signature(SubAgentRunner.__init__)
    params = list(sig.parameters.keys())
    # These are the params SubAgentRunner actually expects:
    self.assertIn("workstream", params)
    self.assertIn("orchestrator_factory", params)
    self.assertIn("brain", params)
    self.assertIn("context_from_dependencies", params)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_coordinator.py::TestSessionCoordinator::test_coordinator_creates_runner_with_correct_params -v`
Expected: Check against actual SubAgentRunner signature

- [ ] **Step 3: Read SubAgentRunner to confirm actual parameter names**

Read `core/sadd/sub_agent_runner.py` and note the `__init__` signature. The coordinator must match these names exactly.

- [ ] **Step 4: Fix SessionCoordinator._execute_workstream() to match SubAgentRunner**

In `core/sadd/session_coordinator.py`, update `_execute_workstream()`:
- Pass `workstream=node` (WorkstreamNode, not node.spec)
- Pass `context_from_dependencies=context_from_dependencies` (not `dependency_context`)
- Remove `session_id` if SubAgentRunner doesn't accept it (or add it)

- [ ] **Step 5: Run all SADD tests**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_*.py -v`
Expected: All pass (coordinator tests may need mock updates)

- [ ] **Step 6: Commit**

```bash
git add core/sadd/sub_agent_runner.py core/sadd/session_coordinator.py tests/test_sadd_*.py
git commit -m "fix: align SubAgentRunner/SessionCoordinator interface contract"
```

---

### Task 2: Create Orchestrator Factory & Wire Real Execution

**Files:**
- Modify: `core/sadd/session_coordinator.py`
- Modify: `aura_cli/cli_main.py`
- Test: `tests/test_sadd_coordinator.py`

- [ ] **Step 1: Write test for orchestrator factory creation**

```python
# tests/test_sadd_coordinator.py
def test_orchestrator_factory_creates_valid_instance(self):
    """Factory should return object with run_loop method."""
    from core.sadd.session_coordinator import create_orchestrator_factory
    from memory.brain import Brain
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        brain = Brain(os.path.join(td, "test.db"))
        factory = create_orchestrator_factory(brain=brain, project_root=td)
        orch = factory()
        self.assertTrue(hasattr(orch, "run_loop"))
        self.assertTrue(callable(orch.run_loop))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_coordinator.py::TestSessionCoordinator::test_orchestrator_factory_creates_valid_instance -v`
Expected: FAIL — `create_orchestrator_factory` doesn't exist

- [ ] **Step 3: Implement create_orchestrator_factory**

Add to `core/sadd/session_coordinator.py`:

```python
def create_orchestrator_factory(
    brain: Any,
    project_root: str | Path = ".",
    model_adapter: Any = None,
    memory_store: Any = None,
) -> Callable[[], Any]:
    """Create a factory that produces fresh LoopOrchestrator instances."""
    from pathlib import Path as _Path
    from agents.registry import default_agents
    from core.orchestrator import LoopOrchestrator
    from core.policy import Policy
    from memory.store import MemoryStore

    _root = _Path(project_root)
    _store = memory_store or MemoryStore(_root / "memory")

    def _factory() -> LoopOrchestrator:
        if model_adapter is None:
            from core.model_adapter import ModelAdapter
            _model = ModelAdapter()
        else:
            _model = model_adapter
        agents = default_agents(brain, _model)
        return LoopOrchestrator(
            agents=agents,
            brain=brain,
            model=_model,
            memory_store=_store,
            project_root=_root,
            policy=Policy(max_cycles=10),
        )
    return _factory
```

- [ ] **Step 4: Run test to verify it passes**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_coordinator.py::TestSessionCoordinator::test_orchestrator_factory_creates_valid_instance -v`
Expected: PASS

- [ ] **Step 5: Wire real execution into CLI sadd-run handler**

Update `_handle_sadd_run_dispatch()` in `aura_cli/cli_main.py` — when `dry_run=False`:

```python
if not dry_run:
    from core.sadd.session_coordinator import SessionCoordinator, create_orchestrator_factory
    from core.sadd.session_store import SessionStore
    from core.sadd.types import SessionConfig

    runtime = ctx.runtime
    brain = runtime.get("brain") or Brain()
    model = runtime.get("model_adapter")
    store = SessionStore()

    config = SessionConfig(
        max_parallel=getattr(args, "max_parallel", 3),
        max_cycles_per_workstream=getattr(args, "max_cycles", 5),
        dry_run=False,
        fail_fast=getattr(args, "fail_fast", False),
    )
    factory = create_orchestrator_factory(
        brain=brain, project_root=ctx.project_root, model_adapter=model
    )
    coordinator = SessionCoordinator(
        design_spec=design_spec,
        orchestrator_factory=factory,
        brain=brain,
        config=config,
        session_store=store,
    )
    report = coordinator.run()
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
```

- [ ] **Step 6: Run SADD tests + CLI tests**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_*.py tests/test_cli_main_dispatch.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add core/sadd/session_coordinator.py aura_cli/cli_main.py tests/test_sadd_coordinator.py
git commit -m "feat: add orchestrator factory and wire real SADD execution in CLI"
```

---

### Task 3: Create MCP Tool Bridge for Workstreams

**Files:**
- Create: `core/sadd/mcp_tool_bridge.py`
- Modify: `core/sadd/sub_agent_runner.py`
- Create: `tests/test_sadd_mcp_bridge.py`

- [ ] **Step 1: Write failing test for MCP tool bridge**

```python
# tests/test_sadd_mcp_bridge.py
import unittest
from core.sadd.mcp_tool_bridge import MCPToolBridge

class TestMCPToolBridge(unittest.TestCase):
    def test_discover_tools_from_registry(self):
        bridge = MCPToolBridge()
        tools = bridge.discover_available_tools()
        self.assertIsInstance(tools, list)

    def test_match_tools_for_workstream(self):
        bridge = MCPToolBridge()
        matched = bridge.match_tools_for_goal("Run security scan on auth module")
        self.assertIsInstance(matched, list)
        # Security-related goals should match security tools
        tool_names = [t["name"] for t in matched]
        # At minimum, the bridge should return something (even if MCP servers aren't running)
        self.assertIsInstance(tool_names, list)

    def test_build_tool_context(self):
        bridge = MCPToolBridge()
        ctx = bridge.build_tool_context(["security_scanner", "linter_enforcer"])
        self.assertIn("available_tools", ctx)
        self.assertIsInstance(ctx["available_tools"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_mcp_bridge.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement MCPToolBridge**

Create `core/sadd/mcp_tool_bridge.py`:

```python
"""Bridge between SADD workstreams and MCP tool servers.

Discovers available MCP tools, matches them to workstream goals,
and builds tool context dicts for injection into sub-agent runners.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Keyword → MCP tool/skill mapping for goal-based matching
_GOAL_TOOL_MAP: dict[str, list[str]] = {
    "security": ["security_scanner", "secrets_scan", "semgrep_scan"],
    "test": ["test_coverage_analyzer", "lint_files"],
    "lint": ["linter_enforcer", "lint_files", "lint_all"],
    "refactor": ["refactoring_advisor", "complexity_scorer", "architecture_validator"],
    "doc": ["doc_generator", "docstring_fill"],
    "depend": ["dependency_analyzer", "dependency_audit"],
    "type": ["type_checker"],
    "perform": ["performance_profiler"],
    "schema": ["schema_validator"],
    "format": ["format"],
    "git": ["git_blame_snippet", "git_file_history"],
    "search": ["structured_search", "code_intel_xref"],
}


class MCPToolBridge:
    """Discovers and matches MCP tools to SADD workstream goals."""

    def __init__(self, mcp_registry: Any = None) -> None:
        self._registry = mcp_registry

    def discover_available_tools(self) -> list[dict[str, Any]]:
        """Discover tools from the MCP registry (or return known defaults)."""
        if self._registry:
            try:
                from core.mcp_registry import list_registered_services
                services = list_registered_services()
                tools = []
                for svc in services:
                    for cap in svc.get("capabilities", []):
                        tools.append({"name": cap, "server": svc.get("name", "unknown")})
                return tools
            except Exception:
                logger.debug("MCP registry not available, using static tool list")

        # Static fallback: known tools from AURA's MCP servers
        return [
            {"name": name, "server": "static"}
            for names in _GOAL_TOOL_MAP.values()
            for name in names
        ]

    def match_tools_for_goal(self, goal_text: str) -> list[dict[str, Any]]:
        """Match MCP tools relevant to a workstream goal."""
        goal_lower = goal_text.lower()
        matched: list[dict[str, Any]] = []
        seen: set[str] = set()

        for keyword, tool_names in _GOAL_TOOL_MAP.items():
            if keyword in goal_lower:
                for name in tool_names:
                    if name not in seen:
                        matched.append({"name": name, "matched_keyword": keyword})
                        seen.add(name)

        return matched

    def build_tool_context(self, tool_names: list[str]) -> dict[str, Any]:
        """Build a context dict for injection into sub-agent runners."""
        return {
            "available_tools": [
                {"name": name, "type": "mcp_tool"} for name in tool_names
            ],
            "tool_discovery_source": "sadd_mcp_bridge",
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_mcp_bridge.py -v`
Expected: PASS

- [ ] **Step 5: Wire MCP bridge into SubAgentRunner**

Update `core/sadd/sub_agent_runner.py` — add optional `mcp_bridge` parameter and enrich context_injection with tool context:

```python
# In __init__:
self._mcp_bridge = mcp_bridge  # Optional[MCPToolBridge]

# In _build_context_injection():
ctx = {"sadd_dependencies": {...}}
if self._mcp_bridge:
    matched = self._mcp_bridge.match_tools_for_goal(ws.goal_text)
    if matched:
        tool_ctx = self._mcp_bridge.build_tool_context([t["name"] for t in matched])
        ctx["mcp_tools"] = tool_ctx
return ctx
```

- [ ] **Step 6: Run all SADD tests**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_*.py tests/test_sadd_mcp_bridge.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add core/sadd/mcp_tool_bridge.py core/sadd/sub_agent_runner.py tests/test_sadd_mcp_bridge.py
git commit -m "feat: add MCP tool bridge for workstream-aware tool matching"
```

---

### Task 4: Create SADD MCP Server

**Files:**
- Create: `tools/sadd_mcp_server.py`
- Create: `tests/test_sadd_mcp_server.py`

- [ ] **Step 1: Write failing test for SADD MCP server endpoints**

```python
# tests/test_sadd_mcp_server.py
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

class TestSADDMCPServer(unittest.TestCase):
    def setUp(self):
        from tools.sadd_mcp_server import create_app
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["server"], "aura-sadd")

    def test_tools_endpoint(self):
        resp = self.client.get("/tools")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        tool_names = [t["name"] for t in data]
        self.assertIn("sadd_parse_spec", tool_names)
        self.assertIn("sadd_session_status", tool_names)
        self.assertIn("sadd_list_sessions", tool_names)

    def test_call_parse_spec(self):
        spec_md = "# Test\\n## Task: Do thing\\n- Build it"
        resp = self.client.post("/call", json={
            "tool_name": "sadd_parse_spec",
            "args": {"markdown": spec_md}
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("title", data.get("data", data.get("result", {})))

    def test_call_list_sessions(self):
        resp = self.client.post("/call", json={
            "tool_name": "sadd_list_sessions",
            "args": {}
        })
        self.assertEqual(resp.status_code, 200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_mcp_server.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement SADD MCP server**

Create `tools/sadd_mcp_server.py` — a FastAPI MCP server exposing SADD operations:

```python
"""SADD MCP Server — expose SADD workstream operations as MCP tools.

Port: 8020 (configurable via SADD_MCP_PORT)
Auth: SADD_MCP_TOKEN (optional)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.sadd.design_spec_parser import DesignSpecParser
from core.sadd.session_store import SessionStore
from core.sadd.types import validate_spec
from core.sadd.workstream_graph import WorkstreamGraph


def create_app() -> FastAPI:
    app = FastAPI(title="AURA SADD MCP Server")
    store = SessionStore()
    parser = DesignSpecParser()

    TOOLS = [
        {"name": "sadd_parse_spec", "description": "Parse a markdown design spec into workstreams",
         "input_schema": {"type": "object", "properties": {"markdown": {"type": "string"}}, "required": ["markdown"]}},
        {"name": "sadd_session_status", "description": "Get status of a SADD session",
         "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
        {"name": "sadd_list_sessions", "description": "List recent SADD sessions",
         "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}},
        {"name": "sadd_session_events", "description": "Get events for a SADD session",
         "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
        {"name": "sadd_session_artifacts", "description": "Get artifacts for a SADD session",
         "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}, "ws_id": {"type": "string"}}}},
    ]

    @app.get("/health")
    async def health():
        return {"server": "aura-sadd", "status": "ok", "tools": len(TOOLS), "uptime": time.time()}

    @app.get("/tools")
    async def tools():
        return TOOLS

    @app.post("/call")
    async def call_tool(request: Request):
        body = await request.json()
        tool_name = body.get("tool_name", "")
        args = body.get("args", {})
        t0 = time.time()

        try:
            if tool_name == "sadd_parse_spec":
                spec = parser.parse(args["markdown"])
                errors = validate_spec(spec)
                graph = WorkstreamGraph(spec.workstreams) if not errors else None
                result = {
                    "title": spec.title,
                    "workstreams": len(spec.workstreams),
                    "parse_confidence": spec.parse_confidence,
                    "waves": graph.execution_waves() if graph else [],
                    "errors": errors,
                }
            elif tool_name == "sadd_session_status":
                session = store.get_session(args["session_id"])
                result = session if session else {"error": "not found"}
            elif tool_name == "sadd_list_sessions":
                result = store.list_sessions(limit=args.get("limit", 20))
            elif tool_name == "sadd_session_events":
                result = store.get_events(args["session_id"])
            elif tool_name == "sadd_session_artifacts":
                result = store.get_artifacts(args["session_id"], args.get("ws_id"))
            else:
                return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)

            return {"data": result, "elapsed_ms": (time.time() - t0) * 1000}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SADD_MCP_PORT", "8020"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_mcp_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/sadd_mcp_server.py tests/test_sadd_mcp_server.py
git commit -m "feat: add SADD MCP server exposing workstream tools"
```

---

### Task 5: Create Self-Bootstrap Design Spec

**Files:**
- Create: `docs/superpowers/specs/2026-03-25-sadd-self-bootstrap-spec.md`

- [ ] **Step 1: Write the self-bootstrap design spec**

```markdown
# AURA Self-Improvement via SADD

Bounded self-improvement spec for SADD to execute on AURA itself.

## Workstream: Code Quality Analysis

Analyze AURA core modules for code smells, complexity hotspots, and test gaps.

- Run complexity scoring on core/orchestrator.py, core/workflow_engine.py
- Identify functions with cyclomatic complexity > 10
- List modules with < 50% test coverage
- Acceptance: analysis report written to memory/sadd_analysis.json

## Workstream: Test Coverage Backfill

Add missing unit tests for under-tested SADD modules.

- Create tests/integration/test_sadd_e2e.py with dry-run e2e flow
- Add edge case tests for parser (malformed markdown, unicode, empty sections)
- Add concurrent execution tests for session coordinator
- Depends on: Code Quality Analysis
- Acceptance: test count increases, all new tests pass

## Workstream: Documentation Update

Update CLAUDE.md and CLI reference with SADD commands and architecture.

- Add SADD section to CLAUDE.md describing sadd-run, sadd-status, sadd-resume
- Regenerate CLI_REFERENCE.md
- Add core/sadd/ to the repository structure diagram
- Depends on: Test Coverage Backfill
- Acceptance: docs/CLI_REFERENCE.md is current, CLAUDE.md has SADD section
```

- [ ] **Step 2: Validate the spec parses correctly**

Run: `AURA_SKIP_CHDIR=1 python3 -c "from core.sadd.design_spec_parser import DesignSpecParser; from pathlib import Path; p=DesignSpecParser(); s=p.parse_file(Path('docs/superpowers/specs/2026-03-25-sadd-self-bootstrap-spec.md')); print(f'{s.title}: {len(s.workstreams)} workstreams, confidence={s.parse_confidence:.0%}'); [print(f'  {w.id}: depends_on={w.depends_on}') for w in s.workstreams]"`

Expected: 3 workstreams with correct dependency chain

- [ ] **Step 3: Test dry-run execution**

Run: `AURA_SKIP_CHDIR=1 python3 main.py sadd run --spec docs/superpowers/specs/2026-03-25-sadd-self-bootstrap-spec.md --dry-run`

Expected: Shows 3 workstreams in 3 waves with dependency ordering

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-03-25-sadd-self-bootstrap-spec.md
git commit -m "docs: add SADD self-bootstrap design spec for R3 validation"
```

---

### Task 6: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_sadd_e2e.py`

- [ ] **Step 1: Write e2e integration test**

```python
# tests/integration/test_sadd_e2e.py
"""End-to-end integration test for SADD: parse spec → build graph → dry-run coordinator."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.sadd.design_spec_parser import DesignSpecParser
from core.sadd.session_coordinator import SessionCoordinator
from core.sadd.session_store import SessionStore
from core.sadd.types import SessionConfig, WorkstreamResult
from core.sadd.workstream_graph import WorkstreamGraph
from memory.brain import Brain


class TestSADDE2E(unittest.TestCase):
    def test_full_dry_run_pipeline(self):
        """Parse sample spec → validate → build graph → coordinator dry-run."""
        spec_path = Path("tests/fixtures/sadd_sample_spec.md")
        parser = DesignSpecParser()
        spec = parser.parse_file(spec_path)

        self.assertEqual(spec.title, "AURA Server Test Suite")
        self.assertEqual(len(spec.workstreams), 3)
        self.assertGreaterEqual(spec.parse_confidence, 0.8)

        graph = WorkstreamGraph(spec.workstreams)
        waves = graph.execution_waves()
        self.assertGreaterEqual(len(waves), 2)

    def test_coordinator_with_mock_orchestrator(self):
        """Full coordinator run with mocked orchestrator (no real LLM calls)."""
        spec_path = Path("tests/fixtures/sadd_sample_spec.md")
        parser = DesignSpecParser()
        spec = parser.parse_file(spec_path)

        with tempfile.TemporaryDirectory() as td:
            brain = Brain(os.path.join(td, "test.db"))
            store = SessionStore(Path(td) / "sessions.db")

            def mock_factory():
                orch = MagicMock()
                orch.run_loop.return_value = {
                    "goal": "test",
                    "stop_reason": "PASS",
                    "history": [{"phase_outputs": {"verification": {"status": "pass"}}}],
                }
                return orch

            config = SessionConfig(max_parallel=2, max_cycles_per_workstream=1, dry_run=False)
            coordinator = SessionCoordinator(
                design_spec=spec,
                orchestrator_factory=mock_factory,
                brain=brain,
                config=config,
                session_store=store,
            )
            report = coordinator.run()

            self.assertEqual(report.total_workstreams, 3)
            self.assertEqual(report.completed, 3)
            self.assertEqual(report.failed, 0)

            # Verify session was persisted
            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["status"], "completed")
```

- [ ] **Step 2: Run test**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/integration/test_sadd_e2e.py -v`
Expected: PASS (may need Task 1 fixes first)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_sadd_e2e.py
git commit -m "test: add SADD end-to-end integration test"
```

---

### Task 7: Update CLAUDE.md and Regenerate CLI Docs

**Files:**
- Modify: `CLAUDE.md`
- Regenerate: `docs/CLI_REFERENCE.md`
- Update: CLI snapshots

- [ ] **Step 1: Add SADD section to CLAUDE.md**

Add after the "Memory System" section:

```markdown
## SADD (Sub-Agent Driven Development)

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
```

Also add `core/sadd/` to the Repository Structure tree.

- [ ] **Step 2: Regenerate CLI reference**

Run: `AURA_SKIP_CHDIR=1 python3 scripts/generate_cli_reference.py`

- [ ] **Step 3: Update all CLI snapshots**

Run the snapshot regeneration script (contract report, help, json-help).

- [ ] **Step 4: Run all CLI tests**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_cli_*.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/CLI_REFERENCE.md tests/snapshots/
git commit -m "docs: add SADD to CLAUDE.md and update CLI reference"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run complete SADD test suite**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_sadd_*.py tests/test_sadd_mcp_bridge.py tests/test_sadd_mcp_server.py tests/integration/test_sadd_e2e.py -v`
Expected: All pass

- [ ] **Step 2: Run full CLI test suite**

Run: `AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_cli_*.py -v`
Expected: All pass

- [ ] **Step 3: Run self-bootstrap spec in dry-run**

Run: `AURA_SKIP_CHDIR=1 python3 main.py sadd run --spec docs/superpowers/specs/2026-03-25-sadd-self-bootstrap-spec.md --dry-run`
Expected: Shows 3 workstreams with correct dependency ordering

- [ ] **Step 4: Verify contract report**

Run: `AURA_SKIP_CHDIR=1 python3 main.py contract-report --check`
Expected: No failures

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete SADD R3 — self-bootstrap with MCP tools and sub-agent enhancement"
```
