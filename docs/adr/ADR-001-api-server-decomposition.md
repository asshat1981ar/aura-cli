# ADR-001: api\_server.py Decomposition Plan

**Date:** 2025-01-01  
**Status:** Proposed  
**Deciders:** AURA Core Team  
**Tags:** architecture, fastapi, maintainability

---

## Context

`aura_cli/server.py` has grown to **703 lines** and mixes at least five distinct concerns in a single flat module:

| Concern | Current lines | Description |
|---|---|---|
| FastAPI app creation & lifespan | 65–81 | `lifespan()`, `app = FastAPI(...)` |
| Auth / security | 516–523 | `require_auth()` — Bearer header parsing, constant-time token comparison |
| Run-tool execution + SSE streaming | 178–513 | Shell subprocess lifecycle, output clamping, denylist, audit logging, SSE helpers, `_execute_*` handlers |
| Discovery / agent metadata routes | 545–581 | `/tools`, `/discovery`, `/environments`, `/architecture` |
| Health & metrics routes | 526–542 | `/health`, `/metrics` |
| n8n webhook / goal routes | 598–703 | `/webhook/goal`, `/webhook/status/{goal_id}`, `/webhook/plan-review` |
| Pydantic request models | 84–104 | `ExecuteRequest`, `WebhookGoalRequest`, `WebhookPlanReviewRequest` |
| Runtime state management | 106–163 | `_ensure_runtime_initialized()`, `_resolve_runtime_component()`, etc. |
| Module-level env var constants | 37–62 | `RUN_TOOL_TIMEOUT_S`, `RUN_TOOL_ENV_ALLOWLIST`, `RUN_TOOL_DENYLIST` |

Symptoms of this monolith:
- Any change to auth logic forces a diff that touches lines co-located with subprocess I/O code.
- Unit-testing `require_auth` in isolation requires importing the entire server, dragging in FastAPI, runtime state, and subprocess helpers.
- Developers onboarding to the codebase report difficulty locating which code handles a specific route.
- pytest runs import the whole file; a syntax error anywhere blocks all tests.

---

## Decision

Split `aura_cli/server.py` into a proper **`aura_cli/api/`** package. Every new file must stay **under 200 lines**.

### Proposed package layout

```
aura_cli/api/
├── __init__.py          # re-export `app` for uvicorn
├── app.py               # thin FastAPI composition root (<80 lines)
├── models.py            # Pydantic request/response models
├── runtime.py           # runtime state helpers (_ensure_*, _resolve_*, etc.)
├── middleware/
│   ├── __init__.py
│   └── auth.py          # JWT/Bearer verification
└── routers/
    ├── __init__.py
    ├── health.py         # /health, /ready, /metrics
    ├── runs.py           # /execute, /webhook/goal, /webhook/status, /webhook/plan-review
    ├── agents.py         # /tools, /discovery, /environments, /architecture
    └── ws.py             # /ws/* WebSocket handlers (future)
```

### File-by-file mapping

#### `aura_cli/api/app.py` — FastAPI composition root
**Source lines from server.py:** 1–81

Responsibilities:
- Import all routers and include them on the `FastAPI` instance.
- Define the `lifespan` context manager (lines 70–73) that calls `_ensure_runtime_initialized()`.
- Set `title`, `version`, `description`.
- Export a single `app` symbol.

Target length: ~70 lines.

```python
# aura_cli/api/app.py (sketch)
from contextlib import asynccontextmanager
from fastapi import FastAPI
from aura_cli.api.runtime import _ensure_runtime_initialized
from aura_cli.api.routers import health, runs, agents

@asynccontextmanager
async def lifespan(_: FastAPI):
    await _ensure_runtime_initialized()
    yield

app = FastAPI(title="AURA Dev Tools MCP", version="1.0.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(runs.router)
app.include_router(agents.router)
```

---

#### `aura_cli/api/models.py` — Pydantic models
**Source lines from server.py:** 84–104

- `ExecuteRequest` (line 84)
- `WebhookGoalRequest` (line 89)
- `WebhookPlanReviewRequest` (line 100)

No business logic, no imports from `core.*`—pure data contracts.

---

#### `aura_cli/api/runtime.py` — Runtime state management
**Source lines from server.py:** 106–163

Functions migrating here:
- `_current_project_root()` — line 106, reads `AURA_PROJECT_ROOT` env var
- `_apply_runtime_state()` — line 111
- `_ensure_runtime_initialized()` — line 119 (async, calls CLI entrypoint shim)
- `_resolve_runtime_component()` — line 136 (async, raises 503 on missing component)
- `_runtime_metrics_snapshot()` — line 151

Global mutable state (`runtime`, `orchestrator`, `model_adapter`, `memory_store`, `_runtime_init_error`) is encapsulated here behind accessor functions so routers never import module-level globals directly.

---

#### `aura_cli/api/middleware/auth.py` — JWT verification
**Source lines from server.py:** 516–523

```python
# aura_cli/api/middleware/auth.py
import os, secrets
from fastapi import Header, HTTPException
from typing import Optional

def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Validate Bearer token from AGENT_API_TOKEN env var.
    No-ops when AGENT_API_TOKEN is unset (open-access mode).
    """
    token = os.getenv("AGENT_API_TOKEN")
    if not token:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not secrets.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(status_code=403, detail="Invalid token")
```

This is the only file that reads `AGENT_API_TOKEN`. Future JWT verification (RS256 key rotation, expiry checks) adds here only.

---

#### `aura_cli/api/routers/runs.py` — Run tool + Webhook goal endpoints
**Source lines from server.py:** 166–513, 584–595, 598–703

Module-level constants migrating here:
- `RUN_TOOL_TIMEOUT_S`, `RUN_TOOL_MAX_OUTPUT_BYTES`, `RUN_TOOL_READ_CHUNK_BYTES` (lines 37–39)
- `RUN_TOOL_MAX_TIMEOUT_S`, `RUN_TOOL_MAX_OUTPUT_HARD_CAP`, `RUN_TOOL_MAX_READ_CHUNK_BYTES` (lines 40–42)
- `RUN_TOOL_ENV_ALLOWLIST` (lines 43–52)
- `RUN_TOOL_DENYLIST` (lines 53–62)

Internal helpers migrating here:
- `_clamped_run_tool_timeout_s()` — line 166
- `_clamped_run_tool_output_bytes()` — line 170
- `_clamped_run_tool_read_chunk_bytes()` — line 174
- `_is_denylisted_command()` — line 178
- `_log_run_tool_event()` — line 186
- `_persist_run_tool_audit()` — line 190
- `_sse_event()` — line 220
- `_run_tool_env()` — line 224
- `_enqueue_stream()` — line 228
- `_terminate_process()` — line 240
- `_close_process_transport()` — line 257

Execute handlers migrating here:
- `_execute_ask()` — line 267
- `_execute_env()` — line 274
- `_execute_run()` and its `run_generator` inner function — line 278 (~215 lines of streaming subprocess logic)
- `_execute_goal()` — line 496

Routes migrating here:
- `POST /execute` — line 584 (dispatches to `_execute_*` handlers)
- `POST /webhook/goal` — line 604
- `GET  /webhook/status/{goal_id}` — line 652
- `POST /webhook/plan-review` — line 675

> **Note:** The `_execute_run` function is itself ~215 lines. It should be extracted into a helper module `aura_cli/api/run_tool.py` in a follow-up ticket, keeping `routers/runs.py` under 200 lines.

In-memory goal queue (`_webhook_goal_queue`, line 601) moves to `runs.py` scope. A future Redis ADR (ADR-002) will replace this dict.

---

#### `aura_cli/api/routers/agents.py` — Discovery & tool-listing endpoints
**Source lines from server.py:** 545–581

Routes migrating here:
- `GET /tools` — line 545
- `GET /discovery` — line 558
- `GET /environments` — line 567
- `GET /architecture` — line 575

These are read-only MCP introspection endpoints with no shared state beyond `PROJECT_ROOT`.

---

#### `aura_cli/api/routers/health.py` — Health & metrics
**Source lines from server.py:** 526–542

Routes migrating here:
- `GET /health` — line 526 (no auth required)
- `GET /metrics` — line 537 (requires auth)

A future `/ready` endpoint (readiness probe for k8s) should be added here.

---

#### `aura_cli/api/routers/ws.py` — WebSocket handlers (placeholder)
No WebSocket routes exist in the current `server.py`. This file is scaffolded now as an empty placeholder so that the first WebSocket feature has a clear home without reopening the architecture debate.

---

## Consequences

### Positive
- Each file has a **single concern** and fits in one screen.
- `auth.py` can be unit-tested with zero FastAPI startup overhead (just call `require_auth` with a mock header).
- Adding a new route category never touches existing files.
- Testability: `pytest` can import `aura_cli.api.routers.health` without instantiating the orchestrator.
- Onboarding: a new developer can find auth logic in 30 seconds.

### Negative
- One-time migration risk: any caller that does `from aura_cli.server import app` must be updated (see Migration Plan).
- Circular import risk between `runtime.py` and routers — must be guarded with careful import ordering.

### Neutral
- `_execute_run` is still long after migration; a follow-up `run_tool.py` helper module should extract it.

---

## Migration Plan

### Phase 0 — Backward-compat shim (do first, keeps CI green)

Leave `aura_cli/server.py` in place as a **one-liner re-export**:

```python
# aura_cli/server.py  ← keep this file, replace contents
"""Backward-compatibility shim. New code should import from aura_cli.api."""
from aura_cli.api.app import app  # noqa: F401
```

All existing callers (`uvicorn aura_cli.server:app`, `.mcp.json`, tests) continue to work unmodified.

### Phase 1 — Create package skeleton

```
mkdir -p aura_cli/api/middleware aura_cli/api/routers
touch aura_cli/api/__init__.py
touch aura_cli/api/middleware/__init__.py
touch aura_cli/api/routers/__init__.py
```

### Phase 2 — Migrate in dependency order

1. `models.py` (no deps)
2. `runtime.py` (deps: `core.*` only)
3. `middleware/auth.py` (deps: stdlib only)
4. `routers/health.py` (deps: `runtime.py`, `middleware/auth.py`)
5. `routers/agents.py` (deps: `runtime.py`, `middleware/auth.py`, `core.*`)
6. `routers/runs.py` (deps: all of the above)
7. `app.py` (composes everything)
8. Update `server.py` to shim

### Phase 3 — Update call sites

```bash
grep -rn "from aura_cli.server import\|aura_cli.server:" . --include="*.py" --include="*.json" --include="*.md"
```

Update each hit to use `aura_cli.api.app` (or the specific sub-module).

### Phase 4 — Remove shim (after Sprint 1 stabilisation)

Delete `aura_cli/server.py` once no callers remain.

---

## Acceptance Criteria

- [ ] `pytest tests/` passes with zero regressions after migration.
- [ ] No single file in `aura_cli/api/` exceeds 200 lines (excluding blank lines and comments).
- [ ] `ruff check aura_cli/api/` returns no errors.
- [ ] `uvicorn aura_cli.server:app` continues to boot via the shim.
- [ ] CI import test: `python -c "from aura_cli.api.middleware.auth import require_auth"` succeeds.

---

## Related

- ADR-002: Redis Optionality Contract (replaces in-memory `_webhook_goal_queue` in `runs.py`)
- Issue: `aura_cli/server.py` exceeds 700 lines (monolith)
