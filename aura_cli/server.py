from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator
import re
from starlette.middleware.base import BaseHTTPMiddleware

from aura_cli.api import state as _state
from aura_cli.api.runtime_bootstrap import (
    RuntimeBootstrapState,
    apply_runtime_state as _bootstrap_apply_runtime_state,
    current_project_root as _bootstrap_current_project_root,
    ensure_runtime_initialized as _bootstrap_ensure_runtime_initialized,
    load_model_config_status,
    resolve_runtime_component as _bootstrap_resolve_runtime_component,
    run_db_migrations as _bootstrap_run_db_migrations,
    runtime_metrics_snapshot as _bootstrap_runtime_metrics_snapshot,
)
from core.logging_utils import log_json
from core.mcp_contracts import (
    build_discovery_payload,
    build_tool_descriptor,
)
from core.mcp_registry import list_registered_services
from core.ai_environment_registry import list_ai_environments
from core.mcp_architecture import default_routing_profile
from core.operator_runtime import build_run_tool_audit_summary
from core.running_runs import register_run, deregister_run
from tools.mcp_auth import require_dev_tools_auth

# ---------------------------------------------------------------------------
# Prometheus metrics — prometheus_client is a required dependency.
# ---------------------------------------------------------------------------
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


def _get_or_create_metric(factory, name: str, documentation: str, labelnames=()):
    """Return an existing collector when this module is re-imported in tests."""
    try:
        return factory(name, documentation, labelnames)
    except ValueError:
        collector = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        if collector is None and not name.endswith("_total"):
            collector = REGISTRY._names_to_collectors.get(f"{name}_total")  # type: ignore[attr-defined]
        if collector is None:
            raise
        return collector


pipeline_runs_total = _get_or_create_metric(
    Counter,
    "aura_pipeline_runs",
    "Total pipeline runs",
    ["status"],
)
active_pipeline_runs = _get_or_create_metric(
    Gauge,
    "aura_active_pipeline_runs",
    "Currently executing pipelines",
)
goal_queue_depth = _get_or_create_metric(
    Gauge,
    "aura_goal_queue_depth",
    "Number of goals waiting in the webhook queue",
)
sandbox_violations_total = _get_or_create_metric(
    Counter,
    "aura_sandbox_violations",
    "Sandbox violations caught by the run-tool denylist",
    ["type"],
)
http_request_duration = _get_or_create_metric(
    Histogram,
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)
aura_request_latency_seconds = _get_or_create_metric(
    Histogram,
    "aura_request_latency_seconds",
    "AURA API request latency in seconds",
    ["endpoint"],
)

# Project root for environment listing
PROJECT_ROOT = Path.cwd()

# Global runtime state (set by CLI entrypoint or tests)
runtime: Dict[str, Any] = {}
orchestrator = None
model_adapter = None
memory_store = None
_runtime_init_error: str | None = None
_bootstrap_state = RuntimeBootstrapState()

RUN_TOOL_TIMEOUT_S = float(os.getenv("AURA_RUN_TOOL_TIMEOUT_S", "15"))
RUN_TOOL_MAX_OUTPUT_BYTES = int(os.getenv("AURA_RUN_TOOL_MAX_OUTPUT_BYTES", str(64 * 1024)))
RUN_TOOL_READ_CHUNK_BYTES = int(os.getenv("AURA_RUN_TOOL_READ_CHUNK_BYTES", "1024"))
RUN_TOOL_MAX_TIMEOUT_S = 60.0
RUN_TOOL_MAX_OUTPUT_HARD_CAP = 256 * 1024
RUN_TOOL_MAX_READ_CHUNK_BYTES = 8 * 1024
RUN_TOOL_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "TERM",
    "TMPDIR",
    "USER",
)
RUN_TOOL_DENYLIST = (
    "halt",
    "poweroff",
    "reboot",
    "shutdown",
    "rm -rf /",
    "rm -fr /",
    "mkfs",
    ":(){ :|:& };:",
)


def _beads_runtime_snapshot():
    """Helper for TUI/Studio metadata (used in SSE)."""
    return {"enabled": False, "required": False, "scope": "none"}


def _sync_runtime_state_from_exports() -> None:
    _bootstrap_state.runtime = runtime
    _bootstrap_state.orchestrator = orchestrator
    _bootstrap_state.model_adapter = model_adapter
    _bootstrap_state.memory_store = memory_store
    _bootstrap_state.runtime_init_error = _runtime_init_error


def _sync_runtime_exports() -> None:
    global runtime, orchestrator, model_adapter, memory_store, _runtime_init_error
    runtime = _bootstrap_state.runtime
    orchestrator = _bootstrap_state.orchestrator
    model_adapter = _bootstrap_state.model_adapter
    memory_store = _bootstrap_state.memory_store
    _runtime_init_error = _bootstrap_state.runtime_init_error


def _run_db_migrations() -> None:
    _bootstrap_run_db_migrations(log_json)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(_run_db_migrations)
    await _ensure_runtime_initialized()
    yield


app = FastAPI(
    title="AURA Dev Tools MCP",
    description="Main entry point for AURA developer tools and orchestration.",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.environ.get("AURA_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)

from aura_cli.middleware.rate_limit import rate_limit_middleware

app.middleware("http")(rate_limit_middleware)


# ---------------------------------------------------------------------------
# X-Request-ID correlation middleware
# ---------------------------------------------------------------------------
try:
    from core.correlation import CorrelationManager as _CorrelationManager

    _CORRELATION_AVAILABLE = True
except ImportError:
    _CORRELATION_AVAILABLE = False


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate or generate an X-Request-ID for every request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        if _CORRELATION_AVAILABLE:
            with _CorrelationManager.scope(request_id):
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Prometheus HTTP request duration middleware
# ---------------------------------------------------------------------------
_RATE_LIMIT = int(os.getenv("AURA_RATE_LIMIT", "1000"))


@app.middleware("http")
async def _prometheus_timing_middleware(request: Request, call_next):
    """Record HTTP request duration for every endpoint and add rate-limit headers."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    # Normalise path to avoid high-cardinality labels (strip UUIDs etc.)
    path = request.url.path
    http_request_duration.labels(endpoint=path).observe(duration)
    aura_request_latency_seconds.labels(endpoint=path).observe(duration)
    response.headers["X-RateLimit-Limit"] = str(_RATE_LIMIT)
    return response


class ExecuteRequest(BaseModel):
    tool_name: str
    args: List[Any] = []
    goal: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    max_cycles: int = Field(default=5, ge=1, le=20)
    dry_run: bool = Field(default=False)

    @field_validator("goal")
    @classmethod
    def validate_goal_content(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        suspicious = re.compile(
            r"(ignore previous|system prompt|<\|im_start\|>|###\s*instruction)",
            re.IGNORECASE,
        )
        if suspicious.search(v):
            raise ValueError("Goal contains disallowed patterns")
        return v.strip()


class WebhookGoalRequest(BaseModel):
    goal: str
    priority: int = 5
    dry_run: bool = False
    # metadata carries pipeline-routing context from n8n:
    # - complexity: "low" | "medium" | "high"
    # - pipeline_run_id: str (trace ID from P1/P3)
    # - quality_gate_critique: str (injected by P3 before AURA act phase)
    metadata: Dict[str, Any] = {}


class WebhookPlanReviewRequest(BaseModel):
    task_bundle: Dict[str, Any]
    goal: str
    pipeline_run_id: str = ""


def _current_project_root() -> Path:
    return _bootstrap_current_project_root(PROJECT_ROOT)


def _apply_runtime_state(runtime_state: Dict[str, Any]) -> None:
    _bootstrap_apply_runtime_state(_bootstrap_state, runtime_state, _state)
    _sync_runtime_exports()


async def _ensure_runtime_initialized() -> Dict[str, Any]:
    _sync_runtime_state_from_exports()
    from aura_cli.cli_main import create_runtime

    runtime_state = await _bootstrap_ensure_runtime_initialized(
        state=_bootstrap_state,
        project_root=_current_project_root(),
        create_runtime_func=create_runtime,
        shared_state_module=_state,
        log_json=log_json,
    )
    _sync_runtime_exports()
    return runtime_state


async def _resolve_runtime_component(name: str) -> Any:
    _sync_runtime_state_from_exports()
    result = await _bootstrap_resolve_runtime_component(
        state=_bootstrap_state,
        name=name,
        ensure_runtime_initialized_func=_ensure_runtime_initialized,
    )
    _sync_runtime_exports()
    return result


def _runtime_metrics_snapshot() -> Dict[str, Any]:
    _sync_runtime_state_from_exports()
    return _bootstrap_runtime_metrics_snapshot(
        state=_bootstrap_state,
        project_root=PROJECT_ROOT,
        list_registered_services_func=list_registered_services,
        list_ai_environments_func=list_ai_environments,
        build_run_tool_audit_summary_func=build_run_tool_audit_summary,
    )


def _clamped_run_tool_timeout_s() -> float:
    return max(1.0, min(float(RUN_TOOL_TIMEOUT_S), RUN_TOOL_MAX_TIMEOUT_S))


def _clamped_run_tool_output_bytes() -> int:
    return max(1024, min(int(RUN_TOOL_MAX_OUTPUT_BYTES), RUN_TOOL_MAX_OUTPUT_HARD_CAP))


def _clamped_run_tool_read_chunk_bytes() -> int:
    return max(128, min(int(RUN_TOOL_READ_CHUNK_BYTES), RUN_TOOL_MAX_READ_CHUNK_BYTES))


def _is_denylisted_command(command: str) -> str | None:
    normalized = " ".join(command.lower().split())
    for pattern in RUN_TOOL_DENYLIST:
        if pattern in normalized:
            return pattern
    return None


def _log_run_tool_event(event: str, *, command: str, **details: Any) -> None:
    log_json("INFO", event, details={"command": command[:512], **details})


def _persist_run_tool_audit(
    *,
    command: str,
    pid: int | None,
    code: int | None,
    timed_out: bool,
    truncated: bool,
    duration_s: float,
    output_bytes: int,
) -> None:
    if memory_store is None or not hasattr(memory_store, "append_log"):
        return

    entry = {
        "type": "server_run_tool",
        "timestamp": time.time(),
        "command": command[:512],
        "pid": pid,
        "code": code,
        "timed_out": timed_out,
        "truncated": truncated,
        "duration_s": duration_s,
        "output_bytes": output_bytes,
    }
    try:
        memory_store.append_log(entry)
    except Exception as exc:
        log_json("WARN", "aura_server_run_tool_audit_persist_failed", details={"error": str(exc)})


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _run_tool_env() -> Dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in RUN_TOOL_ENV_ALLOWLIST and value}


async def _enqueue_stream(stream: asyncio.StreamReader | None, event_type: str, queue: asyncio.Queue[tuple[str, bytes | None]]) -> None:
    if stream is None:
        await queue.put((event_type, None))
        return
    while True:
        chunk = await stream.read(_clamped_run_tool_read_chunk_bytes())
        if not chunk:
            break
        await queue.put((event_type, chunk))
    await queue.put((event_type, None))


async def _terminate_process(process: asyncio.subprocess.Process) -> int:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            return int(process.returncode or 0)
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return int(process.returncode or 0)
            await process.wait()
    return int(process.returncode or 0)


def _close_process_transport(process: asyncio.subprocess.Process) -> None:
    transport = getattr(process, "_transport", None)
    if transport is None:
        return
    try:
        transport.close()
    except Exception:
        pass


async def _execute_ask(req: ExecuteRequest):
    prompt = req.args[0] if req.args else ""
    adapter = await _resolve_runtime_component("model_adapter")
    res = adapter.respond(prompt)
    return {"status": "success", "data": res}


async def _execute_env(_: ExecuteRequest):
    raise HTTPException(status_code=501, detail="The 'env' tool is currently disabled due to security concerns.")


async def _execute_run(req: ExecuteRequest):
    if os.getenv("AGENT_API_ENABLE_RUN") != "1":
        raise HTTPException(status_code=403, detail="Run tool is disabled")
    if not req.args:
        raise HTTPException(status_code=400, detail="Missing command in args")
    command = str(req.args[0]).strip()
    if not command:
        raise HTTPException(status_code=400, detail="Missing command in args")
    blocked_pattern = _is_denylisted_command(command)
    if blocked_pattern is not None:
        sandbox_violations_total.labels(type="denylist").inc()
        raise HTTPException(status_code=403, detail=f"Command blocked by policy: {blocked_pattern}")

    async def run_generator():
        timeout_s = _clamped_run_tool_timeout_s()
        output_limit = _clamped_run_tool_output_bytes()
        started_at = asyncio.get_running_loop().time()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(PROJECT_ROOT),
            env=_run_tool_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _log_run_tool_event(
            "aura_server_run_tool_started",
            command=command,
            pid=process.pid,
            timeout_s=timeout_s,
            output_limit_bytes=output_limit,
        )
        yield _sse_event({"type": "start", "command": command, "pid": process.pid})

        queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()
        readers = [
            asyncio.create_task(_enqueue_stream(process.stdout, "stdout", queue)),
            asyncio.create_task(_enqueue_stream(process.stderr, "stderr", queue)),
        ]
        deadline = asyncio.get_running_loop().time() + timeout_s
        total_output = 0
        closed_streams = 0

        try:
            while closed_streams < len(readers):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": timeout_s})
                    yield _sse_event({"type": "exit", "code": code, "timed_out": True})
                    return

                try:
                    event_type, chunk = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": timeout_s})
                    yield _sse_event({"type": "exit", "code": code, "timed_out": True})
                    return
                if chunk is None:
                    closed_streams += 1
                    continue

                remaining_bytes = output_limit - total_output
                if remaining_bytes <= 0:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "truncated", "limit_bytes": output_limit})
                    yield _sse_event({"type": "exit", "code": code, "truncated": True})
                    return

                emitted = chunk[:remaining_bytes]
                total_output += len(emitted)
                text = emitted.decode("utf-8", errors="replace")
                if text:
                    yield _sse_event({"type": event_type, "data": text})

                if len(chunk) > len(emitted) or total_output >= output_limit:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "truncated", "limit_bytes": output_limit})
                    yield _sse_event({"type": "exit", "code": code, "truncated": True})
                    return

            remaining = max(0.1, deadline - asyncio.get_running_loop().time())
            try:
                code = await asyncio.wait_for(process.wait(), timeout=remaining)
                duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                _log_run_tool_event(
                    "aura_server_run_tool_finished",
                    command=command,
                    pid=process.pid,
                    code=int(code or 0),
                    timed_out=False,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                _persist_run_tool_audit(
                    command=command,
                    pid=process.pid,
                    code=int(code or 0),
                    timed_out=False,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                yield _sse_event({"type": "exit", "code": int(code or 0)})
            except asyncio.TimeoutError:
                code = await _terminate_process(process)
                duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                _log_run_tool_event(
                    "aura_server_run_tool_finished",
                    command=command,
                    pid=process.pid,
                    code=code,
                    timed_out=True,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                _persist_run_tool_audit(
                    command=command,
                    pid=process.pid,
                    code=code,
                    timed_out=True,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": timeout_s})
                yield _sse_event({"type": "exit", "code": code, "timed_out": True})
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            _close_process_transport(process)

    return StreamingResponse(run_generator(), media_type="text/event-stream")


async def _execute_goal(req: ExecuteRequest):
    if not req.args:
        raise HTTPException(status_code=400, detail="Missing goal text in args")
    active_orchestrator = await _resolve_runtime_component("orchestrator")

    async def goal_generator():
        yield f"data: {json.dumps({'type': 'start', 'goal': req.args[0]})}\n\n"
        await asyncio.sleep(0.1)

        health_info = {"type": "health", "status": "ok", "providers": {"openai": "connected", "openrouter": "connected", "gemini": "connected"}, "beads_runtime": _beads_runtime_snapshot()}
        yield f"data: {json.dumps(health_info)}\n\n"
        await asyncio.sleep(0.1)

        cycle_data = await asyncio.to_thread(active_orchestrator.run_cycle, req.args[0])
        yield f"data: {json.dumps({'type': 'cycle', 'summary': cycle_data.get('cycle_summary', cycle_data)})}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'status': 'success', 'stop_reason': cycle_data.get('stop_reason', 'done'), 'history': [cycle_data]})}\n\n"

    return StreamingResponse(goal_generator(), media_type="text/event-stream")


def require_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> None:
    require_dev_tools_auth(x_api_key, authorization)


@app.get("/health")
async def health() -> Dict:
    return {
        "status": "healthy",
        "version": "0.1.0",
        "providers": {
            "openai": "connected",
            "openrouter": "connected",
            "gemini": "connected",
        },
    }


@app.get("/ready")
async def ready():
    """Readiness probe: checks all critical and optional subsystems."""
    import socket
    import sqlite3 as _sqlite3
    import subprocess

    from fastapi.responses import JSONResponse

    overall_start = time.perf_counter()
    components: Dict[str, Any] = {}
    status = "ready"

    # --- brain_db ---
    brain_db_path = Path(os.environ.get("AURA_BRAIN_DB_PATH", "memory/brain.db"))
    t0 = time.perf_counter()
    try:
        conn = _sqlite3.connect(str(brain_db_path), timeout=2.0)
        conn.execute("SELECT 1")
        conn.close()
        components["brain_db"] = {"status": "ready", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    except Exception as exc:
        components["brain_db"] = {"status": "degraded", "error": str(exc), "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
        status = "degraded"

    # --- auth_db ---
    try:
        from core.auth import _default_auth_db_path  # noqa: PLC0415
    except ImportError:

        def _default_auth_db_path():
            return Path(os.environ.get("AURA_AUTH_DB_PATH", "aura_auth.db"))

    auth_db_path = str(_default_auth_db_path())
    t0 = time.perf_counter()
    try:
        conn = _sqlite3.connect(auth_db_path, timeout=2.0)
        conn.execute("SELECT 1")
        conn.close()
        components["auth_db"] = {"status": "ready", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    except Exception as exc:
        components["auth_db"] = {"status": "degraded", "error": str(exc), "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
        if status == "ready":
            status = "degraded"

    # --- redis (optional) ---
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        t0 = time.perf_counter()
        try:
            import redis as _redis  # noqa: PLC0415

            client = _redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1)
            client.ping()
            components["redis"] = {"status": "ready", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
        except Exception:
            components["redis"] = {"status": "unavailable", "latency_ms": None}
            if status == "ready":
                status = "degraded"
    else:
        components["redis"] = {"status": "unavailable", "latency_ms": None}

    # --- mcp_server (optional — unavailable is not a failure) ---
    t0 = time.perf_counter()
    try:
        sock = socket.create_connection(("localhost", 8001), timeout=1.0)
        sock.close()
        components["mcp_server"] = {"status": "ready", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    except (OSError, ConnectionRefusedError):
        components["mcp_server"] = {"status": "unavailable", "latency_ms": None}

    # --- model_config ---
    model_configured = load_model_config_status(PROJECT_ROOT)
    components["model_config"] = {"status": "configured" if model_configured else "unconfigured"}

    # --- sandbox ---
    t0 = time.perf_counter()
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            ["python3", "-c", "print('ok')"],
            capture_output=True,
            timeout=2.0,
        )
        sandbox_ok = proc.returncode == 0 and b"ok" in proc.stdout
        components["sandbox"] = {
            "status": "ready" if sandbox_ok else "degraded",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        if not sandbox_ok and status == "ready":
            status = "degraded"
    except Exception as exc:
        components["sandbox"] = {
            "status": "degraded",
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        if status == "ready":
            status = "degraded"

    overall_latency_ms = round((time.perf_counter() - overall_start) * 1000, 2)
    http_status = 200 if status in ("ready", "degraded") else 503
    return JSONResponse(
        {"status": status, "components": components, "overall_latency_ms": overall_latency_ms},
        status_code=http_status,
    )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Expose Prometheus-format metrics (no auth — scraper must be on private network)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/tools")
async def tools(_: None = Depends(require_auth)) -> Dict:
    return {
        "status": "success",
        "tools": [
            build_tool_descriptor("ask", "Ask AURA a question"),
            build_tool_descriptor("run", "Run a shell command"),
            build_tool_descriptor("env", "Get environment variables"),
            build_tool_descriptor("goal", "Execute an autonomous goal"),
        ],
    }


@app.get("/discovery")
async def discovery(_: None = Depends(require_auth)) -> Dict:
    return build_discovery_payload(
        current_server={"name": "aura-dev-tools", "port": 8001},
        servers=list_registered_services(),
        supported_environments=list_ai_environments(PROJECT_ROOT),
    )


@app.get("/environments")
async def environments(_: None = Depends(require_auth)) -> Dict:
    return {
        "status": "success",
        "environments": list_ai_environments(PROJECT_ROOT),
    }


@app.get("/architecture")
async def architecture(_: None = Depends(require_auth)) -> Dict:
    return {
        "routing": default_routing_profile(),
        "knowledge_backends": [{"name": "neo4j"}, {"name": "weaviate"}],
        "supported_environments": list_ai_environments(PROJECT_ROOT),
    }


@app.post("/execute")
async def execute(req: ExecuteRequest, _: None = Depends(require_auth)):
    handlers = {
        "ask": _execute_ask,
        "env": _execute_env,
        "run": _execute_run,
        "goal": _execute_goal,
    }
    handler = handlers.get(req.tool_name)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")
    return await handler(req)


# --- n8n pipeline integration webhooks ---

# In-memory goal queue for webhook-submitted goals
_webhook_goal_queue: Dict[str, Dict[str, Any]] = {}


@app.post("/webhook/goal")
async def webhook_goal(req: WebhookGoalRequest, _: None = Depends(require_auth)) -> Dict:
    """Enqueue a goal submitted by n8n (P1 fast lane or direct trigger)."""
    goal_id = secrets.token_hex(12)
    entry: Dict[str, Any] = {
        "goal_id": goal_id,
        "goal": req.goal,
        "priority": req.priority,
        "dry_run": req.dry_run,
        "metadata": req.metadata,
        "status": "queued",
        "queued_at": time.time(),
    }
    _webhook_goal_queue[goal_id] = entry
    goal_queue_depth.set(sum(1 for e in _webhook_goal_queue.values() if e["status"] == "queued"))
    log_json(
        "INFO",
        "aura_webhook_goal_received",
        details={
            "goal_id": goal_id,
            "goal": req.goal[:200],
            "pipeline_run_id": req.metadata.get("pipeline_run_id", ""),
            "complexity": req.metadata.get("complexity", "unknown"),
        },
    )

    # Fire-and-forget: run the cycle in background so n8n can poll /webhook/status
    async def _run_cycle() -> None:
        active_pipeline_runs.inc()
        goal_queue_depth.set(sum(1 for e in _webhook_goal_queue.values() if e["status"] == "queued"))
        try:
            _webhook_goal_queue[goal_id]["status"] = "running"
            _webhook_goal_queue[goal_id]["started_at"] = time.time()
            active_orchestrator = await _resolve_runtime_component("orchestrator")
            result = await asyncio.to_thread(
                active_orchestrator.run_cycle,
                req.goal,
            )
            _webhook_goal_queue[goal_id].update(
                {
                    "status": "done",
                    "result": result,
                    "completed_at": time.time(),
                }
            )
            pipeline_runs_total.labels(status="success").inc()
        except Exception as exc:
            _webhook_goal_queue[goal_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": time.time(),
                }
            )
            log_json("WARN", "aura_webhook_goal_failed", details={"goal_id": goal_id, "error": str(exc)})
            pipeline_runs_total.labels(status="failure").inc()
        finally:
            active_pipeline_runs.dec()

    asyncio.create_task(_run_cycle())
    return {"status": "queued", "goal_id": goal_id}


@app.get("/webhook/status/{goal_id}")
async def webhook_goal_status(goal_id: str, _: None = Depends(require_auth)) -> Dict:
    """Poll the status of a webhook-submitted goal (used by n8n P1 fast lane)."""
    entry = _webhook_goal_queue.get(goal_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found")
    result: Dict[str, Any] = {
        "goal_id": goal_id,
        "status": entry["status"],
        "goal": entry["goal"],
        "queued_at": entry["queued_at"],
    }
    if entry["status"] == "done":
        result["result"] = entry.get("result", {})
        result["completed_at"] = entry.get("completed_at")
    elif entry["status"] == "failed":
        result["error"] = entry.get("error", "unknown error")
        result["completed_at"] = entry.get("completed_at")
    elif entry["status"] == "running":
        result["started_at"] = entry.get("started_at")
    return result


@app.post("/webhook/plan-review")
async def webhook_plan_review(req: WebhookPlanReviewRequest, _: None = Depends(require_auth)) -> Dict:
    """Format a task bundle as a plan text for Dev Suite Quality Gate (P2) review.

    Called by P3 Pipeline Coordinator before triggering AURA act phase.
    Returns human-readable plan text that Dev Suite agents can critique.
    """
    plan_steps = req.task_bundle.get("plan", [])
    if isinstance(plan_steps, str):
        plan_text = plan_steps
    elif isinstance(plan_steps, list):
        plan_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan_steps))
    else:
        plan_text = str(plan_steps)

    review_payload = {
        "goal": req.goal,
        "pipeline_run_id": req.pipeline_run_id,
        "plan_text": plan_text,
        "task_bundle_keys": list(req.task_bundle.keys()),
        "file_targets": req.task_bundle.get("file_targets", []),
        "critique": req.task_bundle.get("critique", ""),
    }
    log_json(
        "INFO",
        "aura_webhook_plan_review_requested",
        details={
            "goal": req.goal[:200],
            "pipeline_run_id": req.pipeline_run_id,
            "plan_steps": len(plan_steps) if isinstance(plan_steps, list) else 1,
        },
    )
    return {"status": "ok", "review_payload": review_payload}


class RunRequest(BaseModel):
    """Request body for POST /run."""

    goal: str
    max_cycles: int = 1
    dry_run: bool = False


@app.post("/run")
async def run_pipeline(req: RunRequest, _: None = Depends(require_auth)) -> Dict:
    """Trigger a goal-oriented pipeline run via LoopOrchestrator.

    Requires the ``AGENT_API_ENABLE_RUN=1`` environment variable to be set.
    Returns a ``run_id`` that can be used to track or cancel the run.
    """
    if os.getenv("AGENT_API_ENABLE_RUN") != "1":
        raise HTTPException(status_code=403, detail="Run endpoint disabled; set AGENT_API_ENABLE_RUN=1")

    run_id = secrets.token_hex(12)

    async def _background_run() -> None:
        register_run(run_id)
        try:
            active_orchestrator = await _resolve_runtime_component("orchestrator")
            await asyncio.to_thread(
                active_orchestrator.run_loop,
                req.goal,
                req.max_cycles,
                req.dry_run,
            )
        finally:
            deregister_run(run_id)

    asyncio.create_task(_background_run())
    return {"run_id": run_id, "status": "accepted"}


# ---------------------------------------------------------------------------
# Router registration — extracted sub-modules wired in at startup.
# Routes in runs_router and health_router are mounted under /api to avoid
# conflicts with the existing endpoints defined above.
# ---------------------------------------------------------------------------
try:
    from aura_cli.api.routers.ws import router as _ws_router
    from aura_cli.api.routers.health import router as _health_router
    from aura_cli.api.routers.runs import router as _runs_router
    from aura_cli.api.routers.auth import router as _auth_router

    app.include_router(_ws_router)
    app.include_router(_health_router, prefix="/api")
    app.include_router(_runs_router, prefix="/api")
    app.include_router(_auth_router)
except Exception as _router_import_err:  # pragma: no cover
    log_json("WARN", "aura_server_router_registration_failed", details={"error": str(_router_import_err)})
