"""
Agentic Loop MCP Server — port 8006

Exposes AURA's workflow engine and agentic loop as MCP-compatible HTTP tools.

Tools:
  Workflow management:
    workflow_define   — register a custom workflow definition
    workflow_list     — list all known workflow definitions
    workflow_run      — start a workflow execution (non-blocking, returns exec_id)
    workflow_status   — get live status + step history for an execution
    workflow_output   — get the output dict for a specific completed step
    workflow_cancel   — cancel a running execution
    workflow_pause    — pause a running execution
    workflow_resume   — resume a paused execution

  Agentic loop management:
    loop_create       — create a new agentic loop (goal + max_cycles)
    loop_tick         — execute one cycle of the loop (call repeatedly)
    loop_status       — get loop state, cycle history, stop reason
    loop_stop         — stop a loop immediately
    loop_pause        — pause a running loop
    loop_resume       — resume a paused loop
    loop_health       — deadlock/stall detection for a loop
    loop_list         — list all loops with optional status filter

Endpoints:
  GET  /tools              → MCP tool descriptors
  POST /call               → invoke any tool by name
  GET  /tool/{name}        → single tool descriptor
  GET  /health             → health check + engine stats
  GET  /metrics            → call counts + error rates
  GET  /workflows          → shortcut: list workflow definitions
  GET  /loops              → shortcut: list loops

Auth (optional):
  Set AGENTIC_LOOP_TOKEN env var — all requests must include
  Authorization: Bearer <token>

Start:
  uvicorn tools.agentic_loop_mcp:app --port 8006
  # or:
  python tools/agentic_loop_mcp.py
"""
from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from core.logging_utils import log_json
from core.workflow_engine import (
    WorkflowDefinition,
    WorkflowStep,
    RetryPolicy,
    get_engine,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic Loop MCP Server",
    description="MCP-compatible workflow engine and agentic loop server for AURA.",
    version="1.0.0",
)

_TOKEN = os.getenv("AGENTIC_LOOP_TOKEN", "")
_SERVER_START = time.time()
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    if authorization.removeprefix("Bearer ").strip() != _TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Tool schemas (MCP descriptors)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: Dict[str, Dict] = {
    "workflow_define": {
        "description": "Register a custom workflow definition as an ordered list of skill steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique workflow name."},
                "description": {"type": "string"},
                "steps": {
                    "type": "array",
                    "description": "Ordered list of step definitions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "skill_name": {"type": "string", "description": "AURA skill name."},
                            "static_inputs": {"type": "object"},
                            "inputs_from": {"type": "object",
                                            "description": "{'dest_key': 'step_name.output_key'}"},
                            "skip_if_false": {"type": "string"},
                            "max_attempts": {"type": "integer", "default": 3},
                            "timeout_s": {"type": "number", "default": 120},
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": ["name", "steps"],
        },
    },
    "workflow_list": {
        "description": "List all registered workflow definitions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "workflow_run": {
        "description": "Start a workflow execution. Returns execution_id for polling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_name": {"type": "string"},
                "inputs": {"type": "object",
                           "description": "Initial inputs (e.g. {'project_root': '.'})"},
                "background": {"type": "boolean", "default": False,
                               "description": "Run in background thread; returns immediately."},
            },
            "required": ["workflow_name"],
        },
    },
    "workflow_status": {
        "description": "Get live status and step history for a workflow execution.",
        "inputSchema": {
            "type": "object",
            "properties": {"exec_id": {"type": "string"}},
            "required": ["exec_id"],
        },
    },
    "workflow_output": {
        "description": "Get the full output dict of a specific completed step.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "exec_id": {"type": "string"},
                "step_name": {"type": "string"},
            },
            "required": ["exec_id", "step_name"],
        },
    },
    "workflow_cancel": {
        "description": "Cancel a running or paused workflow execution.",
        "inputSchema": {
            "type": "object",
            "properties": {"exec_id": {"type": "string"}},
            "required": ["exec_id"],
        },
    },
    "workflow_pause": {
        "description": "Pause a running execution (it will stop after the current step).",
        "inputSchema": {
            "type": "object",
            "properties": {"exec_id": {"type": "string"}},
            "required": ["exec_id"],
        },
    },
    "workflow_resume": {
        "description": "Resume a paused workflow execution from where it stopped.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "exec_id": {"type": "string"},
                "workflow_name": {"type": "string"},
            },
            "required": ["exec_id"],
        },
    },
    "loop_create": {
        "description": "Create a new agentic goal-processing loop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The goal for the agentic loop."},
                "max_cycles": {"type": "integer", "default": 5},
            },
            "required": ["goal"],
        },
    },
    "loop_tick": {
        "description": "Execute one cycle of the agentic loop. Call repeatedly until loop_status shows terminal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "loop_id": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["loop_id"],
        },
    },
    "loop_status": {
        "description": "Get full loop state: cycle history, stop reason, elapsed time.",
        "inputSchema": {
            "type": "object",
            "properties": {"loop_id": {"type": "string"}},
            "required": ["loop_id"],
        },
    },
    "loop_stop": {
        "description": "Stop a running loop immediately.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "loop_id": {"type": "string"},
                "reason": {"type": "string", "default": "user_requested"},
            },
            "required": ["loop_id"],
        },
    },
    "loop_pause": {
        "description": "Pause a running loop (it will stop after the current cycle).",
        "inputSchema": {
            "type": "object",
            "properties": {"loop_id": {"type": "string"}},
            "required": ["loop_id"],
        },
    },
    "loop_resume": {
        "description": "Resume a paused loop.",
        "inputSchema": {
            "type": "object",
            "properties": {"loop_id": {"type": "string"}},
            "required": ["loop_id"],
        },
    },
    "loop_health": {
        "description": "Detect stalls, deadlocks, or repeated errors in a running loop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "loop_id": {"type": "string"},
                "stall_threshold_s": {"type": "number", "default": 300},
            },
            "required": ["loop_id"],
        },
    },
    "loop_list": {
        "description": "List all agentic loops with optional status filter.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string",
                           "enum": ["running", "paused", "completed", "failed", "stopped"]},
            },
        },
    },
}


def _build_descriptor(name: str) -> Dict:
    schema = _TOOL_SCHEMAS[name]
    return {"name": name, "description": schema["description"],
            "inputSchema": schema["inputSchema"]}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _h_workflow_define(args: Dict) -> Dict:
    name = args.get("name", "").strip()
    if not name:
        raise ValueError("'name' is required.")
    raw_steps = args.get("steps")
    if not raw_steps or not isinstance(raw_steps, list):
        raise ValueError("'steps' must be a non-empty list.")

    steps: List[WorkflowStep] = []
    for s in raw_steps:
        if not isinstance(s, dict) or not s.get("name"):
            raise ValueError(f"Each step must have a 'name'. Got: {s!r}")
        retry = RetryPolicy(max_attempts=int(s.get("max_attempts", 3)))
        steps.append(WorkflowStep(
            name=s["name"],
            skill_name=s.get("skill_name"),
            static_inputs=s.get("static_inputs", {}),
            inputs_from=s.get("inputs_from", {}),
            skip_if_false=s.get("skip_if_false"),
            retry=retry,
            timeout_s=float(s.get("timeout_s", 120)),
        ))

    wf = WorkflowDefinition(
        name=name,
        steps=steps,
        description=args.get("description", ""),
    )
    get_engine().define(wf)
    return {"defined": name, "step_count": len(steps), "steps": [s.name for s in steps]}


def _h_workflow_list(args: Dict) -> Dict:
    return {"workflows": get_engine().list_definitions()}


def _h_workflow_run(args: Dict) -> Dict:
    wf_name = args.get("workflow_name", "").strip()
    if not wf_name:
        raise ValueError("'workflow_name' is required.")
    inputs = args.get("inputs") or {}
    background = bool(args.get("background", False))

    engine = get_engine()
    if background:
        # Start in background thread; return exec_id immediately
        exec_id_holder: Dict = {}
        exc_holder: Dict = {}

        def _run():
            try:
                exec_id_holder["id"] = engine.run_workflow(wf_name, inputs)
            except Exception as e:
                exc_holder["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=0.1)  # give the thread just enough time to register the execution

        if "error" in exc_holder:
            raise ValueError(exc_holder["error"])
        # Return a provisional exec_id if registration happened fast enough
        if "id" in exec_id_holder:
            return {"exec_id": exec_id_holder["id"], "mode": "background", "workflow": wf_name}

        # Fallback: create a stub execution so the caller has an ID to poll
        import uuid as _uuid
        stub_id = str(_uuid.uuid4())
        return {"exec_id": stub_id, "mode": "background_started", "workflow": wf_name,
                "note": "Poll workflow_status after a moment."}
    else:
        exec_id = engine.run_workflow(wf_name, inputs)
        return {"exec_id": exec_id, "mode": "sync", "workflow": wf_name,
                "status": engine.execution_status(exec_id)["status"]}


def _h_workflow_status(args: Dict) -> Dict:
    exec_id = args.get("exec_id", "").strip()
    if not exec_id:
        raise ValueError("'exec_id' is required.")
    return get_engine().execution_status(exec_id)


def _h_workflow_output(args: Dict) -> Dict:
    exec_id = args.get("exec_id", "").strip()
    step_name = args.get("step_name", "").strip()
    if not exec_id or not step_name:
        raise ValueError("'exec_id' and 'step_name' are required.")
    return get_engine().get_step_output(exec_id, step_name)


def _h_workflow_cancel(args: Dict) -> Dict:
    exec_id = args.get("exec_id", "").strip()
    if not exec_id:
        raise ValueError("'exec_id' is required.")
    get_engine().cancel_execution(exec_id)
    return {"cancelled": exec_id}


def _h_workflow_pause(args: Dict) -> Dict:
    exec_id = args.get("exec_id", "").strip()
    if not exec_id:
        raise ValueError("'exec_id' is required.")
    get_engine().pause_execution(exec_id)
    return {"paused": exec_id}


def _h_workflow_resume(args: Dict) -> Dict:
    exec_id = args.get("exec_id", "").strip()
    if not exec_id:
        raise ValueError("'exec_id' is required.")
    wf_name = args.get("workflow_name", "").strip()
    if not wf_name:
        # Try to infer from existing execution
        status = get_engine().execution_status(exec_id)
        wf_name = status["workflow"]
    exec_id_out = get_engine().run_workflow(wf_name, resume_exec_id=exec_id)
    return {"resumed": exec_id_out, "status": get_engine().execution_status(exec_id_out)["status"]}


def _h_loop_create(args: Dict) -> Dict:
    goal = args.get("goal", "").strip()
    if not goal:
        raise ValueError("'goal' is required.")
    max_cycles = int(args.get("max_cycles", 5))
    if max_cycles < 1 or max_cycles > 50:
        raise ValueError("'max_cycles' must be between 1 and 50.")
    loop_id = get_engine().create_loop(goal, max_cycles)
    return {"loop_id": loop_id, "goal": goal, "max_cycles": max_cycles}


def _h_loop_tick(args: Dict) -> Dict:
    loop_id = args.get("loop_id", "").strip()
    if not loop_id:
        raise ValueError("'loop_id' is required.")
    dry_run = bool(args.get("dry_run", False))
    return get_engine().loop_tick(loop_id, dry_run=dry_run)


def _h_loop_status(args: Dict) -> Dict:
    loop_id = args.get("loop_id", "").strip()
    if not loop_id:
        raise ValueError("'loop_id' is required.")
    return get_engine().loop_status(loop_id)


def _h_loop_stop(args: Dict) -> Dict:
    loop_id = args.get("loop_id", "").strip()
    if not loop_id:
        raise ValueError("'loop_id' is required.")
    reason = args.get("reason", "user_requested")
    get_engine().stop_loop(loop_id, reason=reason)
    return {"stopped": loop_id, "reason": reason}


def _h_loop_pause(args: Dict) -> Dict:
    loop_id = args.get("loop_id", "").strip()
    if not loop_id:
        raise ValueError("'loop_id' is required.")
    get_engine().pause_loop(loop_id)
    return {"paused": loop_id}


def _h_loop_resume(args: Dict) -> Dict:
    loop_id = args.get("loop_id", "").strip()
    if not loop_id:
        raise ValueError("'loop_id' is required.")
    get_engine().resume_loop(loop_id)
    return {"resumed": loop_id}


def _h_loop_health(args: Dict) -> Dict:
    loop_id = args.get("loop_id", "").strip()
    if not loop_id:
        raise ValueError("'loop_id' is required.")
    threshold = float(args.get("stall_threshold_s", 300))
    return get_engine().check_loop_health(loop_id, stall_threshold_s=threshold)


def _h_loop_list(args: Dict) -> Dict:
    status_filter = args.get("status")
    return {"loops": get_engine().list_loops(status_filter=status_filter)}


_TOOL_HANDLERS: Dict[str, Any] = {
    "workflow_define":  _h_workflow_define,
    "workflow_list":    _h_workflow_list,
    "workflow_run":     _h_workflow_run,
    "workflow_status":  _h_workflow_status,
    "workflow_output":  _h_workflow_output,
    "workflow_cancel":  _h_workflow_cancel,
    "workflow_pause":   _h_workflow_pause,
    "workflow_resume":  _h_workflow_resume,
    "loop_create":      _h_loop_create,
    "loop_tick":        _h_loop_tick,
    "loop_status":      _h_loop_status,
    "loop_stop":        _h_loop_stop,
    "loop_pause":       _h_loop_pause,
    "loop_resume":      _h_loop_resume,
    "loop_health":      _h_loop_health,
    "loop_list":        _h_loop_list,
}


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)) -> Dict:
    engine = get_engine()
    execs = engine.list_executions()
    loops = engine.list_loops()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _SERVER_START, 1),
        "workflows_defined": len(engine.list_definitions()),
        "executions_active": sum(1 for e in execs if e["status"] == "running"),
        "executions_total": len(execs),
        "loops_running": sum(1 for l in loops if l["status"] == "running"),
        "loops_total": len(loops),
    }


@app.get("/tools")
async def list_tools(_: None = Depends(_check_auth)) -> Dict:
    return {"tools": [_build_descriptor(name) for name in _TOOL_SCHEMAS]}


@app.get("/tool/{name}")
async def get_tool(name: str, _: None = Depends(_check_auth)) -> Dict:
    if name not in _TOOL_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return _build_descriptor(name)


@app.get("/workflows")
async def list_workflows_route(_: None = Depends(_check_auth)) -> Dict:
    return {"workflows": get_engine().list_definitions()}


@app.get("/loops")
async def list_loops_route(status: Optional[str] = None, _: None = Depends(_check_auth)) -> Dict:
    return {"loops": get_engine().list_loops(status_filter=status)}


@app.post("/call")
async def call_tool(request: ToolCallRequest, _: None = Depends(_check_auth)) -> ToolResult:
    name = request.tool_name
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")

    _call_counts[name] = _call_counts.get(name, 0) + 1
    t0 = time.time()
    try:
        result = handler(request.args)
        elapsed = round((time.time() - t0) * 1000, 2)
        log_json("INFO", "agentic_loop_mcp_tool_called",
                 details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except (ValueError, KeyError) as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "agentic_loop_mcp_bad_args",
                 details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "agentic_loop_mcp_error",
                 details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=f"Internal error: {exc}", elapsed_ms=elapsed)


@app.get("/metrics")
async def get_metrics(_: None = Depends(_check_auth)) -> Dict:
    total_calls = sum(_call_counts.values())
    total_errors = sum(_call_errors.values())
    engine = get_engine()
    return {
        "uptime_seconds": round(time.time() - _SERVER_START, 1),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_calls, 1), 4),
        "executions_total": len(engine.list_executions()),
        "loops_total": len(engine.list_loops()),
        "tools": {
            name: {
                "calls": _call_counts.get(name, 0),
                "errors": _call_errors.get(name, 0),
            }
            for name in _TOOL_SCHEMAS
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENTIC_LOOP_PORT", "8006"))
    uvicorn.run("tools.agentic_loop_mcp:app", host="0.0.0.0", port=port, reload=False)
