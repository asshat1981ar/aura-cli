"""
AURA Control MCP Server — exposes AURA's control plane as MCP-compatible HTTP tools.

Tools exposed:
  Goal management : goal_add, goal_list, goal_remove, goal_clear, goal_archive_list
  Memory / Brain  : memory_search, memory_add, memory_weaknesses
  File access     : file_read, file_list  (jailed to project root)
  Meta            : project_status

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check

Start:
  uvicorn tools.aura_control_mcp:app --port 8003
  # or:
  python tools/aura_control_mcp.py

Auth (optional):
  Set MCP_CONTROL_TOKEN env var — all requests must include Authorization: Bearer <token>
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from core.logging_utils import log_json
from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive
from memory.brain import Brain

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AURA Control MCP Server",
    description="MCP-compatible control plane for the AURA autonomous development loop.",
    version="1.0.0",
)

_TOKEN = os.getenv("MCP_CONTROL_TOKEN", "")
_SERVER_START = time.time()

# In-process call counters {tool_name: count}
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}

# Lazy-loaded singletons (initialised on first request to avoid startup cost)
_goal_queue: Optional[GoalQueue] = None
_goal_archive: Optional[GoalArchive] = None
_brain: Optional[Brain] = None


def _get_goal_queue() -> GoalQueue:
    global _goal_queue
    if _goal_queue is None:
        _goal_queue = GoalQueue(queue_path=str(_ROOT / "memory" / "goal_queue_v2.json"))
    return _goal_queue


def _get_goal_archive() -> GoalArchive:
    global _goal_archive
    if _goal_archive is None:
        _goal_archive = GoalArchive(archive_path=str(_ROOT / "memory" / "goal_archive_v2.json"))
    return _goal_archive


def _get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain(db_path=str(_ROOT / "memory" / "brain_v2.db"))
    return _brain


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _TOKEN:
        return
    if authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any] = {}


class ToolResult(BaseModel):
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Tool descriptors (MCP schema format)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: Dict[str, Dict] = {
    # --- Goal management ---
    "goal_add": {
        "description": "Add a new goal to AURA's work queue.",
        "input": {
            "text": {"type": "string", "description": "Goal description to enqueue", "required": True},
        },
    },
    "goal_list": {
        "description": "List all goals currently in the queue.",
        "input": {},
    },
    "goal_remove": {
        "description": "Remove a goal from the queue by its zero-based index.",
        "input": {
            "index": {"type": "integer", "description": "Zero-based index of the goal to remove", "required": True},
        },
    },
    "goal_clear": {
        "description": "Remove ALL goals from the queue (irreversible).",
        "input": {},
    },
    "goal_archive_list": {
        "description": "List completed/archived goals with their outcome scores.",
        "input": {
            "limit": {"type": "integer", "description": "Max entries to return (default 50)", "default": 50},
        },
    },
    # --- Memory / Brain ---
    "memory_search": {
        "description": "Keyword search AURA's brain memory store.",
        "input": {
            "query": {"type": "string", "description": "Search query string", "required": True},
            "limit": {"type": "integer", "description": "Max results to return (default 20)", "default": 20},
        },
    },
    "memory_add": {
        "description": "Store a new memory in AURA's brain.",
        "input": {
            "text": {"type": "string", "description": "Memory content to store", "required": True},
        },
    },
    "memory_weaknesses": {
        "description": "List known agent weaknesses recorded in AURA's brain.",
        "input": {},
    },
    # --- File access (project-root jailed) ---
    "file_read": {
        "description": "Read a file within the AURA project root. Path must be relative.",
        "input": {
            "path": {"type": "string", "description": "Relative file path within project root", "required": True},
        },
    },
    "file_list": {
        "description": "List files and directories within the AURA project root. Path must be relative.",
        "input": {
            "path": {"type": "string", "description": "Relative directory path (default: '.')", "default": "."},
        },
    },
    # --- Meta ---
    "project_status": {
        "description": "Return a summary of AURA's current state: queue size, memory count, archive size.",
        "input": {},
    },
}


def _build_descriptor(name: str) -> Dict:
    schema = _TOOL_SCHEMAS[name]
    return {
        "name": name,
        "description": schema["description"],
        "inputSchema": {
            "type": "object",
            "properties": schema.get("input", {}),
            "required": [k for k, v in schema.get("input", {}).items() if v.get("required")],
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _goal_add(args: Dict) -> Any:
    text = args.get("text", "").strip()
    if not text:
        raise ValueError("'text' is required and must not be empty.")
    q = _get_goal_queue()
    q.add(text)
    return {"added": text, "queue_size": len(list(q.queue))}


def _goal_list(args: Dict) -> Any:
    q = _get_goal_queue()
    goals = list(q.queue)
    return {"goals": goals, "count": len(goals)}


def _goal_remove(args: Dict) -> Any:
    index = args.get("index")
    if index is None:
        raise ValueError("'index' is required.")
    index = int(index)
    q = _get_goal_queue()
    goals = list(q.queue)
    if index < 0 or index >= len(goals):
        raise ValueError(f"Index {index} out of range (queue has {len(goals)} items).")
    removed = goals.pop(index)
    q.queue = deque(goals)
    q._save_queue()
    return {"removed": removed, "queue_size": len(goals)}


def _goal_clear(args: Dict) -> Any:
    q = _get_goal_queue()
    count = len(list(q.queue))
    q.queue = deque()
    q._save_queue()
    return {"cleared": count}


def _goal_archive_list(args: Dict) -> Any:
    limit = int(args.get("limit", 50))
    archive = _get_goal_archive()
    entries = archive._load_archive()
    if isinstance(entries, list):
        entries = entries[-limit:]
    elif isinstance(entries, dict):
        entries = list(entries.items())[-limit:]
    return {"entries": entries, "count": len(entries)}


def _memory_search(args: Dict) -> Any:
    query = args.get("query", "").strip()
    if not query:
        raise ValueError("'query' is required.")
    limit = int(args.get("limit", 20))
    brain = _get_brain()
    all_memories = brain.recall_all()
    q_lower = query.lower()
    matches = [m for m in all_memories if q_lower in str(m).lower()]
    return {"matches": matches[:limit], "total_matched": len(matches)}


def _memory_add(args: Dict) -> Any:
    text = args.get("text", "").strip()
    if not text:
        raise ValueError("'text' is required.")
    brain = _get_brain()
    brain.remember(text)
    return {"stored": text}


def _memory_weaknesses(args: Dict) -> Any:
    brain = _get_brain()
    weaknesses = brain.recall_weaknesses()
    return {"weaknesses": weaknesses, "count": len(weaknesses)}


def _resolve_safe_path(rel_path: str) -> Path:
    """Resolve rel_path under project root, raising ValueError if it escapes."""
    try:
        resolved = (_ROOT / rel_path).resolve()
        resolved.relative_to(_ROOT.resolve())  # raises ValueError if outside
        return resolved
    except ValueError:
        raise ValueError(f"Path '{rel_path}' escapes the project root. Only relative paths are allowed.")


def _file_read(args: Dict) -> Any:
    path_str = args.get("path", "").strip()
    if not path_str:
        raise ValueError("'path' is required.")
    path = _resolve_safe_path(path_str)
    if not path.exists():
        raise ValueError(f"File not found: {path_str}")
    if not path.is_file():
        raise ValueError(f"'{path_str}' is not a file.")
    content = path.read_text(encoding="utf-8", errors="replace")
    return {"path": path_str, "content": content, "size_bytes": path.stat().st_size}


def _file_list(args: Dict) -> Any:
    path_str = args.get("path", ".").strip() or "."
    path = _resolve_safe_path(path_str)
    if not path.exists():
        raise ValueError(f"Path not found: {path_str}")
    if not path.is_dir():
        raise ValueError(f"'{path_str}' is not a directory.")
    entries = []
    for item in sorted(path.iterdir()):
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size_bytes": item.stat().st_size if item.is_file() else None,
        })
    return {"path": path_str, "entries": entries, "count": len(entries)}


def _project_status(args: Dict) -> Any:
    try:
        q = _get_goal_queue()
        queue_size = len(list(q.queue))
    except Exception as e:
        queue_size = f"error: {e}"

    try:
        brain = _get_brain()
        memory_count = len(brain.recall_all())
        weakness_count = len(brain.recall_weaknesses())
    except Exception as e:
        memory_count = weakness_count = f"error: {e}"

    try:
        archive = _get_goal_archive()
        archive_entries = archive._load_archive()
        archive_size = len(archive_entries) if isinstance(archive_entries, list) else len(list(archive_entries))
    except Exception as e:
        archive_size = f"error: {e}"

    return {
        "project_root": str(_ROOT),
        "queue_size": queue_size,
        "memory_count": memory_count,
        "weakness_count": weakness_count,
        "archive_size": archive_size,
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "goal_add": _goal_add,
    "goal_list": _goal_list,
    "goal_remove": _goal_remove,
    "goal_clear": _goal_clear,
    "goal_archive_list": _goal_archive_list,
    "memory_search": _memory_search,
    "memory_add": _memory_add,
    "memory_weaknesses": _memory_weaknesses,
    "file_read": _file_read,
    "file_list": _file_list,
    "project_status": _project_status,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "aura_control_mcp",
        "version": "1.0.0",
    }


@app.get("/tools")
async def list_tools(_: None = Depends(_check_auth)) -> List[Dict]:
    return [_build_descriptor(name) for name in _TOOL_SCHEMAS]


@app.get("/tool/{name}")
async def get_tool(name: str, _: None = Depends(_check_auth)) -> Dict:
    if name not in _TOOL_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return _build_descriptor(name)


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
        log_json("INFO", "control_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "control_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "control_mcp_tool_error", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=f"Internal error: {exc}", elapsed_ms=elapsed)


@app.get("/metrics")
async def get_metrics(_: None = Depends(_check_auth)) -> Dict:
    """Return uptime, per-tool call/error counts, queue size, and memory count."""
    uptime_s = round(time.time() - _SERVER_START, 1)
    total_calls = sum(_call_counts.values())
    total_errors = sum(_call_errors.values())

    # Live queue size (best-effort)
    queue_size: int = 0
    try:
        q = _get_goal_queue()
        queue_size = len(q.queue) if hasattr(q, "queue") else 0
    except Exception:
        pass

    # Live memory count (best-effort)
    memory_count: int = 0
    try:
        memories = _get_brain().recall_all()
        memory_count = len(memories)
    except Exception:
        pass

    per_tool = {
        name: {
            "calls": _call_counts.get(name, 0),
            "errors": _call_errors.get(name, 0),
        }
        for name in _TOOL_SCHEMAS
    }

    return {
        "uptime_seconds": uptime_s,
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_calls, 1), 4),
        "queue_size": queue_size,
        "memory_count": memory_count,
        "tools": per_tool,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_CONTROL_PORT", "8003"))
    uvicorn.run("tools.aura_control_mcp:app", host="0.0.0.0", port=port, reload=False)
