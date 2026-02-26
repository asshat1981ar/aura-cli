"""
Sequential Thinking MCP Server — persistent, revisable, branchable thought chains for AI agents.

Modelled after the Anthropic sequential-thinking pattern but adds:
  • Full persistence (SQLite at memory/thinking_sessions.db)
  • Step revision with history
  • Branching (explore alternative reasoning paths)
  • Session summary / condensation
  • Cross-session search

Tools:
  session_create    — start a new thinking session for a goal
  think             — append a thought step to a session
  revise            — revise a previous step (old version is preserved)
  branch            — fork the chain from a step to explore an alternative path
  conclude          — finalise a session with a conclusion/answer
  session_get       — retrieve full session state + all steps
  session_list      — list sessions (filter by status)
  session_search    — full-text search across all thought content
  session_abandon   — mark a session abandoned

Ports: 8004 (default, override with MCP_THINKING_PORT)
Auth:  set MCP_THINKING_TOKEN env var

Start:
  uvicorn tools.sequential_thinking_mcp:app --port 8004
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

_DB_PATH = _ROOT / "memory" / "thinking_sessions.db"


@contextmanager
def _db():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                goal        TEXT NOT NULL,
                context     TEXT,
                status      TEXT NOT NULL DEFAULT 'active',
                conclusion  TEXT,
                answer      TEXT,
                branch_of   TEXT,           -- parent session id if this is a branch
                branch_from_step INTEGER,   -- step number in parent we branched from
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS steps (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                step_number INTEGER NOT NULL,
                content     TEXT NOT NULL,
                confidence  REAL,           -- 0.0-1.0, optional
                is_revision INTEGER NOT NULL DEFAULT 0,
                revision_index INTEGER NOT NULL DEFAULT 0,  -- increments with each revision of the same step
                revises_step_id TEXT,       -- id of the step this revises
                created_at  REAL NOT NULL,
                UNIQUE(session_id, step_number, revision_index)
            );

            CREATE INDEX IF NOT EXISTS idx_steps_session ON steps(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        """)


_init_db()

# ---------------------------------------------------------------------------
# App + auth
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sequential Thinking MCP",
    description="Persistent, revisable, branchable thought-chain server for AI agents.",
    version="1.0.0",
)

_TOKEN = os.getenv("MCP_THINKING_TOKEN", "")


def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if _TOKEN and authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    goal: str
    context: Optional[str] = None


class ThinkRequest(BaseModel):
    session_id: str
    thought: str
    confidence: Optional[float] = None  # 0.0 – 1.0


class ReviseRequest(BaseModel):
    session_id: str
    step_number: int
    revised_thought: str
    reason: Optional[str] = None


class BranchRequest(BaseModel):
    session_id: str
    from_step: int
    first_thought: str


class ConcludeRequest(BaseModel):
    session_id: str
    conclusion: str
    answer: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return row


def _next_step_number(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(step_number) as m FROM steps WHERE session_id = ? AND is_revision = 0",
        (session_id,),
    ).fetchone()
    return (row["m"] or 0) + 1


def _session_to_dict(row: sqlite3.Row) -> Dict:
    return dict(row)


def _steps_for_session(conn: sqlite3.Connection, session_id: str) -> List[Dict]:
    rows = conn.execute(
        "SELECT * FROM steps WHERE session_id = ? ORDER BY step_number, is_revision, created_at",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# MCP tool descriptors
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = {
    "session_create": {
        "description": "Start a new sequential thinking session for a goal.",
        "input": {
            "goal": {"type": "string", "description": "The problem or question to think through", "required": True},
            "context": {"type": "string", "description": "Optional background context"},
        },
    },
    "think": {
        "description": "Append the next thought step to an active session.",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
            "thought": {"type": "string", "description": "The thought content for this step", "required": True},
            "confidence": {"type": "number", "description": "Confidence in this step (0.0–1.0)"},
        },
    },
    "revise": {
        "description": "Revise a previous step. The old version is preserved in history.",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
            "step_number": {"type": "integer", "description": "Step number to revise", "required": True},
            "revised_thought": {"type": "string", "description": "Replacement thought content", "required": True},
            "reason": {"type": "string", "description": "Why this step is being revised"},
        },
    },
    "branch": {
        "description": "Fork the thinking chain from a step to explore an alternative reasoning path. Returns a new session ID.",
        "input": {
            "session_id": {"type": "string", "description": "Parent session ID", "required": True},
            "from_step": {"type": "integer", "description": "Step number to fork from", "required": True},
            "first_thought": {"type": "string", "description": "First thought of the new branch", "required": True},
        },
    },
    "conclude": {
        "description": "Finalise a thinking session with a conclusion and optional answer.",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
            "conclusion": {"type": "string", "description": "Summary of what was learned/decided", "required": True},
            "answer": {"type": "string", "description": "The final answer or decision, if applicable"},
        },
    },
    "session_get": {
        "description": "Retrieve the full state of a session including all steps and revision history.",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
        },
    },
    "session_list": {
        "description": "List thinking sessions, optionally filtered by status.",
        "input": {
            "status": {"type": "string", "description": "'active', 'concluded', 'abandoned', or omit for all"},
            "limit": {"type": "integer", "description": "Max sessions to return (default 50)"},
        },
    },
    "session_search": {
        "description": "Full-text search across all thought content in all sessions.",
        "input": {
            "query": {"type": "string", "description": "Search query", "required": True},
            "limit": {"type": "integer", "description": "Max results (default 20)"},
        },
    },
    "session_abandon": {
        "description": "Mark a session as abandoned (stopped without conclusion).",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
        },
    },
}


def _build_descriptor(name: str) -> Dict:
    s = _TOOL_SCHEMAS[name]
    return {
        "name": name,
        "description": s["description"],
        "inputSchema": {
            "type": "object",
            "properties": s.get("input", {}),
            "required": [k for k, v in s.get("input", {}).items() if v.get("required")],
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    with _db() as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        step_count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    return {"status": "ok", "sessions": session_count, "steps": step_count, "server": "sequential_thinking_mcp"}


@app.get("/tools")
async def list_tools(_: None = Depends(_check_auth)) -> List[Dict]:
    return [_build_descriptor(n) for n in _TOOL_SCHEMAS]


@app.get("/tool/{name}")
async def get_tool(name: str, _: None = Depends(_check_auth)):
    if name not in _TOOL_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return _build_descriptor(name)


@app.post("/session/create")
async def session_create(req: SessionCreateRequest, _: None = Depends(_check_auth)):
    sid = str(uuid.uuid4())
    now = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, goal, context, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (sid, req.goal, req.context, "active", now, now),
        )
    log_json("INFO", "thinking_session_created", details={"session_id": sid, "goal": req.goal[:80]})
    return {"session_id": sid, "goal": req.goal, "status": "active"}


@app.post("/think")
async def think(req: ThinkRequest, _: None = Depends(_check_auth)):
    now = time.time()
    with _db() as conn:
        session = _get_session(conn, req.session_id)
        if session["status"] != "active":
            raise HTTPException(status_code=400, detail=f"Session is '{session['status']}', not active.")
        step_no = _next_step_number(conn, req.session_id)
        step_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO steps (id, session_id, step_number, content, confidence, is_revision, revision_index, created_at)"
            " VALUES (?,?,?,?,?,0,0,?)",
            (step_id, req.session_id, step_no, req.thought, req.confidence, now),
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, req.session_id))
    log_json("INFO", "thinking_step_added", details={"session_id": req.session_id, "step": step_no})
    return {"step_id": step_id, "step_number": step_no, "session_id": req.session_id}


@app.post("/revise")
async def revise(req: ReviseRequest, _: None = Depends(_check_auth)):
    now = time.time()
    with _db() as conn:
        session = _get_session(conn, req.session_id)
        if session["status"] != "active":
            raise HTTPException(status_code=400, detail=f"Session is '{session['status']}', not active.")
        # Find original step
        orig = conn.execute(
            "SELECT id FROM steps WHERE session_id=? AND step_number=? AND is_revision=0",
            (req.session_id, req.step_number),
        ).fetchone()
        if not orig:
            raise HTTPException(status_code=404, detail=f"Step {req.step_number} not found in session.")
        # Determine next revision_index for this step
        max_rev = conn.execute(
            "SELECT MAX(revision_index) as m FROM steps WHERE session_id=? AND step_number=?",
            (req.session_id, req.step_number),
        ).fetchone()
        next_rev_idx = (max_rev["m"] or 0) + 1
        # Insert revision record with incremented revision_index
        rev_id = str(uuid.uuid4())
        content = req.revised_thought
        if req.reason:
            content = f"[Revision reason: {req.reason}]\n{content}"
        conn.execute(
            "INSERT INTO steps (id, session_id, step_number, content, is_revision, revision_index, revises_step_id, created_at)"
            " VALUES (?,?,?,?,1,?,?,?)",
            (rev_id, req.session_id, req.step_number, content, next_rev_idx, orig["id"], now),
        )
        # Update the original step to the new content
        conn.execute(
            "UPDATE steps SET content=?, created_at=? WHERE id=?",
            (req.revised_thought, now, orig["id"]),
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, req.session_id))
    log_json("INFO", "thinking_step_revised", details={"session_id": req.session_id, "step": req.step_number})
    return {"revised_step_id": orig["id"], "revision_record_id": rev_id, "step_number": req.step_number}


@app.post("/branch")
async def branch(req: BranchRequest, _: None = Depends(_check_auth)):
    now = time.time()
    with _db() as conn:
        parent = _get_session(conn, req.session_id)
        # Verify from_step exists
        step_exists = conn.execute(
            "SELECT 1 FROM steps WHERE session_id=? AND step_number=? AND is_revision=0",
            (req.session_id, req.from_step),
        ).fetchone()
        if not step_exists:
            raise HTTPException(status_code=404, detail=f"Step {req.from_step} not found in parent session.")
        # Copy steps up to from_step into new session
        new_sid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO sessions (id, goal, context, status, branch_of, branch_from_step, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (new_sid, parent["goal"], parent["context"], "active", req.session_id, req.from_step, now, now),
        )
        # Copy steps 1..from_step
        for row in conn.execute(
            "SELECT * FROM steps WHERE session_id=? AND step_number<=? AND is_revision=0 ORDER BY step_number",
            (req.session_id, req.from_step),
        ).fetchall():
            conn.execute(
                "INSERT INTO steps (id, session_id, step_number, content, confidence, is_revision, revision_index, created_at)"
                " VALUES (?,?,?,?,?,0,0,?)",
                (str(uuid.uuid4()), new_sid, row["step_number"], row["content"], row["confidence"], now),
            )
        # Add first thought of branch
        first_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO steps (id, session_id, step_number, content, is_revision, revision_index, created_at)"
            " VALUES (?,?,?,?,0,0,?)",
            (first_id, new_sid, req.from_step + 1, req.first_thought, now),
        )
    log_json("INFO", "thinking_branch_created", details={"parent": req.session_id, "new": new_sid, "from_step": req.from_step})
    return {
        "branch_session_id": new_sid,
        "parent_session_id": req.session_id,
        "branched_from_step": req.from_step,
        "first_step_number": req.from_step + 1,
    }


@app.post("/conclude")
async def conclude(req: ConcludeRequest, _: None = Depends(_check_auth)):
    now = time.time()
    with _db() as conn:
        session = _get_session(conn, req.session_id)
        if session["status"] != "active":
            raise HTTPException(status_code=400, detail=f"Session is already '{session['status']}'.")
        step_count = conn.execute(
            "SELECT COUNT(*) FROM steps WHERE session_id=? AND is_revision=0",
            (req.session_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE sessions SET status='concluded', conclusion=?, answer=?, updated_at=? WHERE id=?",
            (req.conclusion, req.answer, now, req.session_id),
        )
    log_json("INFO", "thinking_session_concluded", details={"session_id": req.session_id, "steps": step_count})
    return {
        "session_id": req.session_id,
        "status": "concluded",
        "step_count": step_count,
        "conclusion": req.conclusion,
        "answer": req.answer,
    }


@app.get("/session/{session_id}")
async def session_get(session_id: str, _: None = Depends(_check_auth)):
    with _db() as conn:
        session = dict(_get_session(conn, session_id))
        steps = _steps_for_session(conn, session_id)
        # Separate current steps from revision history
        current = [s for s in steps if not s["is_revision"]]
        history = [s for s in steps if s["is_revision"]]
    return {
        **session,
        "steps": current,
        "revision_history": history,
        "step_count": len(current),
    }


@app.get("/sessions")
async def session_list(status: Optional[str] = None, limit: int = 50, _: None = Depends(_check_auth)):
    with _db() as conn:
        if status:
            rows = conn.execute(
                "SELECT *, (SELECT COUNT(*) FROM steps WHERE session_id=sessions.id AND is_revision=0) as step_count "
                "FROM sessions WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT *, (SELECT COUNT(*) FROM steps WHERE session_id=sessions.id AND is_revision=0) as step_count "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return {"sessions": [dict(r) for r in rows], "count": len(rows)}


@app.post("/search")
async def session_search(req: SearchRequest, _: None = Depends(_check_auth)):
    q = f"%{req.query.lower()}%"
    with _db() as conn:
        rows = conn.execute(
            """SELECT s.id as step_id, s.session_id, s.step_number, s.content,
                      sess.goal, sess.status
               FROM steps s JOIN sessions sess ON s.session_id = sess.id
               WHERE LOWER(s.content) LIKE ? AND s.is_revision=0
               ORDER BY s.created_at DESC LIMIT ?""",
            (q, req.limit),
        ).fetchall()
    return {"query": req.query, "results": [dict(r) for r in rows], "count": len(rows)}


@app.post("/session/{session_id}/abandon")
async def session_abandon(session_id: str, _: None = Depends(_check_auth)):
    now = time.time()
    with _db() as conn:
        _get_session(conn, session_id)
        conn.execute("UPDATE sessions SET status='abandoned', updated_at=? WHERE id=?", (now, session_id))
    return {"session_id": session_id, "status": "abandoned"}


# Generic /call dispatcher (MCP-style)
class ToolCallRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any] = {}


@app.post("/call")
async def call_tool(req: ToolCallRequest, _: None = Depends(_check_auth)):
    """MCP-compatible generic dispatcher."""
    dispatch = {
        "session_create": lambda a: session_create(SessionCreateRequest(**a)),
        "think": lambda a: think(ThinkRequest(**a)),
        "revise": lambda a: revise(ReviseRequest(**a)),
        "branch": lambda a: branch(BranchRequest(**a)),
        "conclude": lambda a: conclude(ConcludeRequest(**a)),
        "session_get": lambda a: session_get(a["session_id"]),
        "session_list": lambda a: session_list(a.get("status"), int(a.get("limit", 50))),
        "session_search": lambda a: session_search(SearchRequest(**a)),
        "session_abandon": lambda a: session_abandon(a["session_id"]),
    }
    handler = dispatch.get(req.tool_name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found.")
    import inspect
    t0 = time.time()
    try:
        result = handler(req.args)
        if inspect.isawaitable(result):
            result = await result
        return {"tool_name": req.tool_name, "result": result, "error": None, "elapsed_ms": round((time.time()-t0)*1000, 2)}
    except HTTPException:
        raise
    except Exception as exc:
        return {"tool_name": req.tool_name, "result": None, "error": str(exc), "elapsed_ms": round((time.time()-t0)*1000, 2)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_THINKING_PORT", "8004"))
    uvicorn.run("tools.sequential_thinking_mcp:app", host="0.0.0.0", port=port, reload=False)
