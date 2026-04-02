# core/agent_sdk/session_persistence.py
"""SQLite session persistence and cost tracking.

Stores session state for resumption and tracks cost per session,
per goal type, per model tier.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COST_PER_1M = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a model call."""
    rates = COST_PER_1M.get(model, COST_PER_1M["claude-sonnet-4-6"])
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE,
    goal TEXT NOT NULL,
    goal_type TEXT NOT NULL,
    workflow TEXT NOT NULL,
    model_tier TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    phases_completed INTEGER NOT NULL DEFAULT 0,
    resumed_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_pk INTEGER NOT NULL REFERENCES sessions(id),
    phase TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    model_used TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    elapsed_ms INTEGER NOT NULL DEFAULT 0,
    success BOOLEAN NOT NULL DEFAULT 1,
    error_msg TEXT,
    created_at TEXT NOT NULL
);
"""


class SessionStore:
    """SQLite-backed session persistence with cost tracking."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def create_session(
        self, session_id: str, goal: str, goal_type: str,
        workflow: str, model_tier: str,
    ) -> int:
        """Create a new session. Returns the primary key."""
        now = datetime.utcnow().isoformat()
        cur = self._conn.execute(
            """INSERT INTO sessions (session_id, goal, goal_type, workflow,
               model_tier, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
            (session_id, goal, goal_type, workflow, model_tier, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_event(
        self, session_pk: int, phase: str, tool_name: str,
        model: str, tokens_in: int, tokens_out: int,
        success: bool, error: Optional[str],
    ) -> None:
        """Record a cycle event and update session totals."""
        cost = compute_cost(model, tokens_in, tokens_out)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO cycle_events (session_pk, phase, tool_name, model_used,
               input_tokens, output_tokens, cost_usd, success, error_msg, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_pk, phase, tool_name, model, tokens_in, tokens_out,
             cost, success, error, now),
        )
        self._conn.execute(
            """UPDATE sessions SET
               total_cost_usd = total_cost_usd + ?,
               total_input_tokens = total_input_tokens + ?,
               total_output_tokens = total_output_tokens + ?,
               phases_completed = phases_completed + ?,
               updated_at = ?
               WHERE id = ?""",
            (cost, tokens_in, tokens_out, 1 if success else 0, now, session_pk),
        )
        self._conn.commit()

    def update_status(
        self, session_pk: int, status: str, error_summary: Optional[str] = None,
    ) -> None:
        """Update session status."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """UPDATE sessions SET status = ?, error_summary = ?, updated_at = ?
               WHERE id = ?""",
            (status, error_summary, now, session_pk),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by SDK session ID."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self, status: Optional[str] = None, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """Cost breakdown by goal_type and model for recent sessions."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """SELECT goal_type, model_tier, COUNT(*) as count,
               SUM(total_cost_usd) as total_cost
               FROM sessions WHERE created_at >= ?
               GROUP BY goal_type, model_tier""",
            (cutoff,),
        ).fetchall()
        return {"period_days": days, "breakdown": [dict(r) for r in rows]}

    def get_resumable(self) -> List[Dict[str, Any]]:
        """Get paused sessions that can be resumed."""
        return self.list_sessions(status="paused")
