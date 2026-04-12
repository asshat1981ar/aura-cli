"""Durable session persistence for SADD — SQLite-backed checkpoints, events, and resume."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.sadd.types import (
    DesignSpec,
    SessionConfig,
    SessionReport,
)

# Default DB location (alongside other memory DBs).
_DEFAULT_DB_PATH = Path("memory/sadd_sessions.db")


def _open_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the SADD session database."""
    path = db_path or _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            spec_json   TEXT NOT NULL,
            config_json TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            graph_json  TEXT,
            report_json TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(id),
            graph_json  TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(id),
            ws_id       TEXT,
            event_type  TEXT NOT NULL,
            payload     TEXT,
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(id),
            ws_id       TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id)")
    conn.commit()
    return conn


@contextmanager
def _db(db_path: Path | None = None):
    conn = _open_db(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class SessionStore:
    """Persistent storage for SADD sessions with checkpoints and event logging."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        # Ensure schema exists on init.
        conn = _open_db(self._db_path)
        conn.close()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        spec: DesignSpec,
        config: SessionConfig,
        session_id: str | None = None,
    ) -> str:
        """Create a new session record. Returns session_id."""
        sid = session_id or str(uuid.uuid4())
        now = time.time()
        with _db(self._db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, spec_json, config_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (sid, spec.title, json.dumps(spec.to_dict()), json.dumps(asdict(config)), now, now),
            )
        self.log_event(sid, None, "session_created", {"title": spec.title})
        return sid

    def update_status(self, session_id: str, status: str) -> None:
        """Update session status (pending, running, completed, failed)."""
        with _db(self._db_path) as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), session_id),
            )

    def save_report(self, session_id: str, report: SessionReport) -> None:
        """Save the final session report."""
        with _db(self._db_path) as conn:
            conn.execute(
                "UPDATE sessions SET report_json = ?, status = 'completed', updated_at = ? WHERE id = ?",
                (json.dumps(report.to_dict()), time.time(), session_id),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session record by ID."""
        with _db(self._db_path) as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            return dict(row)

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions (most recent first)."""
        with _db(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, title, status, created_at, updated_at FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        session_id: str,
        graph_state: dict[str, Any],
        results: dict[str, Any],
    ) -> int:
        """Save a checkpoint of the graph state and accumulated results.

        Returns the checkpoint ID.
        """
        now = time.time()
        with _db(self._db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO checkpoints (session_id, graph_json, results_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, json.dumps(graph_state), json.dumps(results), now),
            )
            # Also update the session's graph_json for quick access.
            conn.execute(
                "UPDATE sessions SET graph_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(graph_state), now, session_id),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def load_latest_checkpoint(self, session_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Load the most recent checkpoint for a session.

        Returns (graph_state, results) or None if no checkpoint exists.
        """
        with _db(self._db_path) as conn:
            row = conn.execute(
                "SELECT graph_json, results_json FROM checkpoints WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["graph_json"]), json.loads(row["results_json"])

    def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a session."""
        with _db(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, created_at FROM checkpoints WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def log_event(
        self,
        session_id: str,
        ws_id: str | None,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log a session event."""
        with _db(self._db_path) as conn:
            conn.execute(
                "INSERT INTO events (session_id, ws_id, event_type, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, ws_id, event_type, json.dumps(payload) if payload else None, time.time()),
            )

    def get_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve events for a session, optionally filtered by type."""
        with _db(self._db_path) as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM events WHERE session_id = ? AND event_type = ? ORDER BY id DESC LIMIT ?",
                    (session_id, event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def record_artifact(self, session_id: str, ws_id: str, file_path: str) -> None:
        """Record a file artifact produced by a workstream."""
        with _db(self._db_path) as conn:
            conn.execute(
                "INSERT INTO artifacts (session_id, ws_id, file_path, created_at) VALUES (?, ?, ?, ?)",
                (session_id, ws_id, file_path, time.time()),
            )

    def get_artifacts(self, session_id: str, ws_id: str | None = None) -> list[dict[str, Any]]:
        """Get artifacts for a session, optionally filtered by workstream."""
        with _db(self._db_path) as conn:
            if ws_id:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE session_id = ? AND ws_id = ? ORDER BY id",
                    (session_id, ws_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def resumable_sessions(self) -> list[dict[str, Any]]:
        """List sessions that can be resumed (status = 'running' or have checkpoints)."""
        with _db(self._db_path) as conn:
            rows = conn.execute(
                "SELECT s.id, s.title, s.status, s.created_at, s.updated_at, "
                "  (SELECT COUNT(*) FROM checkpoints c WHERE c.session_id = s.id) AS checkpoint_count "
                "FROM sessions s "
                "WHERE s.status IN ('running', 'pending') OR "
                "  EXISTS (SELECT 1 FROM checkpoints c WHERE c.session_id = s.id) "
                "ORDER BY s.updated_at DESC",
            ).fetchall()
            return [dict(r) for r in rows]

    def load_session_for_resume(self, session_id: str) -> tuple[DesignSpec, SessionConfig, dict[str, Any], dict[str, Any]] | None:
        """Load everything needed to resume a session.

        Returns (spec, config, graph_state, results) or None.
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        spec = DesignSpec.from_dict(json.loads(session["spec_json"]))
        config = SessionConfig(**json.loads(session["config_json"]))

        checkpoint = self.load_latest_checkpoint(session_id)
        if checkpoint:
            graph_state, results = checkpoint
        else:
            graph_state = {}
            results = {}

        return spec, config, graph_state, results
