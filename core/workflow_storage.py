import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from core.logging_utils import log_json
from core.workflow_models import WorkflowExecution, AgenticLoop

_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "workflow_engine.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS executions (
            id          TEXT PRIMARY KEY,
            workflow    TEXT NOT NULL,
            status      TEXT NOT NULL,
            step_index  INTEGER NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            summary     TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loops (
            id          TEXT PRIMARY KEY,
            goal        TEXT NOT NULL,
            max_cycles  INTEGER NOT NULL,
            current_cycle INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL,
            score       REAL NOT NULL DEFAULT 0,
            stop_reason TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


@contextmanager
def db_connection():
    conn = _open_db()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def journal_execution(exc: WorkflowExecution) -> None:
    try:
        with db_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO executions
                   (id, workflow, status, step_index, created_at, updated_at, summary)
                   VALUES (?,?,?,?,?,?,?)""",
                (exc.id, exc.workflow_name, exc.status, exc.current_step_index,
                 exc.started_at, exc.updated_at,
                 f"{len(exc.history)} steps run, error={exc.error}"),
            )
    except Exception as e:
        log_json("WARN", "workflow_journal_failed", details={"error": str(e)})


def journal_loop(loop: AgenticLoop) -> None:
    try:
        with db_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO loops
                   (id, goal, max_cycles, current_cycle, status, score,
                    stop_reason, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (loop.id, loop.goal, loop.max_cycles, loop.current_cycle,
                 loop.status, loop.score, loop.stop_reason,
                 loop.started_at, loop.updated_at),
            )
    except Exception as e:
        log_json("WARN", "loop_journal_failed", details={"error": str(e)})


def get_execution_status(execution_id: str) -> Optional[Dict[str, Any]]:
    try:
        with db_connection() as conn:
            row = conn.execute(
                "SELECT id, workflow, status, step_index, created_at, updated_at, summary FROM executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
            if row:
                return dict(row)
    except Exception as e:
        log_json("WARN", "get_execution_failed", details={"error": str(e)})
    return None


def get_loop_status(loop_id: str) -> Optional[Dict[str, Any]]:
    try:
        with db_connection() as conn:
            row = conn.execute(
                "SELECT id, goal, max_cycles, current_cycle, status, score, stop_reason, created_at, updated_at FROM loops WHERE id = ?",
                (loop_id,),
            ).fetchone()
            if row:
                return dict(row)
    except Exception as e:
        log_json("WARN", "get_loop_failed", details={"error": str(e)})
    return None
