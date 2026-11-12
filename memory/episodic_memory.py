"""Episodic memory module -- SQLite-backed task trace storage.

Stores structured records of tasks the agent has executed: what was attempted,
what actions were taken, the outcome, and any lessons learned.  Uses FTS5
full-text search for efficient retrieval.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from core.logging_utils import log_json
from memory.memory_module import MemoryEntry, MemoryModule

_DEFAULT_DB_DIR = os.path.join(os.path.expanduser("~"), ".aura", "memory")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "episodic.db")


class EpisodicMemory(MemoryModule):
    """SQLite-backed task-trace memory with FTS5 full-text search.

    Each entry records a task episode: description, actions taken, outcome,
    and lessons learned.  Entries are indexed by ``task_id``, ``task_type``,
    ``outcome``, and ``timestamp`` for efficient filtering, and the ``content``
    column is indexed via FTS5 for free-text search.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``~/.aura/memory/episodic.db``.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()
        log_json(
            "INFO",
            "episodic_memory_init",
            details={"db_path": self._db_path},
        )

    # ------------------------------------------------------------------
    # Connection handling (per-thread connections for thread safety)
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def _db_lock(self):
        """Serialise write operations across threads."""
        with self._lock:
            yield

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indices if they do not exist."""
        conn = self._get_conn()
        with self._db_lock():
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries(
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    task_id     TEXT,
                    task_type   TEXT,
                    outcome     TEXT,
                    timestamp   REAL NOT NULL,
                    metadata_json TEXT
                )
            """)
            # Indices for common query patterns
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_task_id ON entries(task_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_timestamp ON entries(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_outcome ON entries(outcome)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_task_type ON entries(task_type)"
            )
            # FTS5 virtual table for full-text search on content
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
                USING fts5(content, content=entries, content_rowid=rowid)
            """)
            # Triggers to keep FTS index in sync
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
                    INSERT INTO entries_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
                    INSERT INTO entries_fts(entries_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
                    INSERT INTO entries_fts(entries_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                    INSERT INTO entries_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END;
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def write(self, content: str, metadata: Optional[dict] = None) -> str:
        """Store a new task-trace entry.

        Expected metadata keys (all optional):
            - ``task_id``: Identifier of the originating task.
            - ``task_type``: Category of the task (e.g. ``"code_fix"``).
            - ``outcome``: Result descriptor (e.g. ``"success"``, ``"failure"``).
        """
        entry_id = uuid.uuid4().hex[:16]
        meta = metadata or {}
        task_id = meta.pop("task_id", None)
        task_type = meta.pop("task_type", None)
        outcome = meta.pop("outcome", None)
        now = time.time()

        conn = self._get_conn()
        with self._db_lock():
            try:
                conn.execute(
                    """INSERT INTO entries(id, content, task_id, task_type, outcome, timestamp, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry_id,
                        content,
                        task_id,
                        task_type,
                        outcome,
                        now,
                        json.dumps(meta) if meta else None,
                    ),
                )
                conn.commit()
            except sqlite3.Error as exc:
                log_json(
                    "ERROR",
                    "episodic_memory_write_failed",
                    details={"error": str(exc), "entry_id": entry_id},
                )
                raise

        log_json(
            "INFO",
            "episodic_memory_write",
            details={
                "entry_id": entry_id,
                "task_id": task_id,
                "task_type": task_type,
            },
        )
        return entry_id

    def read(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search for task traces using FTS5 full-text search.

        Falls back to a ``LIKE`` query if FTS matching fails (e.g. for very
        short or special-character queries).
        """
        return self._fts_search(query, top_k)

    def search(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """Alias for :meth:`read` with a larger default ``top_k``."""
        return self._fts_search(query, top_k)

    def delete(self, entry_id: str) -> bool:
        """Delete a single entry by ID."""
        conn = self._get_conn()
        with self._db_lock():
            cursor = conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log_json(
                "INFO",
                "episodic_memory_delete",
                details={"entry_id": entry_id},
            )
        return deleted

    def clear(self) -> None:
        """Remove all entries (both main table and FTS index)."""
        conn = self._get_conn()
        with self._db_lock():
            conn.execute("DELETE FROM entries")
            conn.execute("DELETE FROM entries_fts")
            conn.commit()
        log_json("INFO", "episodic_memory_clear")

    def stats(self) -> dict:
        """Return entry count and database size."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        count = row[0] if row else 0
        try:
            size_bytes = os.path.getsize(self._db_path)
        except OSError:
            size_bytes = 0
        return {
            "type": "episodic",
            "count": count,
            "storage_bytes": size_bytes,
        }

    # ------------------------------------------------------------------
    # Extended query helpers
    # ------------------------------------------------------------------

    def query_by_task(self, task_id: str, limit: int = 50) -> list[MemoryEntry]:
        """Retrieve all entries for a specific task, newest first."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entries WHERE task_id = ? ORDER BY timestamp DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def query_by_outcome(
        self, outcome: str, limit: int = 50
    ) -> list[MemoryEntry]:
        """Retrieve entries filtered by outcome."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entries WHERE outcome = ? ORDER BY timestamp DESC LIMIT ?",
            (outcome, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fts_search(self, query: str, top_k: int) -> list[MemoryEntry]:
        """Run an FTS5 query, falling back to LIKE on error."""
        conn = self._get_conn()
        # Sanitize query for FTS5: wrap each token in double quotes to treat
        # them as literal phrases, avoiding FTS syntax errors.
        tokens = query.strip().split()
        if not tokens:
            # Empty query -- return most recent entries.
            rows = conn.execute(
                "SELECT * FROM entries ORDER BY timestamp DESC LIMIT ?",
                (top_k,),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        try:
            rows = conn.execute(
                """SELECT e.*, rank
                   FROM entries_fts f
                   JOIN entries e ON e.rowid = f.rowid
                   WHERE entries_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, top_k),
            ).fetchall()
            return [self._row_to_entry(r, score=-r["rank"]) for r in rows]
        except sqlite3.OperationalError:
            # Fallback: simple LIKE search
            like_pattern = f"%{query}%"
            rows = conn.execute(
                "SELECT * FROM entries WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (like_pattern, top_k),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row, score: float = 0.0) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        meta = {}
        if row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                pass
        # Re-attach task-level fields into metadata for callers.
        if row["task_id"]:
            meta["task_id"] = row["task_id"]
        if row["task_type"]:
            meta["task_type"] = row["task_type"]
        if row["outcome"]:
            meta["outcome"] = row["outcome"]

        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            metadata=meta,
            timestamp=row["timestamp"],
            score=score,
        )
