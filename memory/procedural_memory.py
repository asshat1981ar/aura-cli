"""Procedural memory module -- strategy and procedure store.

Tracks which strategies the agent has tried for various task types, their
success rates, and allows recommendation of the best-performing strategies
for future similar tasks.
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
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "procedural.db")


class ProceduralMemory(MemoryModule):
    """SQLite-backed strategy store with success-rate tracking.

    Each entry represents a strategy (a named approach for a specific task
    type).  The module tracks how many times each strategy has been attempted
    and how many of those attempts succeeded, enabling data-driven strategy
    recommendation.

    Args:
        db_path: Path to the SQLite database.  Defaults to
            ``~/.aura/memory/procedural.db``.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()
        log_json(
            "INFO",
            "procedural_memory_init",
            details={"db_path": self._db_path},
        )

    # ------------------------------------------------------------------
    # Connection handling
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
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = self._get_conn()
        with self._db_lock():
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies(
                    id              TEXT PRIMARY KEY,
                    task_type       TEXT,
                    strategy_name   TEXT,
                    content         TEXT NOT NULL,
                    attempts        INTEGER DEFAULT 0,
                    successes       INTEGER DEFAULT 0,
                    metadata_json   TEXT,
                    timestamp       REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_proc_task_type ON strategies(task_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_proc_strategy ON strategies(strategy_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_proc_timestamp ON strategies(timestamp)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def write(self, content: str, metadata: Optional[dict] = None) -> str:
        """Store a new strategy.

        Expected metadata keys (all optional):
            - ``task_type``: The type of task this strategy applies to.
            - ``strategy_name``: A human-readable name for the strategy.
        """
        entry_id = uuid.uuid4().hex[:16]
        meta = dict(metadata) if metadata else {}
        task_type = meta.pop("task_type", None)
        strategy_name = meta.pop("strategy_name", None)
        now = time.time()

        conn = self._get_conn()
        with self._db_lock():
            try:
                conn.execute(
                    """INSERT INTO strategies(id, task_type, strategy_name, content,
                                             attempts, successes, metadata_json, timestamp)
                       VALUES (?, ?, ?, ?, 0, 0, ?, ?)""",
                    (
                        entry_id,
                        task_type,
                        strategy_name,
                        content,
                        json.dumps(meta) if meta else None,
                        now,
                    ),
                )
                conn.commit()
            except sqlite3.Error as exc:
                log_json(
                    "ERROR",
                    "procedural_memory_write_failed",
                    details={"error": str(exc)},
                )
                raise

        log_json(
            "INFO",
            "procedural_memory_write",
            details={
                "entry_id": entry_id,
                "task_type": task_type,
                "strategy_name": strategy_name,
            },
        )
        return entry_id

    def read(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Return the best strategies for *query* (interpreted as ``task_type``).

        Results are ordered by success rate (descending), then by number of
        attempts (descending) to favour well-tested strategies.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM strategies
               WHERE task_type = ?
               ORDER BY
                   CASE WHEN attempts > 0 THEN CAST(successes AS REAL) / attempts ELSE 0.0 END DESC,
                   attempts DESC
               LIMIT ?""",
            (query, top_k),
        ).fetchall()

        if not rows:
            # Fallback: search content with LIKE
            rows = conn.execute(
                """SELECT * FROM strategies
                   WHERE content LIKE ? OR strategy_name LIKE ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", top_k),
            ).fetchall()

        return [self._row_to_entry(r) for r in rows]

    def search(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """Search strategies by content, name, or task_type (word-level LIKE match).

        Each whitespace-separated token in *query* is matched independently
        with LIKE, and results are returned if they match **any** token.
        """
        tokens = query.strip().split()
        if not tokens:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM strategies ORDER BY timestamp DESC LIMIT ?",
                (top_k,),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

        # Build a WHERE clause that matches any token in any searchable column.
        clauses: list[str] = []
        params: list[str] = []
        for tok in tokens:
            pattern = f"%{tok}%"
            clauses.append(
                "(content LIKE ? OR strategy_name LIKE ? OR task_type LIKE ?)"
            )
            params.extend([pattern, pattern, pattern])

        where = " OR ".join(clauses)
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT * FROM strategies WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            (*params, top_k),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def delete(self, entry_id: str) -> bool:
        conn = self._get_conn()
        with self._db_lock():
            cursor = conn.execute(
                "DELETE FROM strategies WHERE id = ?", (entry_id,)
            )
            conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log_json(
                "INFO",
                "procedural_memory_delete",
                details={"entry_id": entry_id},
            )
        return deleted

    def clear(self) -> None:
        conn = self._get_conn()
        with self._db_lock():
            conn.execute("DELETE FROM strategies")
            conn.commit()
        log_json("INFO", "procedural_memory_clear")

    def stats(self) -> dict:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()
        count = row[0] if row else 0
        # Also compute aggregate success rate.
        agg = conn.execute(
            "SELECT SUM(attempts), SUM(successes) FROM strategies"
        ).fetchone()
        total_attempts = agg[0] or 0
        total_successes = agg[1] or 0
        try:
            size_bytes = os.path.getsize(self._db_path)
        except OSError:
            size_bytes = 0
        return {
            "type": "procedural",
            "count": count,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_success_rate": round(total_successes / total_attempts, 3)
            if total_attempts
            else 0.0,
            "storage_bytes": size_bytes,
        }

    # ------------------------------------------------------------------
    # Extended API -- strategy management
    # ------------------------------------------------------------------

    def recommend(self, task_type: str, top_k: int = 5) -> list[MemoryEntry]:
        """Return the best strategies for *task_type* sorted by success rate.

        Strategies that have never been tried are ranked below those with at
        least one attempt, to avoid recommending untested approaches over
        proven ones.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM strategies
               WHERE task_type = ?
               ORDER BY
                   CASE WHEN attempts > 0 THEN 1 ELSE 0 END DESC,
                   CASE WHEN attempts > 0 THEN CAST(successes AS REAL) / attempts ELSE 0.0 END DESC,
                   attempts DESC
               LIMIT ?""",
            (task_type, top_k),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def record_outcome(self, strategy_id: str, success: bool) -> None:
        """Increment the attempt counter (and success counter if *success*)."""
        conn = self._get_conn()
        with self._db_lock():
            if success:
                conn.execute(
                    "UPDATE strategies SET attempts = attempts + 1, successes = successes + 1 WHERE id = ?",
                    (strategy_id,),
                )
            else:
                conn.execute(
                    "UPDATE strategies SET attempts = attempts + 1 WHERE id = ?",
                    (strategy_id,),
                )
            conn.commit()
        log_json(
            "INFO",
            "procedural_memory_record_outcome",
            details={"strategy_id": strategy_id, "success": success},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        meta = {}
        if row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                pass
        meta["task_type"] = row["task_type"]
        meta["strategy_name"] = row["strategy_name"]
        meta["attempts"] = row["attempts"]
        meta["successes"] = row["successes"]
        success_rate = (
            round(row["successes"] / row["attempts"], 3)
            if row["attempts"] > 0
            else 0.0
        )
        meta["success_rate"] = success_rate

        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            metadata=meta,
            timestamp=row["timestamp"],
            score=success_rate,
        )
