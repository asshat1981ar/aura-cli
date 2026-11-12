"""State Checkpointing System for AURA CLI workflow graphs.

Provides persistent checkpoint/resume/fork/rollback capabilities for
graph-based workflow execution.  Checkpoints capture the full graph state
(completed nodes, pending nodes, state data, optional memory snapshot) and
are integrity-protected with SHA-256 checksums.

Capability 4 -- Issue #434.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when the requested checkpoint does not exist."""


class CheckpointCorruptedError(CheckpointError):
    """Raised when a loaded checkpoint fails integrity verification."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """Immutable snapshot of a workflow graph's state at a given node.

    Attributes:
        checkpoint_id:   Unique identifier (UUID4).
        workflow_name:   Human-readable name of the workflow being executed.
        node_id:         Identifier of the current / last-completed node.
        state_data:      Arbitrary graph state dictionary.
        completed_nodes: Ordered list of nodes that have finished execution.
        pending_nodes:   List of nodes still awaiting execution.
        memory_snapshot: Optional snapshot of the memory module contents.
        metadata:        Ancillary data (timestamps, step_count, token_usage, ...).
        checksum:        SHA-256 hex digest of the deterministically serialized *state_data*.
        schema_version:  Schema version for forward-compatible deserialization.
        created_at:      Unix epoch timestamp when the checkpoint was created.
        parent_id:       If this checkpoint was forked, the ID of its parent.
    """

    checkpoint_id: str
    workflow_name: str
    node_id: str
    state_data: Dict[str, Any]
    completed_nodes: List[str]
    pending_nodes: List[str]
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    schema_version: int = 1
    created_at: float = 0.0
    parent_id: Optional[str] = None


@dataclass
class CheckpointSummary:
    """Lightweight projection of a :class:`Checkpoint` for listing."""

    checkpoint_id: str
    workflow_name: str
    node_id: str
    created_at: float
    state_size_bytes: int
    parent_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract storage interface
# ---------------------------------------------------------------------------


class CheckpointStorage(ABC):
    """Abstract interface that all checkpoint backends must implement."""

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> str:
        """Persist *checkpoint* and return its ``checkpoint_id``."""

    @abstractmethod
    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load and return the checkpoint identified by *checkpoint_id*.

        Raises:
            CheckpointNotFoundError: If no checkpoint with that ID exists.
        """

    @abstractmethod
    def list(
        self, workflow_name: Optional[str] = None, limit: int = 50
    ) -> List[CheckpointSummary]:
        """Return summaries of stored checkpoints, newest first.

        Args:
            workflow_name: If supplied, filter to this workflow only.
            limit: Maximum number of summaries to return.
        """

    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """Delete the checkpoint identified by *checkpoint_id*.

        Returns ``True`` if a checkpoint was actually removed.
        """

    @abstractmethod
    def gc(self, keep_last: int = 10) -> int:
        """Garbage-collect old checkpoints, keeping the most recent *keep_last*
        per workflow.

        Returns the total number of checkpoints deleted.
        """


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".aura" / "checkpoints"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "checkpoints.db"


class SQLiteCheckpointStorage(CheckpointStorage):
    """SQLite-backed checkpoint storage.

    Follows the same patterns as ``memory/brain.py``: WAL journal mode,
    ``PRAGMA synchronous=NORMAL``, a threading lock for write serialization,
    and ``sqlite3.Row`` for dict-like row access.

    The database is stored at ``~/.aura/checkpoints/checkpoints.db`` by
    default but can be overridden (useful for tests).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        db_file = Path(db_path) if db_path else _DEFAULT_DB_PATH
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_file
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # -- context helpers -----------------------------------------------------

    @contextmanager
    def _db_lock(self):
        """Serialize write operations across threads."""
        with self._lock:
            yield

    # -- schema --------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=10000")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id              TEXT PRIMARY KEY,
                    workflow_name   TEXT NOT NULL,
                    node_id         TEXT NOT NULL,
                    state_json      TEXT NOT NULL,
                    completed_nodes_json TEXT NOT NULL,
                    pending_nodes_json   TEXT NOT NULL,
                    memory_json     TEXT,
                    metadata_json   TEXT,
                    checksum        TEXT NOT NULL,
                    schema_version  INTEGER NOT NULL DEFAULT 1,
                    created_at      REAL NOT NULL,
                    parent_id       TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_workflow "
                "ON checkpoints(workflow_name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_created "
                "ON checkpoints(created_at)"
            )
            self._conn.commit()
        log_json(
            "INFO",
            "checkpoint_storage_init",
            details={"db_path": str(self._db_path)},
        )

    # -- public API ----------------------------------------------------------

    def save(self, checkpoint: Checkpoint) -> str:
        """Persist a checkpoint row. Returns the checkpoint_id."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    id, workflow_name, node_id, state_json,
                    completed_nodes_json, pending_nodes_json,
                    memory_json, metadata_json, checksum,
                    schema_version, created_at, parent_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.workflow_name,
                    checkpoint.node_id,
                    json.dumps(checkpoint.state_data, sort_keys=True),
                    json.dumps(checkpoint.completed_nodes),
                    json.dumps(checkpoint.pending_nodes),
                    json.dumps(checkpoint.memory_snapshot, sort_keys=True)
                    if checkpoint.memory_snapshot
                    else None,
                    json.dumps(checkpoint.metadata, sort_keys=True)
                    if checkpoint.metadata
                    else None,
                    checkpoint.checksum,
                    checkpoint.schema_version,
                    checkpoint.created_at,
                    checkpoint.parent_id,
                ),
            )
            self._conn.commit()
        log_json(
            "INFO",
            "checkpoint_saved",
            details={
                "checkpoint_id": checkpoint.checkpoint_id,
                "workflow_name": checkpoint.workflow_name,
                "node_id": checkpoint.node_id,
            },
        )
        return checkpoint.checkpoint_id

    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load a checkpoint by ID.

        Raises:
            CheckpointNotFoundError: If no matching row exists.
        """
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            raise CheckpointNotFoundError(
                f"Checkpoint '{checkpoint_id}' not found"
            )
        return self._row_to_checkpoint(row)

    def list(
        self, workflow_name: Optional[str] = None, limit: int = 50
    ) -> List[CheckpointSummary]:
        """List checkpoint summaries, newest first."""
        if workflow_name:
            rows = self._conn.execute(
                "SELECT id, workflow_name, node_id, created_at, state_json, parent_id "
                "FROM checkpoints WHERE workflow_name = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (workflow_name, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, workflow_name, node_id, created_at, state_json, parent_id "
                "FROM checkpoints ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            CheckpointSummary(
                checkpoint_id=r["id"],
                workflow_name=r["workflow_name"],
                node_id=r["node_id"],
                created_at=r["created_at"],
                state_size_bytes=len(r["state_json"].encode("utf-8")),
                parent_id=r["parent_id"],
            )
            for r in rows
        ]

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a single checkpoint. Returns True if a row was removed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,)
            )
            self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log_json(
                "INFO",
                "checkpoint_deleted",
                details={"checkpoint_id": checkpoint_id},
            )
        return deleted

    def gc(self, keep_last: int = 10) -> int:
        """Delete all but the *keep_last* most-recent checkpoints per workflow.

        Returns the total number of rows deleted.
        """
        total_deleted = 0
        # Get distinct workflows
        workflows = self._conn.execute(
            "SELECT DISTINCT workflow_name FROM checkpoints"
        ).fetchall()
        for wf_row in workflows:
            wf_name = wf_row["workflow_name"]
            # Find the created_at cutoff (the Nth newest)
            cutoff_row = self._conn.execute(
                "SELECT created_at FROM checkpoints "
                "WHERE workflow_name = ? "
                "ORDER BY created_at DESC LIMIT 1 OFFSET ?",
                (wf_name, keep_last - 1),
            ).fetchone()
            if cutoff_row is None:
                # Fewer than keep_last checkpoints for this workflow -- nothing to GC
                continue
            cutoff_ts = cutoff_row["created_at"]
            with self._lock:
                cursor = self._conn.execute(
                    "DELETE FROM checkpoints "
                    "WHERE workflow_name = ? AND created_at < ?",
                    (wf_name, cutoff_ts),
                )
                self._conn.commit()
            total_deleted += cursor.rowcount
        if total_deleted:
            log_json(
                "INFO",
                "checkpoint_gc",
                details={"deleted": total_deleted, "keep_last": keep_last},
            )
        return total_deleted

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
        """Deserialize a database row into a :class:`Checkpoint`."""
        return Checkpoint(
            checkpoint_id=row["id"],
            workflow_name=row["workflow_name"],
            node_id=row["node_id"],
            state_data=json.loads(row["state_json"]),
            completed_nodes=json.loads(row["completed_nodes_json"]),
            pending_nodes=json.loads(row["pending_nodes_json"]),
            memory_snapshot=json.loads(row["memory_json"])
            if row["memory_json"]
            else {},
            metadata=json.loads(row["metadata_json"])
            if row["metadata_json"]
            else {},
            checksum=row["checksum"],
            schema_version=row["schema_version"],
            created_at=row["created_at"],
            parent_id=row["parent_id"],
        )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# CheckpointManager -- high-level orchestration API
# ---------------------------------------------------------------------------


class CheckpointManager:
    """High-level API for creating, resuming, forking, and garbage-collecting
    workflow checkpoints.

    Args:
        storage:           Backend to use.  Defaults to :class:`SQLiteCheckpointStorage`.
        checkpoint_every:  When using :meth:`on_node_complete`, a checkpoint is
                           created every *checkpoint_every* completed nodes.
    """

    def __init__(
        self,
        storage: Optional[CheckpointStorage] = None,
        checkpoint_every: int = 1,
    ) -> None:
        self._storage = storage or SQLiteCheckpointStorage()
        self._checkpoint_every = max(1, checkpoint_every)
        self._node_counter: int = 0
        log_json(
            "INFO",
            "checkpoint_manager_init",
            details={"checkpoint_every": self._checkpoint_every},
        )

    # -- public API ----------------------------------------------------------

    def create_checkpoint(
        self,
        workflow_name: str,
        node_id: str,
        state: Dict[str, Any],
        completed_nodes: List[str],
        pending_nodes: List[str],
        memory_snapshot: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Create and persist a new checkpoint.

        Args:
            workflow_name:   Name of the executing workflow.
            node_id:         Current / last-completed graph node.
            state:           Full graph state dictionary.
            completed_nodes: Nodes that have finished.
            pending_nodes:   Nodes still queued.
            memory_snapshot: Optional memory module dump.
            metadata:        Optional dict (step_count, token_usage, ...).
            parent_id:       Set when this checkpoint is derived from another.

        Returns:
            The ``checkpoint_id`` (UUID4 string) of the newly created checkpoint.
        """
        checkpoint_id = uuid.uuid4().hex
        checksum = self._compute_checksum(state)
        now = time.time()

        meta = dict(metadata) if metadata else {}
        meta.setdefault("created_ts_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)))

        cp = Checkpoint(
            checkpoint_id=checkpoint_id,
            workflow_name=workflow_name,
            node_id=node_id,
            state_data=state,
            completed_nodes=list(completed_nodes),
            pending_nodes=list(pending_nodes),
            memory_snapshot=memory_snapshot or {},
            metadata=meta,
            checksum=checksum,
            schema_version=1,
            created_at=now,
            parent_id=parent_id,
        )
        self._storage.save(cp)
        log_json(
            "INFO",
            "checkpoint_created",
            details={
                "checkpoint_id": checkpoint_id,
                "workflow_name": workflow_name,
                "node_id": node_id,
                "state_size_bytes": len(
                    json.dumps(state, sort_keys=True).encode("utf-8")
                ),
            },
        )
        return checkpoint_id

    def resume(self, checkpoint_id: str) -> Checkpoint:
        """Load a checkpoint and verify its integrity.

        Raises:
            CheckpointNotFoundError:  If the checkpoint does not exist.
            CheckpointCorruptedError: If the checksum does not match.
        """
        cp = self._storage.load(checkpoint_id)
        self._verify_integrity(cp)
        log_json(
            "INFO",
            "checkpoint_resumed",
            details={
                "checkpoint_id": checkpoint_id,
                "workflow_name": cp.workflow_name,
                "node_id": cp.node_id,
            },
        )
        return cp

    def fork(self, checkpoint_id: str) -> str:
        """Create a copy of an existing checkpoint with a new ID.

        The new checkpoint's ``parent_id`` is set to the source checkpoint's
        ``checkpoint_id``.

        Returns:
            The ``checkpoint_id`` of the forked checkpoint.
        """
        source = self._storage.load(checkpoint_id)
        self._verify_integrity(source)

        forked_id = uuid.uuid4().hex
        now = time.time()

        meta = dict(source.metadata)
        meta["forked_from"] = checkpoint_id
        meta["forked_ts_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)
        )

        forked = Checkpoint(
            checkpoint_id=forked_id,
            workflow_name=source.workflow_name,
            node_id=source.node_id,
            state_data=source.state_data,
            completed_nodes=list(source.completed_nodes),
            pending_nodes=list(source.pending_nodes),
            memory_snapshot=dict(source.memory_snapshot)
            if source.memory_snapshot
            else {},
            metadata=meta,
            checksum=source.checksum,
            schema_version=source.schema_version,
            created_at=now,
            parent_id=checkpoint_id,
        )
        self._storage.save(forked)
        log_json(
            "INFO",
            "checkpoint_forked",
            details={
                "source_id": checkpoint_id,
                "forked_id": forked_id,
                "workflow_name": source.workflow_name,
            },
        )
        return forked_id

    def rollback(self, checkpoint_id: str) -> Checkpoint:
        """Semantic alias for :meth:`resume` -- load a previous checkpoint to
        restore workflow state to that point.

        Raises:
            CheckpointNotFoundError:  If the checkpoint does not exist.
            CheckpointCorruptedError: If the checksum does not match.
        """
        log_json(
            "INFO",
            "checkpoint_rollback",
            details={"checkpoint_id": checkpoint_id},
        )
        return self.resume(checkpoint_id)

    def list_checkpoints(
        self, workflow_name: Optional[str] = None, limit: int = 50
    ) -> List[CheckpointSummary]:
        """Return summaries of stored checkpoints, newest first."""
        return self._storage.list(workflow_name=workflow_name, limit=limit)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Remove a single checkpoint."""
        return self._storage.delete(checkpoint_id)

    def gc(self, keep_last: int = 10) -> int:
        """Garbage-collect old checkpoints, keeping *keep_last* per workflow."""
        return self._storage.gc(keep_last=keep_last)

    # -- auto-checkpoint callback -------------------------------------------

    def on_node_complete(
        self,
        workflow_name: str,
        node_id: str,
        state: Dict[str, Any],
        completed_nodes: List[str],
        pending_nodes: List[str],
        memory_snapshot: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> Optional[str]:
        """Callback intended to be invoked after each graph node completes.

        A checkpoint is only created every ``checkpoint_every`` invocations
        (configured at construction time).

        Returns:
            The ``checkpoint_id`` if a checkpoint was created, else ``None``.
        """
        self._node_counter += 1
        if self._node_counter % self._checkpoint_every != 0:
            return None
        return self.create_checkpoint(
            workflow_name=workflow_name,
            node_id=node_id,
            state=state,
            completed_nodes=completed_nodes,
            pending_nodes=pending_nodes,
            memory_snapshot=memory_snapshot,
            metadata=metadata,
            parent_id=parent_id,
        )

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _compute_checksum(state_data: Dict[str, Any]) -> str:
        """Return the SHA-256 hex digest of the deterministically serialized
        *state_data*.
        """
        canonical = json.dumps(state_data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _verify_integrity(checkpoint: Checkpoint) -> None:
        """Verify that the checkpoint's checksum matches its state data.

        Raises:
            CheckpointCorruptedError: On mismatch.
        """
        expected = hashlib.sha256(
            json.dumps(checkpoint.state_data, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if expected != checkpoint.checksum:
            log_json(
                "ERROR",
                "checkpoint_integrity_failure",
                details={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "expected": expected,
                    "actual": checkpoint.checksum,
                },
            )
            raise CheckpointCorruptedError(
                f"Checkpoint '{checkpoint.checkpoint_id}' failed integrity check: "
                f"expected checksum {expected}, got {checkpoint.checksum}"
            )
