"""Command queue management with SQLite persistence."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import CommandPriority, CommandStatus, QueuedCommand


class CommandQueue:
    """File-backed command queue using SQLite."""
    
    DEFAULT_DB_PATH = "~/.aura/offline_queue.db"
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or self.DEFAULT_DB_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_queue (
                    id TEXT PRIMARY KEY,
                    command TEXT NOT NULL,
                    args TEXT NOT NULL,
                    kwargs TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority 
                ON command_queue(status, priority, created_at)
            """)
            conn.commit()
    
    async def add(self, command: QueuedCommand) -> str:
        """Add a command to the queue."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO command_queue
                (id, command, args, kwargs, priority, status, created_at, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command.id,
                    command.command,
                    json.dumps(list(command.args)),
                    json.dumps(command.kwargs),
                    command.priority.value,
                    command.status.value,
                    command.created_at.timestamp(),
                    command.retry_count,
                ),
            )
            conn.commit()
        return command.id
    
    async def get_pending(self, limit: int = 100) -> list[QueuedCommand]:
        """Get pending commands ordered by priority."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM command_queue
                WHERE status = ?
                ORDER BY priority ASC, created_at ASC
                LIMIT ?
                """,
                (CommandStatus.PENDING.value, limit),
            )
            return [self._row_to_command(row) for row in cursor.fetchall()]
    
    async def update_status(self, command_id: str, status: CommandStatus):
        """Update command status."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE command_queue SET status = ? WHERE id = ?",
                (status.value, command_id),
            )
            conn.commit()
    
    async def remove(self, command_id: str) -> bool:
        """Remove a command from the queue."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM command_queue WHERE id = ?",
                (command_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
    
    async def clear(self):
        """Clear all commands from queue."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM command_queue")
            conn.commit()
    
    async def size(self) -> int:
        """Get number of pending commands."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM command_queue WHERE status = ?",
                (CommandStatus.PENDING.value,),
            )
            return cursor.fetchone()[0]
    
    def _row_to_command(self, row) -> QueuedCommand:
        """Convert database row to QueuedCommand."""
        return QueuedCommand(
            id=row[0],
            command=row[1],
            args=tuple(json.loads(row[2])),
            kwargs=json.loads(row[3]),
            priority=CommandPriority(row[4]),
            status=CommandStatus(row[5]),
            created_at=datetime.fromtimestamp(row[6]),
            retry_count=row[7],
        )
