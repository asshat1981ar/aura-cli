"""Feedback collection for autonomous learning."""

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExecutionStatus(Enum):
    """Status of an execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


@dataclass
class ExecutionOutcome:
    """Outcome of a single execution."""
    agent_name: str
    goal: str
    status: ExecutionStatus
    duration_ms: float
    output_quality: float  # 0.0 to 1.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "goal": self.goal,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "output_quality": self.output_quality,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionOutcome":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            agent_name=data["agent_name"],
            goal=data["goal"],
            status=ExecutionStatus(data["status"]),
            duration_ms=data["duration_ms"],
            output_quality=data["output_quality"],
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class FeedbackCollector:
    """Collect and store execution feedback."""
    
    DEFAULT_DB_PATH = "~/.aura/learning_feedback.db"
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or self.DEFAULT_DB_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize feedback database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_outcomes (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    output_quality REAL NOT NULL,
                    error_message TEXT,
                    metadata TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_name 
                ON execution_outcomes(agent_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON execution_outcomes(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON execution_outcomes(status)
            """)
            conn.commit()
    
    def record(self, outcome: ExecutionOutcome) -> str:
        """Record an execution outcome."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO execution_outcomes
                (id, agent_name, goal, status, duration_ms, output_quality, 
                 error_message, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.id,
                    outcome.agent_name,
                    outcome.goal,
                    outcome.status.value,
                    outcome.duration_ms,
                    outcome.output_quality,
                    outcome.error_message,
                    json.dumps(outcome.metadata),
                    outcome.timestamp.timestamp(),
                ),
            )
            conn.commit()
        return outcome.id
    
    def get_recent(
        self,
        agent_name: Optional[str] = None,
        limit: int = 100,
        status: Optional[ExecutionStatus] = None,
    ) -> List[ExecutionOutcome]:
        """Get recent execution outcomes."""
        query = "SELECT * FROM execution_outcomes WHERE 1=1"
        params = []
        
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [self._row_to_outcome(row) for row in rows]
    
    def get_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics."""
        query = "SELECT status, COUNT(*), AVG(duration_ms), AVG(output_quality) FROM execution_outcomes"
        params = []
        
        if agent_name:
            query += " WHERE agent_name = ?"
            params.append(agent_name)
        
        query += " GROUP BY status"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        stats = {
            "total": 0,
            "by_status": {},
            "avg_duration_ms": 0,
            "avg_quality": 0,
        }
        
        total_count = 0
        total_duration = 0
        total_quality = 0
        
        for status, count, avg_duration, avg_quality in rows:
            stats["by_status"][status] = {
                "count": count,
                "avg_duration_ms": avg_duration or 0,
                "avg_quality": avg_quality or 0,
            }
            total_count += count
            total_duration += (avg_duration or 0) * count
            total_quality += (avg_quality or 0) * count
        
        stats["total"] = total_count
        if total_count > 0:
            stats["avg_duration_ms"] = total_duration / total_count
            stats["avg_quality"] = total_quality / total_count
        
        return stats
    
    def get_success_rate(self, agent_name: Optional[str] = None) -> float:
        """Get success rate for an agent or overall."""
        query = """
            SELECT 
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                COUNT(*) as total
            FROM execution_outcomes
        """
        params = []
        
        if agent_name:
            query += " WHERE agent_name = ?"
            params.append(agent_name)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
        
        if row and row[1] > 0:
            return row[0] / row[1]
        return 0.0
    
    def clear_old(self, days: int = 30):
        """Clear outcomes older than specified days."""
        cutoff = time.time() - (days * 24 * 3600)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "DELETE FROM execution_outcomes WHERE timestamp < ?",
                (cutoff,)
            )
            conn.commit()
    
    def _row_to_outcome(self, row) -> ExecutionOutcome:
        """Convert database row to ExecutionOutcome."""
        return ExecutionOutcome(
            id=row[0],
            agent_name=row[1],
            goal=row[2],
            status=ExecutionStatus(row[3]),
            duration_ms=row[4],
            output_quality=row[5],
            error_message=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
            timestamp=datetime.fromtimestamp(row[8]),
        )
