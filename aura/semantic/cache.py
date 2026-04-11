"""Cache for semantic analysis results."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional


class AnalysisCache:
    """SQLite-backed cache for semantic analysis results."""

    DEFAULT_DB_PATH = "~/.aura/semantic_cache.db"
    DEFAULT_TTL = 3600  # 1 hour

    def __init__(self, db_path: Optional[str] = None, ttl: int = DEFAULT_TTL):
        self.db_path = Path(db_path or self.DEFAULT_DB_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._init_db()

    def _init_db(self):
        """Initialize cache database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    result TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path 
                ON analysis_cache(file_path)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON analysis_cache(timestamp)
            """)
            conn.commit()

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for cache invalidation."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, file_path: str, content: str) -> Optional[Dict]:
        """Get cached analysis result if valid."""
        content_hash = self._compute_hash(content)
        key = f"{file_path}:{content_hash}"

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("SELECT result, timestamp, ttl FROM analysis_cache WHERE key = ?", (key,))
            row = cursor.fetchone()

            if row:
                result, timestamp, ttl = row
                # Check if expired
                if time.time() - timestamp < ttl:
                    return json.loads(result)
                else:
                    # Delete expired entry
                    conn.execute("DELETE FROM analysis_cache WHERE key = ?", (key,))
                    conn.commit()

        return None

    def set(self, file_path: str, content: str, result: Dict, ttl: Optional[int] = None):
        """Cache analysis result."""
        content_hash = self._compute_hash(content)
        key = f"{file_path}:{content_hash}"

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_cache
                (key, file_path, content_hash, result, timestamp, ttl)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    file_path,
                    content_hash,
                    json.dumps(result),
                    time.time(),
                    ttl or self.ttl,
                ),
            )
            conn.commit()

    def invalidate(self, file_path: Optional[str] = None):
        """Invalidate cache entries."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if file_path:
                conn.execute("DELETE FROM analysis_cache WHERE file_path = ?", (file_path,))
            else:
                conn.execute("DELETE FROM analysis_cache")
            conn.commit()

    def clear_expired(self):
        """Clear all expired entries."""
        current_time = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM analysis_cache WHERE (? - timestamp) > ttl", (current_time,))
            conn.commit()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()[0]
            expired = conn.execute("SELECT COUNT(*) FROM analysis_cache WHERE (? - timestamp) > ttl", (time.time(),)).fetchone()[0]
            return {
                "total_entries": total,
                "expired_entries": expired,
                "valid_entries": total - expired,
            }
