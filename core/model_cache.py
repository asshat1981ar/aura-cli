import hashlib
import sqlite3
from typing import Optional, Dict
from core.logging_utils import log_json

class ModelCache:
    def __init__(self, db_conn: Optional[sqlite3.Connection] = None, ttl_seconds: int = 3600, momento=None):
        self.db = db_conn
        self.ttl = ttl_seconds
        self.momento = momento
        self.mem_cache: Dict[str, str] = {}
        if self.db:
            self._init_db()

    def _init_db(self):
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db.commit()
            log_json("INFO", "model_cache_enabled", details={
                "ttl": self.ttl,
                "l1_momento": bool(self.momento and self.momento.is_available()),
            })
            self.preload()
        except Exception as e:
            log_json("ERROR", "model_cache_init_failed", details={"error": str(e)})

    def preload(self):
        """Loads the last 50 non-expired entries from the prompt_cache table into mem_cache."""
        if not self.db:
            return
        try:
            cursor = self.db.execute(
                "SELECT prompt_hash, response FROM prompt_cache "
                "WHERE timestamp > datetime('now', ?) "
                "ORDER BY timestamp DESC LIMIT 50",
                (f"-{self.ttl} seconds",)
            )
            rows = cursor.fetchall()
            for prompt_hash, response in rows:
                self.mem_cache[prompt_hash] = response
            log_json("INFO", "model_cache_preloaded", details={"count": len(rows)})
        except Exception as e:
            log_json("WARN", "model_cache_preload_failed", details={"error": str(e)})

    def get(self, prompt: str) -> Optional[str]:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # L0: in-memory dict (fastest)
        if prompt_hash in self.mem_cache:
            log_json("INFO", "model_cache_l0_hit", details={"prompt_hash": prompt_hash})
            return self.mem_cache[prompt_hash]

        # L1: Momento (sub-ms) — check before touching SQLite
        if self.momento and self.momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                key = f"response:{prompt_hash[:16]}"
                val = self.momento.cache_get(WORKING_MEMORY_CACHE, key)
                if val is not None:
                    log_json("INFO", "model_cache_l1_hit", details={"key": key})
                    return val
            except Exception as exc:
                log_json("WARN", "model_cache_l1_query_failed", details={"error": str(exc)})

        # L2: SQLite
        if not self.db:
            return None
        try:
            cursor = self.db.execute(
                "SELECT response FROM prompt_cache WHERE prompt_hash = ? AND timestamp > datetime('now', ?)",
                (prompt_hash, f"-{self.ttl} seconds")
            )
            row = cursor.fetchone()
            if row:
                log_json("INFO", "model_cache_hit", details={"prompt_hash": prompt_hash})
                return row[0]
        except Exception as e:
            log_json("WARN", "model_cache_query_failed", details={"error": str(e)})
        return None

    def set(self, prompt: str, response: str):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # L0: in-memory dict
        self.mem_cache[prompt_hash] = response

        # L1: Momento write-through
        if self.momento and self.momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                key = f"response:{prompt_hash[:16]}"
                self.momento.cache_set(WORKING_MEMORY_CACHE, key, response,
                                  ttl_seconds=self.ttl)
            except Exception as exc:
                log_json("WARN", "model_cache_l1_save_failed", details={"error": str(exc)})

        # L2: SQLite
        if not self.db:
            return
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO prompt_cache (prompt_hash, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (prompt_hash, response)
            )
            self.db.commit()
        except Exception as e:
            log_json("WARN", "model_cache_save_failed", details={"error": str(e)})
