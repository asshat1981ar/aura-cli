"""LLMCache — three-tier prompt/response cache extracted from ModelAdapter (B2).

Tiers:
  L0: In-memory LRU (OrderedDict, bounded to ``max_l0_entries``).
  L1: Momento (optional, sub-ms reads/writes via ``MomentoAdapter``).
  L2: SQLite (persistent, TTL-based expiry).
"""
from __future__ import annotations

import collections
import hashlib
from typing import Optional

from core.logging_utils import log_json

_DEFAULT_MAX_L0 = 500


class LLMCache:
    """Three-tier LLM response cache."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 3600,
        max_l0_entries: int = _DEFAULT_MAX_L0,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_l0 = max_l0_entries
        self._mem: collections.OrderedDict = collections.OrderedDict()
        self._db = None       # SQLite connection (L2)
        self._momento = None  # MomentoAdapter (L1)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def enable(self, db_conn, ttl_seconds: int = 3600, momento=None) -> None:
        """Activate L2 (SQLite) and optionally L1 (Momento) tiers."""
        self._db = db_conn
        self._ttl = ttl_seconds
        self._momento = momento
        try:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._db.commit()
            log_json("INFO", "model_cache_enabled", details={
                "ttl": ttl_seconds,
                "l1_momento": bool(momento and momento.is_available()),
            })
            self.preload()
        except Exception as e:
            log_json("ERROR", "model_cache_init_failed", details={"error": str(e)})

    def preload(self) -> None:
        """Load the last 50 non-expired entries from L2 into L0."""
        if not self._db:
            return
        try:
            cursor = self._db.execute(
                "SELECT prompt_hash, response FROM prompt_cache "
                "WHERE timestamp > datetime('now', ?) "
                "ORDER BY timestamp DESC LIMIT 50",
                (f"-{self._ttl} seconds",),
            )
            rows = cursor.fetchall()
            for prompt_hash, response in rows:
                self._mem[prompt_hash] = response
            log_json("INFO", "model_cache_preloaded", details={"count": len(rows)})
        except Exception as e:
            log_json("WARN", "model_cache_preload_failed", details={"error": str(e)})

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[str]:
        """Look up a cached response across all tiers. Returns ``None`` on miss."""
        prompt_hash = self._hash(prompt)

        # L0: in-memory OrderedDict
        if prompt_hash in self._mem:
            self._mem.move_to_end(prompt_hash)
            log_json("INFO", "model_cache_l0_hit", details={"prompt_hash": prompt_hash})
            return self._mem[prompt_hash]

        # L1: Momento
        if self._momento and self._momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                key = f"response:{prompt_hash[:16]}"
                val = self._momento.cache_get(WORKING_MEMORY_CACHE, key)
                if val is not None:
                    log_json("INFO", "model_cache_l1_hit", details={"key": key})
                    return val
            except Exception as exc:
                log_json("WARN", "model_cache_l1_query_failed", details={"error": str(exc)})

        # L2: SQLite
        if not self._db:
            return None
        try:
            cursor = self._db.execute(
                "SELECT response FROM prompt_cache WHERE prompt_hash = ? AND timestamp > datetime('now', ?)",
                (prompt_hash, f"-{self._ttl} seconds"),
            )
            row = cursor.fetchone()
            if row:
                log_json("INFO", "model_cache_hit", details={"prompt_hash": prompt_hash})
                return row[0]
        except Exception as e:
            log_json("WARN", "model_cache_query_failed", details={"error": str(e)})
        return None

    def put(self, prompt: str, response: str) -> None:
        """Store a response in all available tiers."""
        prompt_hash = self._hash(prompt)

        # L0
        self._mem[prompt_hash] = response
        self._mem.move_to_end(prompt_hash)
        while len(self._mem) > self._max_l0:
            self._mem.popitem(last=False)

        # L1
        if self._momento and self._momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                key = f"response:{prompt_hash[:16]}"
                self._momento.cache_set(WORKING_MEMORY_CACHE, key, response,
                                        ttl_seconds=self._ttl)
            except Exception as exc:
                log_json("WARN", "model_cache_l1_save_failed", details={"error": str(exc)})

        # L2
        if not self._db:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO prompt_cache (prompt_hash, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (prompt_hash, response),
            )
            self._db.commit()
        except Exception as e:
            log_json("WARN", "model_cache_save_failed", details={"error": str(e)})
