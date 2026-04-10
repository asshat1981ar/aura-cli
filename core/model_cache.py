"""Caching logic for ModelAdapter (mixin).

Provides L0 (in-memory), L1 (Momento), and L2 (SQLite) prompt-response
caching with write-through semantics.
"""

from __future__ import annotations

import hashlib

from core.logging_utils import log_json


class CacheMixin:
    """Mixin providing prompt-response caching across three tiers."""

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def enable_cache(self, db_conn, ttl_seconds: int = 3600, momento=None):
        """Enables prompt-response caching.

        Args:
            db_conn:      SQLite connection (L2 persistent cache).
            ttl_seconds:  Cache TTL in seconds (default 1 hour).
            momento:      Optional :class:`MomentoAdapter` for L1 hot cache.
        """
        self.cache_db = db_conn
        self.cache_ttl = ttl_seconds
        self._momento = momento  # L1 cache adapter (may be None or unavailable)
        try:
            self.cache_db.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.cache_db.commit()
            log_json(
                "INFO",
                "model_cache_enabled",
                details={
                    "ttl": ttl_seconds,
                    "l1_momento": bool(momento and momento.is_available()),
                },
            )
            self.preload_cache()
        except Exception as e:
            log_json("ERROR", "model_cache_init_failed", details={"error": str(e)})

    def preload_cache(self):
        """Loads the last 50 non-expired entries from the prompt_cache table into _mem_cache."""
        if not self.cache_db:
            return
        try:
            cursor = self.cache_db.execute("SELECT prompt_hash, response FROM prompt_cache WHERE timestamp > datetime('now', ?) ORDER BY timestamp DESC LIMIT 50", (f"-{self.cache_ttl} seconds",))
            rows = cursor.fetchall()
            for prompt_hash, response in rows:
                self._mem_cache[prompt_hash] = response
            log_json("INFO", "model_cache_preloaded", details={"count": len(rows)})
        except Exception as e:
            log_json("WARN", "model_cache_preload_failed", details={"error": str(e)})

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def _get_cached_response(self, prompt: str) -> str | None:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # L0: in-memory dict (fastest)
        if prompt_hash in self._mem_cache:
            log_json("INFO", "model_cache_l0_hit", details={"prompt_hash": prompt_hash})
            return self._mem_cache[prompt_hash]

        # L1: Momento (sub-ms) -- check before touching SQLite
        momento = getattr(self, "_momento", None)
        if momento and momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE

                key = f"response:{prompt_hash[:16]}"
                val = momento.cache_get(WORKING_MEMORY_CACHE, key)
                if val is not None:
                    log_json("INFO", "model_cache_l1_hit", details={"key": key})
                    return val
            except Exception as exc:
                log_json("WARN", "model_cache_l1_query_failed", details={"error": str(exc)})

        # L2: SQLite
        if not self.cache_db:
            return None
        try:
            cursor = self.cache_db.execute("SELECT response FROM prompt_cache WHERE prompt_hash = ? AND timestamp > datetime('now', ?)", (prompt_hash, f"-{self.cache_ttl} seconds"))
            row = cursor.fetchone()
            if row:
                log_json("INFO", "model_cache_hit", details={"prompt_hash": prompt_hash})
                return row[0]
        except Exception as e:
            log_json("WARN", "model_cache_query_failed", details={"error": str(e)})
        return None

    def _save_to_cache(self, prompt: str, response: str):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # L0: in-memory dict
        self._mem_cache[prompt_hash] = response

        # L1: Momento write-through
        momento = getattr(self, "_momento", None)
        if momento and momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE

                key = f"response:{prompt_hash[:16]}"
                momento.cache_set(WORKING_MEMORY_CACHE, key, response, ttl_seconds=self.cache_ttl)
            except Exception as exc:
                log_json("WARN", "model_cache_l1_save_failed", details={"error": str(exc)})

        # L2: SQLite
        if not self.cache_db:
            return
        try:
            self.cache_db.execute("INSERT OR REPLACE INTO prompt_cache (prompt_hash, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", (prompt_hash, response))
            self.cache_db.commit()
        except Exception as e:
            log_json("WARN", "model_cache_save_failed", details={"error": str(e)})
