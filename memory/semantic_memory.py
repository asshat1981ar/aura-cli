"""Semantic memory module -- BM25-based keyword/similarity search.

Stores factual knowledge (concepts, definitions, learned information) and
retrieves it using BM25 scoring.  This is a pure-Python, zero-dependency
implementation suitable for v1; a future version may integrate dense
embeddings for true semantic similarity.
"""

import json
import math
import os
import re
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
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "semantic.db")

# -----------------------------------------------------------------
# Tokeniser & BM25 constants
# -----------------------------------------------------------------

# English stopwords (kept small to avoid bloating the module).
_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "if", "while", "about", "up", "its", "it", "this",
    "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 free parameters (Okapi defaults).
_BM25_K1 = 1.5
_BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric chars, remove stopwords."""
    return [
        tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS
    ]


class SemanticMemory(MemoryModule):
    """SQLite-backed knowledge store with BM25 ranking.

    Content is tokenised on write and stored alongside the raw text.  At
    search time the query is tokenised the same way and BM25 scores are
    computed in Python against the pre-tokenised corpus.

    Args:
        db_path: Path to the SQLite database.  Defaults to
            ``~/.aura/memory/semantic.db``.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()
        log_json(
            "INFO",
            "semantic_memory_init",
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
                CREATE TABLE IF NOT EXISTS entries(
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    tokens_json TEXT,
                    metadata_json TEXT,
                    timestamp   REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sem_timestamp ON entries(timestamp)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def write(self, content: str, metadata: Optional[dict] = None) -> str:
        """Store a fact with pre-computed tokens for BM25 scoring."""
        entry_id = uuid.uuid4().hex[:16]
        tokens = tokenize(content)
        now = time.time()

        conn = self._get_conn()
        with self._db_lock():
            try:
                conn.execute(
                    """INSERT INTO entries(id, content, tokens_json, metadata_json, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        entry_id,
                        content,
                        json.dumps(tokens),
                        json.dumps(metadata) if metadata else None,
                        now,
                    ),
                )
                conn.commit()
            except sqlite3.Error as exc:
                log_json(
                    "ERROR",
                    "semantic_memory_write_failed",
                    details={"error": str(exc)},
                )
                raise

        log_json(
            "INFO",
            "semantic_memory_write",
            details={"entry_id": entry_id, "token_count": len(tokens)},
        )
        return entry_id

    def read(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve the most relevant entries using BM25 scoring."""
        return self._bm25_search(query, top_k)

    def search(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """Alias for :meth:`read` with a larger default ``top_k``."""
        return self._bm25_search(query, top_k)

    def delete(self, entry_id: str) -> bool:
        conn = self._get_conn()
        with self._db_lock():
            cursor = conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log_json(
                "INFO",
                "semantic_memory_delete",
                details={"entry_id": entry_id},
            )
        return deleted

    def clear(self) -> None:
        conn = self._get_conn()
        with self._db_lock():
            conn.execute("DELETE FROM entries")
            conn.commit()
        log_json("INFO", "semantic_memory_clear")

    def stats(self) -> dict:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        count = row[0] if row else 0
        try:
            size_bytes = os.path.getsize(self._db_path)
        except OSError:
            size_bytes = 0
        return {
            "type": "semantic",
            "count": count,
            "storage_bytes": size_bytes,
        }

    # ------------------------------------------------------------------
    # BM25 implementation
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list[MemoryEntry]:
        """Compute BM25 scores for *query* against all stored documents."""
        query_tokens = tokenize(query)
        if not query_tokens:
            # No meaningful tokens -- return most recent entries.
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM entries ORDER BY timestamp DESC LIMIT ?",
                (top_k,),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, content, tokens_json, metadata_json, timestamp FROM entries"
        ).fetchall()

        if not rows:
            return []

        # Pre-compute corpus-level statistics.
        n_docs = len(rows)
        # Average document length (in tokens).
        doc_lengths: list[int] = []
        doc_token_lists: list[list[str]] = []
        for r in rows:
            tokens = json.loads(r["tokens_json"]) if r["tokens_json"] else []
            doc_token_lists.append(tokens)
            doc_lengths.append(len(tokens))

        avgdl = sum(doc_lengths) / max(n_docs, 1)

        # Document frequency for each query token.
        df: dict[str, int] = {}
        for qt in query_tokens:
            count = sum(1 for tokens in doc_token_lists if qt in tokens)
            df[qt] = count

        # Score each document.
        scored: list[tuple[float, int]] = []
        for idx, tokens in enumerate(doc_token_lists):
            dl = doc_lengths[idx]
            score = 0.0
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for qt in query_tokens:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                n_qt = df.get(qt, 0)
                # IDF component (with +1 smoothing to avoid negatives).
                idf = math.log((n_docs - n_qt + 0.5) / (n_qt + 0.5) + 1.0)
                # TF component.
                tf_norm = (tf * (_BM25_K1 + 1)) / (
                    tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avgdl, 1))
                )
                score += idf * tf_norm

            if score > 0:
                scored.append((score, idx))

        # Sort descending by score and take top_k.
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[MemoryEntry] = []
        for score, idx in scored[:top_k]:
            row = rows[idx]
            results.append(self._row_to_entry(row, score=score))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row, score: float = 0.0) -> MemoryEntry:
        meta = {}
        if row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                pass
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            metadata=meta,
            timestamp=row["timestamp"],
            score=score,
        )
