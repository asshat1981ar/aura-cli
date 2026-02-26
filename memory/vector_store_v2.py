"""ASCM v2: Semantic Context Manager with provenance, metadata, token budgets."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    """A unit of semantic memory with provenance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source_type: str = "memory"
    source_ref: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    goal_id: Optional[str] = None
    agent_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    token_count: int = 0
    embedding_model: str = ""
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not self.token_count and self.content:
            self.token_count = max(1, len(self.content) // 4)


@dataclass
class RetrievalQuery:
    """Parameters for a semantic search."""
    query_text: str
    budget_tokens: int = 4000
    min_score: float = 0.65
    filters: Dict[str, Any] = field(default_factory=dict)
    recency_bias: float = 0.1
    dedupe: bool = True


@dataclass
class SearchHit:
    """A single result from a semantic search."""
    record: MemoryRecord
    score: float
    explanation: str = ""


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


def _keyword_score(query_text: str, content: str) -> float:
    if not content:
        return 0.0
    q_tokens = set(query_text.lower().split())
    c_tokens = set(content.lower().split())
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return overlap / len(q_tokens)


class VectorStoreV2:
    """SQLite-backed semantic memory store with embedding support."""

    DEFAULT_DB = os.path.join(os.path.dirname(__file__), "ascm_v2.db")

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or self.DEFAULT_DB
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT 'memory',
                    source_ref TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    goal_id TEXT,
                    agent_name TEXT,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 1.0,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    record_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dims INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    PRIMARY KEY (record_id, model_id),
                    FOREIGN KEY (record_id) REFERENCES memory_records(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_mr_goal_id ON memory_records(goal_id);
                CREATE INDEX IF NOT EXISTS idx_mr_source_type ON memory_records(source_type);
                CREATE INDEX IF NOT EXISTS idx_mr_content_hash ON memory_records(content_hash);
                CREATE INDEX IF NOT EXISTS idx_mr_created_at ON memory_records(created_at DESC);
            """)
            conn.commit()
        except Exception:
            pass

    def _row_to_record(self, row) -> MemoryRecord:
        tags = []
        try:
            tags = json.loads(row["tags"] or "[]")
        except Exception:
            pass
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            goal_id=row["goal_id"],
            agent_name=row["agent_name"],
            tags=tags,
            importance=row["importance"],
            token_count=row["token_count"],
            embedding_model=row["embedding_model"],
            content_hash=row["content_hash"],
        )

    def upsert(self, record: MemoryRecord, embedding: Optional[list] = None) -> str:
        """Insert or update a record. Deduplicates by content_hash."""
        try:
            conn = self._get_conn()
            # Ensure hash is set
            if not record.content_hash and record.content:
                record.content_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
            if not record.token_count and record.content:
                record.token_count = max(1, len(record.content) // 4)

            # Check for existing record with same content_hash
            existing = conn.execute(
                "SELECT id FROM memory_records WHERE content_hash = ?",
                (record.content_hash,),
            ).fetchone()

            if existing:
                existing_id = existing["id"]
                conn.execute(
                    """UPDATE memory_records SET
                        content=?, source_type=?, source_ref=?, updated_at=?,
                        goal_id=?, agent_name=?, tags=?, importance=?,
                        token_count=?, embedding_model=?
                       WHERE id=?""",
                    (
                        record.content, record.source_type, record.source_ref,
                        time.time(), record.goal_id, record.agent_name,
                        json.dumps(record.tags), record.importance,
                        record.token_count, record.embedding_model, existing_id,
                    ),
                )
                conn.commit()
                if embedding:
                    self._upsert_embedding(existing_id, record.embedding_model or "unknown", embedding)
                return existing_id
            else:
                if not record.id:
                    record.id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO memory_records
                       (id, content, source_type, source_ref, created_at, updated_at,
                        goal_id, agent_name, tags, importance, token_count,
                        embedding_model, content_hash)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record.id, record.content, record.source_type, record.source_ref,
                        record.created_at, record.updated_at, record.goal_id,
                        record.agent_name, json.dumps(record.tags), record.importance,
                        record.token_count, record.embedding_model, record.content_hash,
                    ),
                )
                conn.commit()
                if embedding:
                    self._upsert_embedding(record.id, record.embedding_model or "unknown", embedding)
                return record.id
        except Exception:
            return record.id or ""

    def _upsert_embedding(self, record_id: str, model_id: str, embedding: list):
        try:
            import json as _json
            blob = _json.dumps(embedding).encode("utf-8")
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO embeddings (record_id, model_id, dims, data)
                   VALUES (?, ?, ?, ?)""",
                (record_id, model_id, len(embedding), blob),
            )
            conn.commit()
        except Exception:
            pass

    def _load_embedding(self, record_id: str, model_id: str) -> Optional[list]:
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT data FROM embeddings WHERE record_id=? AND model_id=?",
                (record_id, model_id),
            ).fetchone()
            if row:
                return json.loads(row["data"].decode("utf-8"))
        except Exception:
            pass
        return None

    def search(self, query: RetrievalQuery, query_embedding: Optional[list] = None) -> List[SearchHit]:
        """Search records. Uses cosine similarity if embedding provided, else keyword fallback."""
        try:
            conn = self._get_conn()
            conditions = ["1=1"]
            params: list = []

            if query.filters.get("source_type"):
                conditions.append("source_type = ?")
                params.append(query.filters["source_type"])
            if query.filters.get("goal_id"):
                conditions.append("goal_id = ?")
                params.append(query.filters["goal_id"])
            if query.filters.get("agent_name"):
                conditions.append("agent_name = ?")
                params.append(query.filters["agent_name"])

            where = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT * FROM memory_records WHERE {where} ORDER BY created_at DESC LIMIT 500",
                params,
            ).fetchall()

            hits: List[SearchHit] = []
            seen_hashes: set = set()

            now = time.time()
            for row in rows:
                record = self._row_to_record(row)

                if query_embedding:
                    emb = self._load_embedding(record.id, record.embedding_model or "unknown")
                    if emb:
                        score = _cosine(query_embedding, emb)
                        explanation = "cosine similarity"
                    else:
                        score = _keyword_score(query.query_text, record.content)
                        explanation = "keyword fallback (no embedding)"
                else:
                    score = _keyword_score(query.query_text, record.content)
                    explanation = "keyword search"

                # Recency bias
                if query.recency_bias > 0:
                    age_days = (now - record.created_at) / 86400.0
                    recency = 1.0 / (1.0 + age_days)
                    score = score * (1 - query.recency_bias) + recency * query.recency_bias

                if score < query.min_score:
                    continue

                if query.dedupe:
                    if record.content_hash and record.content_hash in seen_hashes:
                        continue
                    seen_hashes.add(record.content_hash)

                hits.append(SearchHit(record=record, score=score, explanation=explanation))

            hits.sort(key=lambda h: h.score, reverse=True)
            return hits
        except Exception:
            return []

    def filter_by(
        self,
        source_type: Optional[str] = None,
        goal_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        """Filter records by metadata fields."""
        try:
            conn = self._get_conn()
            conditions = ["1=1"]
            params: list = []

            if source_type:
                conditions.append("source_type = ?")
                params.append(source_type)
            if goal_id:
                conditions.append("goal_id = ?")
                params.append(goal_id)
            if agent_name:
                conditions.append("agent_name = ?")
                params.append(agent_name)

            where = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT * FROM memory_records WHERE {where} ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()

            records = [self._row_to_record(r) for r in rows]

            if tags:
                tag_set = set(tags)
                records = [r for r in records if tag_set & set(r.tags)]

            return records
        except Exception:
            return []

    def migrate_from_v1(self, v1_store) -> int:
        """Import all records from old VectorStore, mark source_type='legacy'. Returns count."""
        count = 0
        try:
            # Try common v1 interfaces
            records_iter = None
            if hasattr(v1_store, "_cache_records"):
                records_iter = v1_store._cache_records.values()
            elif hasattr(v1_store, "records"):
                records_iter = v1_store.records
            elif hasattr(v1_store, "all"):
                records_iter = v1_store.all()

            if records_iter is None:
                # Try to get from brain DB directly
                if hasattr(v1_store, "brain") and hasattr(v1_store.brain, "db"):
                    rows = v1_store.brain.db.execute("SELECT * FROM memory_records").fetchall()
                    for row in rows:
                        try:
                            new_rec = MemoryRecord(
                                id=str(uuid.uuid4()),
                                content=row[1] if len(row) > 1 else str(row),
                                source_type="legacy",
                                source_ref=row[3] if len(row) > 3 else "",
                                created_at=row[4] if len(row) > 4 else time.time(),
                                updated_at=time.time(),
                            )
                            self.upsert(new_rec)
                            count += 1
                        except Exception:
                            pass
                return count

            for v1_rec in records_iter:
                try:
                    content = getattr(v1_rec, "content", str(v1_rec))
                    new_rec = MemoryRecord(
                        id=str(uuid.uuid4()),
                        content=content,
                        source_type="legacy",
                        source_ref=getattr(v1_rec, "source_ref", ""),
                        created_at=getattr(v1_rec, "created_at", time.time()),
                        updated_at=time.time(),
                        goal_id=getattr(v1_rec, "goal_id", None),
                        agent_name=getattr(v1_rec, "agent_name", None),
                        tags=getattr(v1_rec, "tags", []),
                        importance=getattr(v1_rec, "importance", 1.0),
                    )
                    self.upsert(new_rec)
                    count += 1
                except Exception:
                    pass
        except Exception:
            pass
        return count

    def count(self) -> int:
        """Return total number of records."""
        try:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def delete(self, record_id: str) -> bool:
        """Delete a record by ID. Returns True if deleted."""
        try:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM memory_records WHERE id = ?", (record_id,))
            conn.commit()
            return cur.rowcount > 0
        except Exception:
            return False

    def close(self):
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
