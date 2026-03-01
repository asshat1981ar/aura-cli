import json
import time
import uuid
import hashlib
import numpy as np
import sqlite3
from typing import List, Dict, Any, Optional, Union
from core.logging_utils import log_json
from core.memory_types import MemoryRecord, RetrievalQuery, SearchHit

class VectorStore:
    """
    Unified Control Plane: Centralized semantic memory store.
    Implements ASCM v2 VectorStoreV2 protocol with multi-model embedding support.
    """
    def __init__(self, model_adapter, brain):
        self.model_adapter = model_adapter
        self.brain = brain
        # ASCM v2: use Row factory for name-based access in search results
        self.brain.db.row_factory = sqlite3.Row
        self._init_db()
        log_json("INFO", "vector_store_initialized", details={"backend": "sqlite_local_v2"})

    def _init_db(self):
        """Initialize the multi-table schema."""
        try:
            self.brain.db.execute("""
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    goal_id TEXT,
                    agent_name TEXT,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 1.0,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT,
                    embedding_dims INTEGER,
                    content_hash TEXT NOT NULL,
                    embedding BLOB
                )
            """)
            self.brain.db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    record_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dims INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    PRIMARY KEY (record_id, model_id),
                    FOREIGN KEY (record_id) REFERENCES memory_records(id) ON DELETE CASCADE
                )
            """)
            self.brain.db.execute("CREATE INDEX IF NOT EXISTS idx_mr_content_hash ON memory_records(content_hash)")
            self.brain.db.execute("CREATE INDEX IF NOT EXISTS idx_mr_source_type ON memory_records(source_type)")
            self.brain.db.commit()
            
            # Migration check for legacy v1 table
            self._migrate_legacy_v1()
        except sqlite3.Error as e:
            log_json("ERROR", "vector_store_init_failed", details={"error": str(e)})

    def _migrate_legacy_v1(self):
        """Migrates data from legacy tables to the v2 schema."""
        try:
            # Check if old table exists
            cursor = self.brain.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vector_store_data'")
            if not cursor.fetchone():
                return

            rows = self.brain.db.execute("SELECT content, embedding FROM vector_store_data").fetchall()
            migrated = 0
            for content, embedding_blob in rows:
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                
                # Check for existence
                existing = self.brain.db.execute("SELECT id FROM memory_records WHERE content_hash=?", (content_hash,)).fetchone()
                if not existing:
                    rid = uuid.uuid4().hex
                    now = time.time()
                    self.brain.db.execute("""
                        INSERT INTO memory_records (id, content, source_type, created_at, updated_at, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (rid, content, "legacy_v1", now, now, content_hash))
                    
                    vec = np.frombuffer(embedding_blob, dtype=np.float32)
                    self.brain.db.execute("""
                        INSERT INTO embeddings (record_id, model_id, dims, data)
                        VALUES (?, ?, ?, ?)
                    """, (rid, "unknown_v1", len(vec), embedding_blob))
                    migrated += 1
            
            if migrated > 0:
                self.brain.db.commit()
                log_json("INFO", "vector_store_legacy_migrated", details={"count": migrated})
        except Exception as e:
            log_json("WARN", "vector_store_migration_failed", details={"error": str(e)})

    def upsert(self, records: List[MemoryRecord]) -> Dict[str, int]:
        """Unified upsert: handles content and model-specific embeddings."""
        count = 0
        current_model = self.model_adapter.model_id()
        
        # 1. Identify records needing embeddings for the current model
        to_embed = []
        for rec in records:
            if not rec.content_hash:
                rec.content_hash = hashlib.sha256(rec.content.encode()).hexdigest()
            
            # Check if this record+model combo already has an embedding in DB
            existing = self.brain.db.execute(
                "SELECT record_id FROM embeddings WHERE record_id=? AND model_id=?",
                (rec.id, current_model)
            ).fetchone()
            
            if not existing and rec.embedding is None:
                to_embed.append(rec)

        # 2. Batch embed if needed
        if to_embed:
            try:
                vectors = self.model_adapter.embed([r.content for r in to_embed])
                for rec, vec in zip(to_embed, vectors):
                    rec.embedding = vec.tobytes()
                    rec.embedding_dims = len(vec)
                    rec.embedding_model = current_model
            except Exception as e:
                log_json("ERROR", "vector_store_batch_embed_failed", details={"error": str(e)})

        # 3. Insert/Update records and embeddings
        for rec in records:
            try:
                self.brain.db.execute("""
                    INSERT OR REPLACE INTO memory_records 
                    (id, content, source_type, source_ref, created_at, updated_at, 
                     goal_id, agent_name, tags, importance, token_count, 
                     embedding_model, embedding_dims, content_hash, embedding)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    rec.id, rec.content, rec.source_type, rec.source_ref,
                    rec.created_at, rec.updated_at, rec.goal_id, rec.agent_name,
                    json.dumps(rec.tags), rec.importance, rec.token_count,
                    rec.embedding_model, rec.embedding_dims, rec.content_hash, rec.embedding
                ))
                
                if rec.embedding:
                    self.brain.db.execute("""
                        INSERT OR REPLACE INTO embeddings (record_id, model_id, dims, data)
                        VALUES (?, ?, ?, ?)
                    """, (rec.id, current_model, rec.embedding_dims, rec.embedding))
                count += 1
            except sqlite3.Error as e:
                log_json("ERROR", "vector_store_upsert_failed", details={"id": rec.id, "error": str(e)})

        self.brain.db.commit()
        return {"upserted": count}

    def search(self, query: Union[str, RetrievalQuery], k: int = 5) -> Union[List[str], List[SearchHit]]:
        """ASCM v2 Search: semantic similarity with filters and metadata."""
        if isinstance(query, str):
            q_obj = RetrievalQuery(query_text=query, k=k)
        else:
            q_obj = query

        current_model = self.model_adapter.model_id()
        try:
            query_vec = self.model_adapter.get_embedding(q_obj.query_text)
        except Exception as e:
            log_json("WARN", "vector_store_search_embed_failed", details={"error": str(e)})
            return []

        # 1. Fetch candidates from DB (constrained by filters)
        # For efficiency, we only load vectors for the active model
        sql = """
            SELECT mr.*, e.data as embedding_blob 
            FROM memory_records mr
            JOIN embeddings e ON mr.id = e.record_id
            WHERE e.model_id = ?
        """
        params = [current_model]
        
        if q_obj.filters:
            for key, val in q_obj.filters.items():
                if key in ["source_type", "goal_id", "agent_name"]:
                    sql += f" AND mr.{key} = ?"
                    params.append(val)

        candidates = self.brain.db.execute(sql, params).fetchall()
        hits = []
        
        # 2. Rank candidates using cosine similarity in Python (sufficient for repo-scale)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0: return []

        for row in candidates:
            vec = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            v_norm = np.linalg.norm(vec)
            if v_norm == 0: continue
            
            score = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            
            # Recency bias (PRD: simple linear decay placeholder)
            if q_obj.recency_bias > 0:
                age = time.time() - row["created_at"]
                if age < 86400: # Boost items from last 24h
                    score += q_obj.recency_bias * 0.1

            if score >= q_obj.min_score:
                hits.append(SearchHit(
                    record_id=row["id"],
                    content=row["content"],
                    score=score,
                    source_ref=row["source_ref"],
                    metadata={"source_type": row["source_type"], "tags": json.loads(row["tags"])},
                    explanation=f"Similarity: {score:.3f}"
                ))

        hits.sort(key=lambda h: h.score, reverse=True)
        
        # Deduplication by content_hash
        if q_obj.dedupe_key == "content_hash":
            unique = []
            seen = set()
            for h in hits:
                # We need the hash from the candidate row
                cand = next(c for c in candidates if c["id"] == h.record_id)
                if cand["content_hash"] not in seen:
                    unique.append(h)
                    seen.add(cand["content_hash"])
            hits = unique

        results = hits[:q_obj.k]
        if isinstance(query, str):
            return [h.content for h in results]
        return results

    def stats(self) -> Dict[str, Any]:
        rec_count = self.brain.db.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
        emb_count = self.brain.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"records": rec_count, "record_count": rec_count, "embeddings": emb_count}

    def delete(self, ids: List[str]) -> int:
        cur = self.brain.db.execute("DELETE FROM memory_records WHERE id IN ({})".format(
            ",".join(["?"] * len(ids))
        ), ids)
        self.brain.db.commit()
        return cur.rowcount

    # Legacy method compatibility
    def add(self, content: str):
        rec = MemoryRecord(
            id=uuid.uuid4().hex,
            content=content,
            source_type="manual_add",
            source_ref="legacy_api",
            created_at=time.time(),
            updated_at=time.time()
        )
        self.upsert([rec])
