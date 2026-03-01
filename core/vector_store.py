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
    Manages semantic embeddings for content, allowing for similarity search.
    Implements ASCM v2 VectorStoreV2 protocol while maintaining v1 compatibility.
    """
    def __init__(self, model_adapter, brain):
        """
        Initializes the VectorStore.

        Args:
            model_adapter: An instance of ModelAdapter for generating embeddings.
            brain: An instance of Brain for database access.
        """
        self.model_adapter = model_adapter
        self.brain = brain
        
        # In-memory cache for vector search (SQLite is storage, memory is index)
        self._cache_records: Dict[str, MemoryRecord] = {}
        self._cache_vectors: Dict[str, np.ndarray] = {}
        
        self._init_db()
        self._load_from_db()
        
        log_json("INFO", "vector_store_initialized", details={"backend": "sqlite_local_v2"})

    def _init_db(self):
        """Initialize the database schema."""
        try:
            # V2 Schema
            self.brain.db.execute("""
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    source_type TEXT,
                    source_ref TEXT,
                    created_at REAL,
                    updated_at REAL,
                    goal_id TEXT,
                    agent_name TEXT,
                    tags TEXT,
                    importance REAL,
                    token_count INTEGER,
                    embedding_model TEXT,
                    embedding_dims INTEGER,
                    content_hash TEXT,
                    embedding BLOB
                )
            """)
            self.brain.db.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON memory_records(content_hash)")
            self.brain.db.commit()
        except sqlite3.Error as e:
            log_json("ERROR", "vector_store_init_failed", details={"error": str(e)})

    def _load_from_db(self):
        """Load records from DB into memory cache."""
        try:
            # 1. Load V2 records
            rows = self.brain.db.execute("SELECT * FROM memory_records").fetchall()
            for row in rows:
                # row is a sqlite3.Row or tuple. Assuming tuple access by index or name if Row factory set.
                # If Row factory is sqlite3.Row, we can use keys.
                # Brain uses row_factory = sqlite3.Row usually, let's assume index to be safe or check keys.
                # Schema order matches CREATE TABLE
                rec = MemoryRecord(
                    id=row[0],
                    content=row[1],
                    source_type=row[2],
                    source_ref=row[3],
                    created_at=row[4],
                    updated_at=row[5],
                    goal_id=row[6],
                    agent_name=row[7],
                    tags=json.loads(row[8]) if row[8] else [],
                    importance=row[9],
                    token_count=row[10],
                    embedding_model=row[11],
                    embedding_dims=row[12],
                    content_hash=row[13],
                    embedding=row[14]
                )
                self._cache_records[rec.id] = rec
                if rec.embedding:
                    self._cache_vectors[rec.id] = np.frombuffer(rec.embedding, dtype=np.float32)

            # 2. Check for V1 legacy data and migrate if needed
            self._migrate_v1_data()
            
            log_json("INFO", "vector_store_loaded", details={"count": len(self._cache_records)})
            
        except Exception as e:
            log_json("WARN", "vector_store_load_error", details={"error": str(e)})

    def _migrate_v1_data(self):
        """Migrate data from legacy vector_store_data table to memory_records."""
        try:
            # Check if legacy table exists
            cursor = self.brain.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vector_store_data'")
            if not cursor.fetchone():
                return

            rows = self.brain.db.execute("SELECT content, embedding FROM vector_store_data").fetchall()
            migrated = 0
            for content, embedding_blob in rows:
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                
                # Check if already exists in V2 via content_hash
                exists = False
                for r in self._cache_records.values():
                    if r.content_hash == content_hash:
                        exists = True
                        break
                
                if not exists:
                    # Create V2 record
                    rec = MemoryRecord(
                        id=uuid.uuid4().hex,
                        content=content,
                        source_type="legacy_migration",
                        source_ref="v1_store",
                        created_at=time.time(),
                        updated_at=time.time(),
                        embedding_model="unknown_v1", # Likely text-embedding-3-small but not guaranteed
                        embedding_dims=len(np.frombuffer(embedding_blob, dtype=np.float32)),
                        content_hash=content_hash,
                        embedding=embedding_blob
                    )
                    self._upsert_single(rec)
                    migrated += 1
            
            if migrated > 0:
                log_json("INFO", "vector_store_v1_migrated", details={"migrated_count": migrated})
                # Optionally drop legacy table, but keeping for safety for now
                
        except Exception as e:
            log_json("WARN", "vector_store_migration_failed", details={"error": str(e)})

    def _upsert_single(self, record: MemoryRecord):
        """Internal helper to upsert a single record to DB and Cache."""
        # Update Cache
        self._cache_records[record.id] = record
        if record.embedding:
            self._cache_vectors[record.id] = np.frombuffer(record.embedding, dtype=np.float32)
        
        # Update DB
        self.brain.db.execute("""
            INSERT OR REPLACE INTO memory_records VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            record.id,
            record.content,
            record.source_type,
            record.source_ref,
            record.created_at,
            record.updated_at,
            record.goal_id,
            record.agent_name,
            json.dumps(record.tags),
            record.importance,
            record.token_count,
            record.embedding_model,
            record.embedding_dims,
            record.content_hash,
            record.embedding
        ))
        self.brain.db.commit()

    def upsert(self, records: List[MemoryRecord]) -> Dict[str, int]:
        """Insert or update memory records using batched embeddings."""
        # 1. Identify records needing embeddings
        missing = [r for r in records if r.embedding is None]
        
        if missing:
            try:
                # 2. Batch embed
                contents = [r.content for r in missing]
                vectors = self.model_adapter.embed(contents)
                
                # 3. Associate vectors back to records
                for rec, vec in zip(missing, vectors):
                    rec.embedding = vec.tobytes()
                    rec.embedding_dims = len(vec)
                    rec.embedding_model = self.model_adapter.model_id()
            except Exception as e:
                log_json("ERROR", "batched_embedding_failed", details={"error": str(e)})
                # Fallback: try individual if batch failed? No, better to let adapter handle retries.
        
        # 4. Upsert all to DB/Cache
        count = 0
        for rec in records:
            if rec.embedding is not None:
                self._upsert_single(rec)
                count += 1
            
        return {"upserted": count}

    def delete(self, ids: List[str]) -> int:
        """Delete records by ID."""
        count = 0
        for rid in ids:
            if rid in self._cache_records:
                del self._cache_records[rid]
                if rid in self._cache_vectors:
                    del self._cache_vectors[rid]
                
                self.brain.db.execute("DELETE FROM memory_records WHERE id=?", (rid,))
                count += 1
        self.brain.db.commit()
        return count

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        return {
            "record_count": len(self._cache_records),
            "vector_count": len(self._cache_vectors)
        }

    def search(self, query: Union[str, RetrievalQuery], k: int = 5) -> Union[List[str], List[SearchHit]]:
        """
        Search the store.
        Args:
            query: Query string (legacy) or RetrievalQuery object (v2).
            k: Number of results (legacy).
        Returns:
            List of content strings (legacy) or List of SearchHit (v2).
        """
        # 1. Handle Legacy Call
        if isinstance(query, str):
            q_obj = RetrievalQuery(query_text=query, k=k)
            hits = self._execute_search(q_obj)
            return [h.content for h in hits]
        
        # 2. Handle V2 Call
        if isinstance(query, RetrievalQuery):
            return self._execute_search(query)
            
        raise ValueError("Invalid query type")

    def _execute_search(self, query: RetrievalQuery) -> List[SearchHit]:
        """Internal search execution logic."""
        if not self._cache_vectors:
            return []

        try:
            query_vec = self.model_adapter.get_embedding(query.query_text)
        except Exception as e:
            log_json("WARN", "search_embedding_failed", details={"error": str(e)})
            return []

        hits = []
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
            
        for rid, vec in self._cache_vectors.items():
            rec = self._cache_records.get(rid)
            if not rec:
                continue
                
            # Apply Filters
            if query.filters:
                match = True
                for key, val in query.filters.items():
                    # Simple attribute check
                    if getattr(rec, key, None) != val:
                        match = False
                        break
                if not match:
                    continue

            # Calculate Cosine Similarity
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
            
            # Recency Bias
            # Simple implementation: boost score slightly for newer items
            if query.recency_bias > 0:
                age = time.time() - rec.created_at
                # decay factor? For now, just a linear boost for recent items (e.g., last 24h)
                # This is a placeholder for more complex decay
                if age < 86400:
                    score += query.recency_bias * 0.1

            if score >= query.min_score:
                hits.append(SearchHit(
                    record_id=rec.id,
                    content=rec.content,
                    score=score,
                    source_ref=rec.source_ref,
                    metadata={"source_type": rec.source_type, "tags": rec.tags},
                    explanation=f"Similarity: {score:.3f}"
                ))

        # Sort by score
        hits.sort(key=lambda h: h.score, reverse=True)
        
        # Deduplication (if dedupe_key provided)
        # Assuming dedupe_key is a field in MemoryRecord, need to look up record from hit
        # But SearchHit doesn't have the full record content hash readily available in its struct unless in metadata
        # Let's trust the raw hits for now or implement if needed. 
        # The PRD mentions dedupe_key="content_hash".
        if query.dedupe_key == "content_hash":
            unique_hits = []
            seen_hashes = set()
            for hit in hits:
                rec = self._cache_records[hit.record_id]
                if rec.content_hash not in seen_hashes:
                    unique_hits.append(hit)
                    seen_hashes.add(rec.content_hash)
            hits = unique_hits

        return hits[:query.k]

    # Legacy method compatibility
    def add(self, content: str):
        """Legacy add method."""
        rec = MemoryRecord(
            id=uuid.uuid4().hex,
            content=content,
            source_type="legacy_add",
            source_ref="unknown",
            created_at=time.time(),
            updated_at=time.time(),
            content_hash=hashlib.sha256(content.encode()).hexdigest()
        )
        self.upsert([rec])
