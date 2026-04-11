from __future__ import annotations

import json
import time
import uuid
import hashlib
import sqlite3
import warnings
from typing import List, Dict, Any, Union


class _MissingPackage:
    """Placeholder for optional dependencies that are not installed."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> None:
        # Raise AttributeError so that getattr(obj, name, default) falls back
        # to the default correctly.  Python's getattr() only suppresses
        # AttributeError, not ImportError, making ImportError here break
        # unittest.mock.patch on Python 3.10.
        raise AttributeError(f"Optional dependency '{self._name}' is required for this operation.")

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise ImportError(f"Optional dependency '{self._name}' is required for this operation.")


try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - exercised via optional-deps tests
    np = _MissingPackage("numpy")  # type: ignore
from core.logging_utils import log_json
from core.memory_types import MemoryRecord, RetrievalQuery, SearchHit

# Maximum number of candidate rows fetched from the DB during a search query.
# Prevents unbounded memory use when the embeddings table grows large.
SEARCH_LIMIT = 1000


class VectorStore:
    """
    Unified Control Plane: Centralized semantic memory store.
    Implements ASCM v2 VectorStoreV2 protocol with multi-model embedding support.
    """

    def __init__(self, model_adapter, brain):
        if type(self) is VectorStore:
            warnings.warn(
                "core.vector_store (v1) is deprecated. Use memory.vector_store_v2 instead.",
                DeprecationWarning,
                stacklevel=2,
            )
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

            # Namespace column migration (ASCM v2)
            try:
                self.brain.db.execute("ALTER TABLE memory_records ADD COLUMN namespace TEXT")
                self.brain.db.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

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
                    self.brain.db.execute(
                        """
                        INSERT INTO memory_records (id, content, source_type, created_at, updated_at, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (rid, content, "legacy_v1", now, now, content_hash),
                    )

                    vec = np.frombuffer(embedding_blob, dtype=np.float32)
                    self.brain.db.execute(
                        """
                        INSERT INTO embeddings (record_id, model_id, dims, data)
                        VALUES (?, ?, ?, ?)
                    """,
                        (rid, "unknown_v1", len(vec), embedding_blob),
                    )
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
            existing = self.brain.db.execute("SELECT record_id FROM embeddings WHERE record_id=? AND model_id=?", (rec.id, current_model)).fetchone()

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
        try:
            # Use a manual transaction for the batch to ensure consistency
            self.brain.db.execute("BEGIN TRANSACTION")
            for rec in records:
                try:
                    self.brain.db.execute(
                        """
                        INSERT OR REPLACE INTO memory_records
                        (id, content, source_type, source_ref, created_at, updated_at,
                         goal_id, agent_name, tags, importance, token_count,
                         embedding_model, embedding_dims, content_hash, embedding, namespace)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                        (
                            rec.id,
                            rec.content,
                            rec.source_type,
                            rec.source_ref,
                            rec.created_at,
                            rec.updated_at,
                            rec.goal_id,
                            rec.agent_name,
                            json.dumps(rec.tags),
                            rec.importance,
                            rec.token_count,
                            rec.embedding_model,
                            rec.embedding_dims,
                            rec.content_hash,
                            rec.embedding,
                            getattr(rec, "namespace", None),
                        ),
                    )

                    if rec.embedding:
                        self.brain.db.execute(
                            """
                            INSERT OR REPLACE INTO embeddings (record_id, model_id, dims, data)
                            VALUES (?, ?, ?, ?)
                        """,
                            (rec.id, current_model, rec.embedding_dims, rec.embedding),
                        )
                    count += 1
                except sqlite3.Error as e:
                    log_json("ERROR", "vector_store_record_upsert_failed", details={"id": rec.id, "error": str(e)})
                    # Individual record failure doesn't necessarily abort the whole batch,
                    # but we should be careful. For now, we continue but don't commit this specific record.

            self.brain.db.commit()
        except sqlite3.Error as e:
            self.brain.db.rollback()
            log_json("ERROR", "vector_store_batch_upsert_failed", details={"error": str(e)})
            return {"upserted": 0, "error": str(e)}

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
        # For efficiency, we only load vectors for the active model.
        # SEARCH_LIMIT caps the result set to prevent unbounded memory use.
        sql = """
            SELECT mr.*, e.data as embedding_blob
            FROM memory_records mr
            JOIN embeddings e ON mr.id = e.record_id
            WHERE e.model_id = ?
        """
        params: list = [current_model]

        if q_obj.filters:
            for key, val in q_obj.filters.items():
                if key in ["source_type", "goal_id", "agent_name"]:
                    sql += f" AND mr.{key} = ?"
                    params.append(val)

        if q_obj.namespace is not None:
            sql += " AND mr.namespace = ?"
            params.append(q_obj.namespace)

        sql += " LIMIT ?"
        params.append(SEARCH_LIMIT)

        candidates = self.brain.db.execute(sql, params).fetchall()
        hits = []

        # 2. Rank candidates using cosine similarity in Python (sufficient for repo-scale)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []

        for row in candidates:
            vec = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            v_norm = np.linalg.norm(vec)
            if v_norm == 0:
                continue

            score = float(np.dot(query_vec, vec) / (q_norm * v_norm))

            # Recency bias (PRD: simple linear decay placeholder)
            if q_obj.recency_bias > 0:
                age = time.time() - row["created_at"]
                if age < 86400:  # Boost items from last 24h
                    score += q_obj.recency_bias * 0.1

            if score >= q_obj.min_score:
                hits.append(
                    SearchHit(
                        record_id=row["id"],
                        content=row["content"],
                        score=score,
                        source_ref=row["source_ref"] or "",
                        metadata={"source_type": row["source_type"], "tags": json.loads(row["tags"]), "importance": row["importance"]},
                        explanation=f"Similarity: {score:.3f}",
                        embedding_model_version=current_model,
                    )
                )

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

        results = hits[: q_obj.k]
        if isinstance(query, str):
            return [h.content for h in results]
        return results

    def stats(self) -> Dict[str, Any]:
        rec_count = self.brain.db.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
        emb_count = self.brain.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"records": rec_count, "record_count": rec_count, "embeddings": emb_count}

    def rebuild(self, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Rebuild embeddings for the active model.

        Options:
          - model_id: explicit target model id
          - source_types: optional list of source_type values to include
          - exclude_source_types: optional list of source_type values to exclude
          - drop_existing_embeddings: delete existing rows for the target model first
          - batch_size: embedding batch size
        """
        opts = dict(options or {})
        target_model = str(opts.get("model_id") or self.model_adapter.model_id())
        source_types = opts.get("source_types") or []
        exclude_source_types = opts.get("exclude_source_types") or []
        drop_existing_embeddings = bool(opts.get("drop_existing_embeddings", True))
        batch_size = max(1, int(opts.get("batch_size", 16)))

        clauses = []
        params: list[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        if exclude_source_types:
            placeholders = ",".join("?" for _ in exclude_source_types)
            clauses.append(f"source_type NOT IN ({placeholders})")
            params.extend(exclude_source_types)
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        stats = {
            "model_id": target_model,
            "records_seen": 0,
            "embeddings_written": 0,
            "records_failed": 0,
            "drop_existing_embeddings": drop_existing_embeddings,
        }

        try:
            rows = self.brain.db.execute(
                f"SELECT id, content FROM memory_records{where_sql} ORDER BY id",
                params,
            ).fetchall()
            stats["records_seen"] = len(rows)
            record_ids = [row["id"] if isinstance(row, sqlite3.Row) else row[0] for row in rows]

            if drop_existing_embeddings and record_ids:
                placeholders = ",".join("?" for _ in record_ids)
                self.brain.db.execute(
                    f"DELETE FROM embeddings WHERE model_id = ? AND record_id IN ({placeholders})",
                    [target_model, *record_ids],
                )
                self.brain.db.execute(
                    f"""
                    UPDATE memory_records
                    SET embedding_model = NULL, embedding_dims = NULL, embedding = NULL
                    WHERE id IN ({placeholders}) AND embedding_model = ?
                    """,
                    [*record_ids, target_model],
                )
                self.brain.db.commit()

            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                texts = [row["content"] if isinstance(row, sqlite3.Row) else row[1] for row in batch]
                vectors = self.model_adapter.embed(texts)

                self.brain.db.execute("BEGIN TRANSACTION")
                try:
                    for row, vec in zip(batch, vectors):
                        record_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
                        embedding_blob = np.array(vec, dtype=np.float32).tobytes()
                        dims = int(len(vec))
                        self.brain.db.execute(
                            """
                            INSERT OR REPLACE INTO embeddings (record_id, model_id, dims, data)
                            VALUES (?, ?, ?, ?)
                            """,
                            (record_id, target_model, dims, embedding_blob),
                        )
                        self.brain.db.execute(
                            """
                            UPDATE memory_records
                            SET embedding_model = ?, embedding_dims = ?, embedding = ?
                            WHERE id = ?
                            """,
                            (target_model, dims, embedding_blob, record_id),
                        )
                        stats["embeddings_written"] += 1
                    self.brain.db.commit()
                except (OSError, IOError, ValueError):
                    self.brain.db.rollback()
                    raise

            log_json("INFO", "vector_store_rebuild_complete", details=stats)
            return stats
        except Exception as exc:
            stats["records_failed"] = stats["records_seen"] - stats["embeddings_written"]
            stats["error"] = str(exc)
            log_json("ERROR", "vector_store_rebuild_failed", details=stats)
            return stats

    def migrate_embedding_model(self, new_model_id: str) -> Dict[str, Any]:
        """
        Rebuild all current records under a new embedding model identifier.
        The caller is responsible for configuring the adapter to actually emit
        vectors for that model before invoking this method.
        """
        return self.rebuild({"model_id": new_model_id, "drop_existing_embeddings": True})

    def delete(self, ids: List[str]) -> int:
        cur = self.brain.db.execute("DELETE FROM memory_records WHERE id IN ({})".format(",".join(["?"] * len(ids))), ids)
        self.brain.db.commit()
        return cur.rowcount

    # Legacy method compatibility
    def add(self, content: str):
        rec = MemoryRecord(id=uuid.uuid4().hex, content=content, source_type="manual_add", source_ref="legacy_api", created_at=time.time(), updated_at=time.time())
        self.upsert([rec])


# ---------------------------------------------------------------------------
# Backwards compatibility shim
# NOTE: memory.vector_store_v2.VectorStoreV2 *inherits* from VectorStore, so
# we cannot reassign the name here without creating a circular dependency.
# New code should import VectorStoreV2 directly from memory.vector_store_v2.
# ---------------------------------------------------------------------------
try:
    from memory.vector_store_v2 import VectorStoreV2  # noqa: F401 – re-exported for convenience
except ImportError:
    pass  # Fall back to v1 implementation above
