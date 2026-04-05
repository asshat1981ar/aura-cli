"""Persistent storage for the knowledge base."""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.knowledge.base import KnowledgeCategory, KnowledgeEntry
from core.logging_utils import log_json


class KnowledgeStore:
    """SQLite-based persistent storage for knowledge entries."""
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the knowledge store.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = Path(__file__).parent / "knowledge.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._local = threading.local()
        self._lock = threading.Lock()
        
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    def _init_db(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Knowledge entries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    entry_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    tags TEXT NOT NULL DEFAULT '[]',
                    related_entries TEXT NOT NULL DEFAULT '[]',
                    context TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed REAL,
                    embedding BLOB
                )
            """)
            
            # Indexes for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_category 
                ON knowledge_entries(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_created 
                ON knowledge_entries(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_access 
                ON knowledge_entries(access_count DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_source 
                ON knowledge_entries(source)
            """)
            
            # FTS for text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    entry_id,
                    content,
                    tags,
                    content_rowid=rowid
                )
            """)
            
            # Schema version tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # Insert or update schema version
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", 
                        (self.SCHEMA_VERSION,))
    
    async def save(self, entry: KnowledgeEntry) -> bool:
        """Save a knowledge entry."""
        try:
            with self._transaction() as conn:
                # Serialize complex fields
                tags_json = json.dumps(entry.tags)
                related_json = json.dumps(entry.related_entries)
                context_json = json.dumps(entry.context)
                embedding_blob = json.dumps(entry.embedding).encode() if entry.embedding else None
                
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge_entries (
                        entry_id, content, source, category, confidence,
                        tags, related_entries, context, created_at,
                        access_count, last_accessed, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.entry_id,
                    entry.content,
                    entry.source,
                    entry.category.value,
                    entry.confidence,
                    tags_json,
                    related_json,
                    context_json,
                    entry.created_at,
                    entry.access_count,
                    entry.last_accessed,
                    embedding_blob
                ))
                
                # Update FTS index
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge_fts (entry_id, content, tags)
                    VALUES (?, ?, ?)
                """, (entry.entry_id, entry.content, tags_json))
                
            return True
            
        except Exception as e:
            log_json("ERROR", "knowledge_save_failed", {
                "entry_id": entry.entry_id,
                "error": str(e)
            })
            return False
    
    async def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        try:
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT * FROM knowledge_entries WHERE entry_id = ?",
                    (entry_id,)
                ).fetchone()
                
                if row:
                    return self._row_to_entry(row)
                return None
                
        except Exception as e:
            log_json("ERROR", "knowledge_get_failed", {
                "entry_id": entry_id,
                "error": str(e)
            })
            return None
    
    async def update(self, entry: KnowledgeEntry) -> bool:
        """Update an existing knowledge entry."""
        return await self.save(entry)
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        try:
            with self._transaction() as conn:
                conn.execute(
                    "DELETE FROM knowledge_entries WHERE entry_id = ?",
                    (entry_id,)
                )
                conn.execute(
                    "DELETE FROM knowledge_fts WHERE entry_id = ?",
                    (entry_id,)
                )
                
                return conn.total_changes > 0
                
        except Exception as e:
            log_json("ERROR", "knowledge_delete_failed", {
                "entry_id": entry_id,
                "error": str(e)
            })
            return False
    
    async def search(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        categories: Optional[List[KnowledgeCategory]] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        min_recency_days: Optional[int] = None,
        limit: int = 100
    ) -> List[KnowledgeEntry]:
        """Search for knowledge entries."""
        try:
            with self._transaction() as conn:
                entries = []
                
                # Build query dynamically
                conditions = ["confidence >= ?"]
                params = [min_confidence]
                
                if categories:
                    cat_values = [c.value for c in categories]
                    placeholders = ','.join('?' * len(cat_values))
                    conditions.append(f"category IN ({placeholders})")
                    params.extend(cat_values)
                
                if min_recency_days:
                    cutoff_time = time.time() - (min_recency_days * 24 * 3600)
                    conditions.append("created_at >= ?")
                    params.append(cutoff_time)
                
                # FTS text search
                if query_text and len(query_text) > 2:
                    fts_results = conn.execute("""
                        SELECT entry_id FROM knowledge_fts 
                        WHERE knowledge_fts MATCH ?
                        LIMIT ?
                    """, (query_text, limit * 2)).fetchall()
                    
                    if fts_results:
                        fts_ids = [r['entry_id'] for r in fts_results]
                        placeholders = ','.join('?' * len(fts_ids))
                        conditions.append(f"entry_id IN ({placeholders})")
                        params.extend(fts_ids)
                
                # Tag filtering (simplified - exact match)
                if tags:
                    # Get all entries and filter in Python for tag matching
                    pass
                
                where_clause = " AND ".join(conditions)
                
                query = f"""
                    SELECT * FROM knowledge_entries 
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                rows = conn.execute(query, params).fetchall()
                
                for row in rows:
                    entry = self._row_to_entry(row)
                    
                    # Tag filtering in Python
                    if tags:
                        if not any(t in entry.tags for t in tags):
                            continue
                    
                    entries.append(entry)
                
                return entries[:limit]
                
        except Exception as e:
            log_json("ERROR", "knowledge_search_failed", {
                "query": query_text[:50] if query_text else None,
                "error": str(e)
            })
            return []
    
    async def get_recent(
        self,
        category: Optional[KnowledgeCategory] = None,
        limit: int = 10,
        days: Optional[int] = None
    ) -> List[KnowledgeEntry]:
        """Get recently created entries."""
        min_time = None
        if days:
            min_time = time.time() - (days * 24 * 3600)
        
        try:
            with self._transaction() as conn:
                if category:
                    rows = conn.execute("""
                        SELECT * FROM knowledge_entries 
                        WHERE category = ?
                        AND (? IS NULL OR created_at >= ?)
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (category.value, min_time, min_time, limit)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM knowledge_entries 
                        WHERE ? IS NULL OR created_at >= ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (min_time, min_time, limit)).fetchall()
                
                return [self._row_to_entry(row) for row in rows]
                
        except Exception as e:
            log_json("ERROR", "knowledge_recent_failed", {"error": str(e)})
            return []
    
    async def get_all(self) -> List[KnowledgeEntry]:
        """Get all knowledge entries."""
        try:
            with self._transaction() as conn:
                rows = conn.execute(
                    "SELECT * FROM knowledge_entries ORDER BY created_at DESC"
                ).fetchall()
                
                return [self._row_to_entry(row) for row in rows]
                
        except Exception as e:
            log_json("ERROR", "knowledge_get_all_failed", {"error": str(e)})
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        try:
            with self._transaction() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM knowledge_entries"
                ).fetchone()[0]
                
                by_category = {}
                rows = conn.execute(
                    "SELECT category, COUNT(*) FROM knowledge_entries GROUP BY category"
                ).fetchall()
                for row in rows:
                    by_category[row[0]] = row[1]
                
                avg_confidence = conn.execute(
                    "SELECT AVG(confidence) FROM knowledge_entries"
                ).fetchone()[0] or 0.0
                
                total_accesses = conn.execute(
                    "SELECT SUM(access_count) FROM knowledge_entries"
                ).fetchone()[0] or 0
                
                return {
                    "total_entries": total,
                    "by_category": by_category,
                    "avg_confidence": avg_confidence,
                    "total_accesses": total_accesses
                }
                
        except Exception as e:
            log_json("ERROR", "knowledge_stats_failed", {"error": str(e)})
            return {"total_entries": 0, "error": str(e)}
    
    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        """Convert database row to KnowledgeEntry."""
        # Parse JSON fields
        tags = json.loads(row['tags']) if row['tags'] else []
        related = json.loads(row['related_entries']) if row['related_entries'] else []
        context = json.loads(row['context']) if row['context'] else {}
        
        embedding = None
        if row['embedding']:
            try:
                embedding = json.loads(row['embedding'].decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        return KnowledgeEntry(
            entry_id=row['entry_id'],
            content=row['content'],
            source=row['source'],
            category=KnowledgeCategory(row['category']),
            confidence=row['confidence'],
            tags=tags,
            related_entries=related,
            context=context,
            created_at=row['created_at'],
            access_count=row['access_count'],
            last_accessed=row['last_accessed'],
            embedding=embedding
        )
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
