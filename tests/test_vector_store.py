"""Tests for core/vector_store.py."""

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from core.vector_store import VectorStore, _MissingPackage, SEARCH_LIMIT
from core.memory_types import MemoryRecord, RetrievalQuery


def make_memory_record(record_id, content, source_type="test"):
    """Helper to create MemoryRecord with required fields."""
    return MemoryRecord(
        id=record_id,
        content=content,
        source_type=source_type,
        source_ref="test",
        created_at=time.time(),
        updated_at=time.time(),
    )


class TestMissingPackage:
    """Test _MissingPackage placeholder class."""
    
    def test_getattr_raises(self):
        """Test attribute access raises ImportError."""
        missing = _MissingPackage("test_package")
        with pytest.raises(ImportError) as exc_info:
            missing.some_attribute
        assert "test_package" in str(exc_info.value)
    
    def test_call_raises(self):
        """Test calling raises ImportError."""
        missing = _MissingPackage("test_package")
        with pytest.raises(ImportError) as exc_info:
            missing()
        assert "test_package" in str(exc_info.value)


class TestVectorStoreInit:
    """Test VectorStore initialization."""
    
    def test_init_creates_tables(self):
        """Test initialization creates database tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            # Check tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            
            assert "memory_records" in tables
            assert "embeddings" in tables
            
            conn.close()


class TestVectorStoreUpsert:
    """Test VectorStore.upsert method."""
    
    def test_upsert_single_record(self):
        """Test upserting a single record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.embed.return_value = [0.1, 0.2, 0.3]
            
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            record = make_memory_record("test-1", "test content")
            
            result = store.upsert([record])
            
            assert result["upserted"] == 1
            
            conn.close()
    
    def test_upsert_multiple_records(self):
        """Test upserting multiple records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.embed.return_value = [0.1, 0.2, 0.3]
            
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            records = [make_memory_record(f"test-{i}", f"content {i}") for i in range(5)]
            
            result = store.upsert(records)
            
            assert result["upserted"] == 5
            
            conn.close()
    
    def test_upsert_empty_list(self):
        """Test upserting empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            result = store.upsert([])
            
            assert result["upserted"] == 0
            
            conn.close()


class TestVectorStoreSearch:
    """Test VectorStore.search method."""
    
    def test_search_returns_list(self):
        """Test search returns list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.get_embedding.return_value = [0.1, 0.2, 0.3]
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            results = store.search("search query", k=5)
            
            assert isinstance(results, list)
            
            conn.close()


class TestVectorStoreStats:
    """Test VectorStore.stats method."""
    
    def test_stats_empty_store(self):
        """Test stats on empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            stats = store.stats()
            
            assert "record_count" in stats
            assert "embeddings" in stats
            assert stats["record_count"] == 0
            
            conn.close()
    
    def test_stats_with_records(self):
        """Test stats with records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.embed.return_value = [0.1, 0.2, 0.3]
            
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            # Add records
            records = [make_memory_record(f"test-{i}", f"content {i}") for i in range(3)]
            store.upsert(records)
            
            stats = store.stats()
            
            assert stats["record_count"] == 3
            assert "embeddings" in stats
            
            conn.close()


class TestVectorStoreDelete:
    """Test VectorStore.delete method."""
    
    def test_delete_existing_records(self):
        """Test deleting existing records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.embed.return_value = [0.1, 0.2, 0.3]
            
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            # Add and then delete
            record = make_memory_record("test-1", "content")
            store.upsert([record])
            
            deleted_count = store.delete(["test-1"])
            
            assert deleted_count == 1
            
            conn.close()
    
    def test_delete_nonexistent_records(self):
        """Test deleting non-existent records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            deleted_count = store.delete(["nonexistent"])
            
            assert deleted_count == 0
            
            conn.close()


class TestVectorStoreAdd:
    """Test VectorStore.add convenience method."""
    
    def test_add_single_content(self):
        """Test adding single content string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.embed.return_value = [0.1, 0.2, 0.3]
            
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            store.add("test content")
            # verify record was persisted
            count = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
            assert count == 1
            
            conn.close()


class TestVectorStoreRebuild:
    """Test VectorStore.rebuild method."""
    
    def test_rebuild_empty_store(self):
        """Test rebuilding empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            result = store.rebuild()
            
            assert "records_seen" in result
            assert "records_failed" in result
            
            conn.close()
    
    def test_rebuild_with_records(self):
        """Test rebuilding store with records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.embed.return_value = [0.1, 0.2, 0.3]
            
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            # Add records
            records = [make_memory_record(f"test-{i}", f"content {i}") for i in range(3)]
            store.upsert(records)
            
            result = store.rebuild()
            
            assert result["records_seen"] >= 0
            
            conn.close()


class TestVectorStoreMigrateEmbeddingModel:
    """Test VectorStore.migrate_embedding_model method."""
    
    def test_migrate_model(self):
        """Test migrating to new embedding model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            
            mock_model = Mock()
            mock_model.model_id.return_value = "test-model"
            mock_model.embed.return_value = [0.1, 0.2, 0.3]
            
            mock_brain = Mock()
            mock_brain.db = conn
            
            store = VectorStore(mock_model, mock_brain)
            
            result = store.migrate_embedding_model("new-model-v1")
            
            assert "embeddings_written" in result
            assert "records_failed" in result
            
            conn.close()


class TestSearchLimit:
    """Test SEARCH_LIMIT constant."""
    
    def test_search_limit_value(self):
        """Test SEARCH_LIMIT is reasonable value."""
        assert SEARCH_LIMIT == 1000
        assert isinstance(SEARCH_LIMIT, int)
        assert SEARCH_LIMIT > 0
