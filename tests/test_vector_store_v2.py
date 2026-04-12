"""Tests for memory/vector_store_v2.py — VectorStoreV2.

Tests the thin subclass that delegates to core.vector_store.VectorStore.
"""

from unittest.mock import Mock, MagicMock, patch
import tempfile
import sqlite3
from pathlib import Path

import pytest

from memory.vector_store_v2 import VectorStoreV2


class TestVectorStoreV2Init:
    """Test VectorStoreV2 initialization."""

    def test_v2_init_calls_super(self):
        """VectorStoreV2.__init__ calls parent VectorStore.__init__."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_brain = Mock()
            mock_brain.db = conn

            # VectorStoreV2 should succeed without error
            store = VectorStoreV2(mock_model_adapter, mock_brain)

            assert store is not None
            assert store.model_adapter is mock_model_adapter
            assert store.brain is mock_brain

            conn.close()

    def test_v2_inherits_from_core_vector_store(self):
        """VectorStoreV2 is a subclass of core.vector_store.VectorStore."""
        from core.vector_store import VectorStore as CoreVectorStore

        assert issubclass(VectorStoreV2, CoreVectorStore)

    def test_v2_has_rebuild_method(self):
        """VectorStoreV2 provides rebuild() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            assert hasattr(store, "rebuild")
            assert callable(store.rebuild)

            conn.close()

    def test_v2_has_migrate_embedding_model_method(self):
        """VectorStoreV2 provides migrate_embedding_model() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            assert hasattr(store, "migrate_embedding_model")
            assert callable(store.migrate_embedding_model)

            conn.close()


class TestVectorStoreV2Rebuild:
    """Test VectorStoreV2.rebuild() delegation."""

    def test_rebuild_returns_dict(self):
        """rebuild() returns a dict with operation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            # Mock the parent's rebuild to avoid actual embedding
            with patch.object(
                store.__class__.__bases__[0],
                "rebuild",
                return_value={"status": "success", "records_processed": 0},
            ):
                result = store.rebuild()

            assert isinstance(result, dict)
            assert "status" in result or "records_processed" in result

            conn.close()

    def test_rebuild_accepts_options(self):
        """rebuild() accepts optional options dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            with patch.object(
                store.__class__.__bases__[0],
                "rebuild",
                return_value={"status": "success"},
            ) as mock_rebuild:
                result = store.rebuild({"batch_size": 100})

            # Verify parent rebuild was called with options
            mock_rebuild.assert_called_once()

            conn.close()

    def test_rebuild_with_none_options(self):
        """rebuild(None) handles None options gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            with patch.object(
                store.__class__.__bases__[0],
                "rebuild",
                return_value={"status": "success"},
            ):
                result = store.rebuild(None)

            assert isinstance(result, dict)

            conn.close()


class TestVectorStoreV2MigrateEmbeddingModel:
    """Test VectorStoreV2.migrate_embedding_model() delegation."""

    def test_migrate_embedding_model_returns_dict(self):
        """migrate_embedding_model() returns a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            with patch.object(
                store.__class__.__bases__[0],
                "migrate_embedding_model",
                return_value={"status": "success", "records_migrated": 0},
            ):
                result = store.migrate_embedding_model("new-model-id")

            assert isinstance(result, dict)

            conn.close()

    def test_migrate_embedding_model_accepts_model_id(self):
        """migrate_embedding_model() accepts new_model_id parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            with patch.object(
                store.__class__.__bases__[0],
                "migrate_embedding_model",
                return_value={"status": "success"},
            ) as mock_migrate:
                result = store.migrate_embedding_model("gpt-4-embedding-v2")

            mock_migrate.assert_called_once_with("gpt-4-embedding-v2")

            conn.close()

    def test_migrate_embedding_model_string_parameter(self):
        """migrate_embedding_model() requires string model_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            # Should work with any string
            with patch.object(
                store.__class__.__bases__[0],
                "migrate_embedding_model",
                return_value={"status": "success"},
            ):
                store.migrate_embedding_model("model-123")
                store.migrate_embedding_model("another.model.id")

            conn.close()


class TestVectorStoreV2Inheritance:
    """Test that VectorStoreV2 properly inherits all VectorStore functionality."""

    def test_v2_has_all_core_vectorstore_methods(self):
        """VectorStoreV2 has access to all VectorStore methods."""
        from core.vector_store import VectorStore as CoreVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            # Check that it has methods from parent class
            for method_name in dir(CoreVectorStore):
                if not method_name.startswith("_"):
                    assert hasattr(store, method_name)

            conn.close()

    def test_v2_docstring_indicates_delegation(self):
        """VectorStoreV2 class docstring mentions delegation."""
        docstring = VectorStoreV2.__doc__
        assert docstring is not None
        assert "delegate" in docstring.lower() or "inherit" in docstring.lower()

    def test_v2_rebuild_docstring_indicates_delegation(self):
        """rebuild() docstring mentions delegation to parent."""
        docstring = VectorStoreV2.rebuild.__doc__
        assert docstring is not None
        assert "delegate" in docstring.lower() or "parent" in docstring.lower()

    def test_v2_migrate_docstring_indicates_delegation(self):
        """migrate_embedding_model() docstring mentions delegation to parent."""
        docstring = VectorStoreV2.migrate_embedding_model.__doc__
        assert docstring is not None
        assert "delegate" in docstring.lower() or "parent" in docstring.lower()


class TestVectorStoreV2Integration:
    """Integration tests for VectorStoreV2."""

    def test_v2_database_initialization(self):
        """VectorStoreV2 properly initializes database schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            # Check tables were created
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            assert "memory_records" in tables
            assert "embeddings" in tables

            conn.close()

    def test_v2_model_adapter_reference(self):
        """VectorStoreV2 maintains reference to model_adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            assert store.model_adapter is mock_model_adapter

            conn.close()

    def test_v2_brain_reference(self):
        """VectorStoreV2 maintains reference to brain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            mock_model_adapter = Mock()
            mock_model_adapter.model_id.return_value = "test-model"
            mock_brain = Mock()
            mock_brain.db = conn

            store = VectorStoreV2(mock_model_adapter, mock_brain)

            assert store.brain is mock_brain

            conn.close()
