"""
Tests for Advanced Semantic Context Manager (ASCM) v2.
Covers VectorStore, ContextManager, and Data Models.
"""
import time
import pytest
from unittest.mock import MagicMock
import numpy as np
from core.vector_store import VectorStore
from core.context_manager import ContextManager
from core.memory_types import MemoryRecord, RetrievalQuery
from memory.brain import Brain
import sqlite3

class MockModelAdapter:
    def __init__(self):
        self._embedding_model = "test-model"
        self._embedding_dims = 4

    def model_id(self):
        return self._embedding_model

    def embed(self, texts):
        # Deterministic dummy embeddings based on text length
        embeddings = []
        for text in texts:
            val = len(text) / 100.0
            embeddings.append(np.array([val, val, val, val], dtype=np.float32))
        return embeddings

    def get_embedding(self, text):
        return self.embed([text])[0]
    
    def estimate_context_budget(self, goal, goal_type="default"):
        return 100

@pytest.fixture
def mock_brain(tmp_path):
    db_path = tmp_path / "test_brain.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    brain = MagicMock()

    brain.db = conn
    return brain

@pytest.fixture
def mock_adapter():
    return MockModelAdapter()

@pytest.fixture
def vector_store(mock_adapter, mock_brain):
    return VectorStore(mock_adapter, mock_brain)

class TestVectorStore:
    def test_upsert_and_search(self, vector_store):
        rec1 = MemoryRecord(
            id="1", content="hello world", source_type="test", source_ref="test.py:1",
            created_at=time.time(), updated_at=time.time(), content_hash="hash1"
        )
        rec2 = MemoryRecord(
            id="2", content="goodbye world", source_type="test", source_ref="test.py:2",
            created_at=time.time(), updated_at=time.time(), content_hash="hash2"
        )
        
        vector_store.upsert([rec1, rec2])
        
        stats = vector_store.stats()
        assert stats["record_count"] == 2
        
        # Search
        query = RetrievalQuery(query_text="hello", k=1, min_score=0.0)
        hits = vector_store.search(query)
        assert len(hits) == 1
        assert hits[0].record_id == "1"

    def test_delete(self, vector_store):
        rec1 = MemoryRecord(
            id="1", content="hello", source_type="test", source_ref="ref", 
            created_at=0, updated_at=0, content_hash="h1"
        )
        vector_store.upsert([rec1])
        assert vector_store.stats()["record_count"] == 1
        
        vector_store.delete(["1"])
        assert vector_store.stats()["record_count"] == 0

    def test_idempotency(self, vector_store):
        rec1 = MemoryRecord(
            id="1", content="hello", source_type="test", source_ref="ref", 
            created_at=0, updated_at=0, content_hash="h1"
        )
        vector_store.upsert([rec1])
        vector_store.upsert([rec1]) # Same ID
        assert vector_store.stats()["record_count"] == 1

    def test_filters(self, vector_store):
        rec1 = MemoryRecord(
            id="1", content="foo", source_type="typeA", source_ref="ref", 
            created_at=0, updated_at=0, content_hash="h1"
        )
        rec2 = MemoryRecord(
            id="2", content="bar", source_type="typeB", source_ref="ref", 
            created_at=0, updated_at=0, content_hash="h2"
        )
        vector_store.upsert([rec1, rec2])
        
        query = RetrievalQuery(query_text="foo", filters={"source_type": "typeA"})
        hits = vector_store.search(query)
        assert len(hits) == 1
        assert hits[0].record_id == "1"

    def test_rebuild_rewrites_embeddings_for_active_model(self, vector_store, mock_adapter, mock_brain):
        rec1 = MemoryRecord(
            id="1", content="hello", source_type="test", source_ref="ref",
            created_at=0, updated_at=0, content_hash="h1"
        )
        vector_store.upsert([rec1])

        mock_adapter._embedding_model = "new-local-model"
        stats = vector_store.rebuild({})

        assert stats["records_seen"] == 1
        assert stats["embeddings_written"] == 1

        row = mock_brain.db.execute(
            "SELECT embedding_model, embedding_dims FROM memory_records WHERE id = ?",
            ("1",),
        ).fetchone()
        assert row[0] == "new-local-model"
        assert row[1] == 4

        emb_row = mock_brain.db.execute(
            "SELECT model_id, dims FROM embeddings WHERE record_id = ?",
            ("1",),
        ).fetchone()
        assert emb_row[0] == "new-local-model"
        assert emb_row[1] == 4

    def test_rebuild_with_source_type_filter(self, vector_store):
        rec_a = MemoryRecord(
            id="a", content="keep me", source_type="typeA", source_ref="ref",
            created_at=0, updated_at=0, content_hash="ha"
        )
        rec_b = MemoryRecord(
            id="b", content="skip me", source_type="typeB", source_ref="ref",
            created_at=0, updated_at=0, content_hash="hb"
        )
        vector_store.upsert([rec_a, rec_b])
        stats = vector_store.rebuild({"source_types": ["typeA"]})
        assert stats["records_seen"] == 1
        assert stats["embeddings_written"] == 1

    def test_rebuild_with_exclude_source_type(self, vector_store):
        rec_a = MemoryRecord(
            id="a", content="keep me", source_type="typeA", source_ref="ref",
            created_at=0, updated_at=0, content_hash="ha"
        )
        rec_b = MemoryRecord(
            id="b", content="skip me", source_type="typeB", source_ref="ref",
            created_at=0, updated_at=0, content_hash="hb"
        )
        vector_store.upsert([rec_a, rec_b])
        stats = vector_store.rebuild({"exclude_source_types": ["typeB"]})
        assert stats["records_seen"] == 1
        assert stats["embeddings_written"] == 1

    def test_rebuild_drop_existing_false_preserves_old_embeddings(self, vector_store, mock_adapter, mock_brain):
        rec1 = MemoryRecord(
            id="1", content="hello", source_type="test", source_ref="ref",
            created_at=0, updated_at=0, content_hash="h1"
        )
        vector_store.upsert([rec1])

        mock_adapter._embedding_model = "model-v2"
        stats = vector_store.rebuild({"drop_existing_embeddings": False})
        assert stats["embeddings_written"] == 1
        # Both the old "test-model" and new "model-v2" embeddings should be present
        count = mock_brain.db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE record_id = ?", ("1",)
        ).fetchone()[0]
        assert count == 2

    def test_migrate_embedding_model(self, vector_store, mock_adapter, mock_brain):
        rec1 = MemoryRecord(
            id="1", content="hello", source_type="test", source_ref="ref",
            created_at=0, updated_at=0, content_hash="h1"
        )
        vector_store.upsert([rec1])

        # Migrate embeddings to a new model without removing embeddings for other model_ids.
        mock_adapter._embedding_model = "migrated-model"
        stats = vector_store.migrate_embedding_model("migrated-model")

        assert stats["records_seen"] == 1
        assert stats["embeddings_written"] == 1

        # There should be exactly one embedding for the migrated model_id
        migrated_count = mock_brain.db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE record_id = ? AND model_id = ?",
            ("1", "migrated-model"),
        ).fetchone()[0]
        assert migrated_count == 1

        # The original embedding for the old model_id should still be present
        original_count = mock_brain.db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE record_id = ? AND model_id = ?",
            ("1", "test-model"),
        ).fetchone()[0]
        assert original_count == 1

        # And in total we should now have both old and new embeddings
        total_count = mock_brain.db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE record_id = ?",
            ("1",),
        ).fetchone()[0]
        assert total_count == 2

    def test_numpy_unavailable_import_does_not_raise(self):
        """Verify the graceful numpy import guard: np=None when numpy raises ImportError."""
        import sys

        real_np = sys.modules.get("numpy")
        real_core_vs = sys.modules.get("core.vector_store")

        class NumpyBlocker:
            """Module finder that blocks numpy imports by raising ModuleNotFoundError."""
            def find_spec(self, fullname, path, target=None):
                if fullname == "numpy" or fullname.startswith("numpy."):
                    raise ModuleNotFoundError(f"No module named '{fullname}'")
                return None

        try:
            blocker = NumpyBlocker()
            sys.meta_path.insert(0, blocker)
            sys.modules.pop("numpy", None)
            sys.modules.pop("core.vector_store", None)

            # Force reimport with numpy blocked
            import core.vector_store as vs_mod

            # Module should load and np should be None due to try/except guard
            assert vs_mod.np is None
        finally:
            # Cleanup
            sys.meta_path.remove(blocker)
            if real_np is not None:
                sys.modules["numpy"] = real_np
            else:
                sys.modules.pop("numpy", None)
            if real_core_vs is not None:
                sys.modules["core.vector_store"] = real_core_vs
            else:
                sys.modules.pop("core.vector_store", None)

class TestContextManager:
    def test_budget_allocation(self, vector_store):
        cm = ContextManager(vector_store=vector_store, max_tokens=100)
        # Mock budgets: 50% snippets -> 50 tokens
        
        # Add snippets that exceed budget
        long_text = "a" * 200 # ~50 tokens
        rec1 = MemoryRecord(id="1", content=long_text, source_type="test", source_ref="ref", created_at=0, updated_at=0, content_hash="h1")
        vector_store.upsert([rec1])
        
        bundle = cm.get_context_bundle(goal="test")
        
        # Should contain snippets
        assert len(bundle["snippets"]) > 0
        # Check budget report
        assert bundle["budget_report"]["snippets"] <= 50

    def test_provenance_formatting(self, vector_store):
        cm = ContextManager(vector_store=vector_store)
        rec1 = MemoryRecord(id="1", content="def foo(): pass", source_type="file", source_ref="core/utils.py:10", created_at=0, updated_at=0, content_hash="h1")
        vector_store.upsert([rec1])
        
        bundle = cm.get_context_bundle(goal="foo")
        prompt = cm.format_as_prompt(bundle)
        
        assert "core/utils.py:10" in prompt
        assert "Confidence: 1.00" in prompt
