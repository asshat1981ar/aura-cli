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
