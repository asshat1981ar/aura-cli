import pytest
import sqlite3
import numpy as np
import time
from unittest.mock import MagicMock
from core.vector_store import VectorStore
from core.memory_types import MemoryRecord
from memory.embedding_provider import LocalEmbeddingProvider

@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    # Initialize schema handled by VectorStore
    return conn

@pytest.fixture
def mock_brain(memory_db):
    brain = MagicMock()
    brain.db = memory_db
    return brain

def test_vector_store_with_local_provider(mock_brain):
    # Setup
    provider = LocalEmbeddingProvider()
    store = VectorStore(provider, mock_brain)
    
    # Insert memories
    records = []
    texts = ["apple", "banana", "cherry"]
    metadatas = [{"type": "fruit"}, {"type": "fruit"}, {"type": "fruit"}]
    ids = ["1", "2", "3"]
    
    for i, text in enumerate(texts):
        records.append(MemoryRecord(
            id=ids[i],
            content=text,
            source_type="test",
            source_ref="test",
            created_at=time.time(),
            updated_at=time.time(),
            content_hash=text, # simple hash
            tags=metadatas[i]
        ))
    
    # Upsert
    store.upsert(records)
    
    # Verify stored in DB
    cursor = mock_brain.db.execute("SELECT content, embedding FROM memory_records WHERE id='1'")
    row = cursor.fetchone()
    assert row[0] == "apple"
    assert len(np.frombuffer(row[1], dtype=np.float32)) == 50
    
    # Search - Exact Match
    # Since we use random vectors seeded by text, searching for "apple" should find "apple" with score ~1.0
    results = store.search("apple", k=1)
    assert len(results) == 1
    # Search returns list of content strings if k=1? No, returns list of strings if query is str
    # VectorStore.search(str) -> List[str]
    assert results[0] == "apple"
    
    # Search with RetrievalQuery to get hits
    from core.memory_types import RetrievalQuery
    q = RetrievalQuery(query_text="apple", k=1)
    hits = store.search(q)
    assert len(hits) == 1
    assert hits[0].record_id == "1"
    assert hits[0].score > 0.99
    
    # Search - No Match
    results = store.search("orange", k=3)
    # It might return results but with low scores
    # But since search(str) returns content, we can't check scores easily.
    # Let's check with RetrievalQuery
    q = RetrievalQuery(query_text="orange", k=3)
    hits = store.search(q)
    if hits:
        assert hits[0].score < 0.9  # Should be lower than exact match

def test_vector_store_provider_integration(mock_brain):
    # Verify that VectorStore calls provider correctly
    mock_provider = MagicMock()
    mock_provider.dimensions.return_value = 10
    mock_provider.model_id.return_value = "mock-v1"
    # Return normalized vector for dot product to work as expected
    vec = np.array([1.0] + [0.0]*9, dtype=np.float32)
    mock_provider.embed.return_value = [vec]
    
    store = VectorStore(mock_provider, mock_brain)
    
    rec = MemoryRecord(
        id="test-id",
        content="test-content",
        source_type="test",
        source_ref="ref",
        created_at=0,
        updated_at=0,
        content_hash="hash"
    )
    
    store.upsert([rec])
    
    mock_provider.embed.assert_called_with(["test-content"])
    
    # Check DB
    # VectorStore stores embedding in 'embeddings' table now
    cursor = mock_brain.db.execute("SELECT data FROM embeddings WHERE record_id='test-id'")
    blob = cursor.fetchone()[0]
    stored_vec = np.frombuffer(blob, dtype=np.float32)
    assert np.allclose(stored_vec, vec)

