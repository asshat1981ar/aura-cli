import pytest
import sqlite3
import numpy as np
from unittest.mock import MagicMock
from core.vector_store import VectorStore
from core.context_manager import ContextManager
from core.memory_types import MemoryRecord, RetrievalQuery
from memory.embedding_provider import LocalEmbeddingProvider

@pytest.fixture
def mock_brain():
    conn = sqlite3.connect(":memory:")
    brain = MagicMock()
    brain.db = conn
    return brain

def test_fts_schema_creation(mock_brain):
    provider = LocalEmbeddingProvider()
    store = VectorStore(provider, mock_brain)
    
    # Check if FTS table exists
    cursor = mock_brain.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_records_fts'")
    assert cursor.fetchone() is not None

def test_fts_sync_and_search(mock_brain):
    provider = LocalEmbeddingProvider()
    store = VectorStore(provider, mock_brain)
    
    # Insert record with specific keyword
    rec = MemoryRecord(
        id="1",
        content="The quick brown fox jumps over the lazy dog",
        source_type="test",
        source_ref="ref",
        created_at=0,
        updated_at=0,
        content_hash="hash1",
        tags=["animal"]
    )
    store.upsert([rec])
    
    # Force fallback by mocking provider failure
    store.embedding_provider.embed = MagicMock(side_effect=Exception("Embedding failed"))
    
    # Search for "fox" using fallback
    hits = store.search("fox", k=1)
    
    assert len(hits) == 1
    assert "fox" in hits[0]

def test_fts_search_hit_structure(mock_brain):
    provider = LocalEmbeddingProvider()
    store = VectorStore(provider, mock_brain)
    
    rec = MemoryRecord(
        id="1", content="keyword match", source_type="test", source_ref="ref", 
        created_at=0, updated_at=0, content_hash="h1"
    )
    store.upsert([rec])
    
    # Force fallback
    store.embedding_provider.embed = MagicMock(return_value=[])
    
    query = RetrievalQuery(query_text="keyword", k=1)
    hits = store.search(query)
    
    assert len(hits) == 1
    assert hits[0].record_id == "1"
    assert "FTS Rank" in hits[0].explanation

def test_context_truncation(mock_brain):
    provider = LocalEmbeddingProvider()
    store = VectorStore(provider, mock_brain)
    cm = ContextManager(vector_store=store, max_tokens=100)
    
    # Setup snippets budget (50% of 100 = 50 tokens)
    # 50 tokens = 200 chars
    
    # Add a snippet that is larger than budget (e.g. 300 chars)
    long_content = "x" * 300
    rec = MemoryRecord(
        id="1", content=long_content, source_type="test", source_ref="ref", 
        created_at=0, updated_at=0, content_hash="h1"
    )
    store.upsert([rec])
    
    # ContextManager logic:
    # 1. Search semantic snippets
    # 2. Add to bundle, truncating if needed
    
    # We need to ensure the search actually returns the hit, even if score is low
    # Set min_score to 0.0 via monkeypatch or config if possible.
    # But ContextManager reads config inside.
    # Let's mock _get_semantic_snippets to bypass search and return our long snippet
    
    cm._get_semantic_snippets = MagicMock(return_value=[{
        "content": long_content,
        "source": "ref",
        "score": 0.9,
        "explanation": "mock"
    }])
    
    bundle = cm.get_context_bundle(goal="test", goal_type="default")
    
    snippets = bundle["snippets"]
    assert len(snippets) == 1
    
    # Should be truncated to fit budget (approx 50 tokens * 4 chars = 200 chars)
    content = snippets[0]["content"]
    assert len(content) <= 200
    assert "truncated" in snippets[0]["explanation"]
