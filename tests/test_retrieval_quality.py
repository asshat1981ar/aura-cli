"""
Regression tests for ASCM retrieval quality.
Ensures VectorStore ranking logic remains stable using deterministic fixtures.
"""
import json
import pytest
import numpy as np
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

from core.vector_store import VectorStore
from core.memory_types import MemoryRecord, RetrievalQuery

# Deterministic vectors for the fixture data
# We design them so dot-product ranking matches our expectations
VECTORS = {
    "rec_1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), # Math
    "rec_2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), # Auth
    "rec_3": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), # DB
    "rec_4": np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32), # Auth + slight DB mix
    
    # Query vectors
    "q_auth": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "q_db":   np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "q_math": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
}

class DeterministicMockAdapter:
    """Returns pre-calculated vectors based on content or query text."""
    def __init__(self):
        self._embedding_model = "test-deterministic"
        self._embedding_dims = 4

    def get_embedding(self, text):
        # Map text to vector
        if "calculate_metric" in text or "calculate metric" in text:
            return VECTORS["rec_1"] # or q_math
        if "UserAuth" in text or "authenticate" in text:
            return VECTORS["q_auth"]
        if "save_to_db" in text or "database saving" in text:
            return VECTORS["q_db"]
        if "middleware" in text:
            return VECTORS["rec_4"]
        return np.zeros(4, dtype=np.float32)

@pytest.fixture
def golden_set():
    path = Path(__file__).parent / "fixtures" / "retrieval_golden_set.json"
    with open(path, "r") as f:
        return json.load(f)

@pytest.fixture
def vector_store(tmp_path):
    db_path = tmp_path / "brain_test.db"
    conn = sqlite3.connect(str(db_path))
    brain = MagicMock()
    brain.db = conn
    return VectorStore(DeterministicMockAdapter(), brain)

def test_retrieval_quality_golden_set(vector_store, golden_set):
    """
    Verifies that VectorStore correctly ranks expected items for known queries.
    Uses hardcoded vectors to isolate ranking logic from embedding quality.
    """
    # 1. Index Records
    records = []
    for r in golden_set["records"]:
        # Inject our deterministic vectors manually to bypass the adapter for records
        # Wait, VectorStore.upsert calls adapter.get_embedding if embedding is None.
        # We can set the embedding field directly in the record.
        
        # Match ID to vector
        vec = VECTORS.get(r["id"])
        
        rec = MemoryRecord(
            id=r["id"],
            content=r["content"],
            source_type="fixture",
            source_ref=r["source_ref"],
            created_at=0,
            updated_at=0,
            content_hash=r["id"],
            embedding=vec.tobytes() if vec is not None else None
        )
        records.append(rec)
    
    vector_store.upsert(records)
    
    # 2. Run Queries
    for q_case in golden_set["queries"]:
        query_text = q_case["text"]
        expected_ids = set(q_case["expected_ids"])
        
        # We map query text to vector in the mock adapter
        query = RetrievalQuery(
            query_text=query_text,
            k=len(expected_ids) + 1, # Fetch enough to verify ranking
            min_score=0.5
        )
        
        hits = vector_store.search(query)
        hit_ids = [h.record_id for h in hits]
        
        # 3. Assert Recall (Are the expected items found?)
        # For this deterministic test, we expect perfect recall because vectors align perfectly.
        found = set(hit_ids) & expected_ids
        recall = len(found) / len(expected_ids)
        
        assert recall == 1.0, f"Query '{query_text}' missed expected items. Found: {hit_ids}, Expected: {expected_ids}"
        
        # 4. Assert Rank (For 'auth', rec_2 (exact match) should beat rec_4 (0.9 match))
        if query_text == "How do I authenticate users?":
            # rec_2 is [0,1,0,0], query is [0,1,0,0] -> score 1.0
            # rec_4 is [0, 0.9, 0.1, 0], query is [0,1,0,0] -> score ~0.9
            assert hit_ids[0] == "rec_2", "Exact match should be ranked first"
