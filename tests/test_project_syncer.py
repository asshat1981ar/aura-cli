"""
Integration test for ProjectKnowledgeSyncer.
"""
import pytest
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

from core.project_syncer import ProjectKnowledgeSyncer
from core.vector_store import VectorStore
from core.memory_types import RetrievalQuery

class MockAdapter:
    def embed(self, texts):
        # Return a simple non-zero vector
        v = np.zeros(1536, dtype=np.float32)
        v[0] = 1.0
        return [v for _ in texts]
    def get_embedding(self, text):
        return self.embed([text])[0]

@pytest.fixture
def temp_project(tmp_path):
    # Create a dummy source file
    src = tmp_path / "agents"
    src.mkdir()
    py_file = src / "test_agent.py"
    py_file.write_text("""
import os
from core.logging_utils import log_json

class MyAgent:
    def run(self, data):
        return data
""", encoding="utf-8")
    return tmp_path

def test_syncer_hydrates_store(temp_project):
    db_path = temp_project / "brain.db"
    conn = sqlite3.connect(str(db_path))
    brain = MagicMock()
    brain.db = conn
    
    adapter = MockAdapter()
    vs = VectorStore(adapter, brain)
    cg = MagicMock() # Mock context graph for now
    
    syncer = ProjectKnowledgeSyncer(vs, cg, project_root=str(temp_project))
    report = syncer.sync_all()
    
    assert report["files_processed"] >= 1
    assert report["chunks_created"] >= 1
    
    # Verify VectorStore has the chunks
    stats = vs.stats()
    assert stats["record_count"] >= 1
    
    # Search for the class
    query = RetrievalQuery(query_text="MyAgent", k=1, min_score=0.0)
    hits = vs.search(query)
    assert len(hits) > 0
    assert "class MyAgent" in hits[0].content
    assert "agents/test_agent.py" in hits[0].source_ref
    
    # Verify relations (cg.add_edge called for imports)
    assert cg.add_edge.called
