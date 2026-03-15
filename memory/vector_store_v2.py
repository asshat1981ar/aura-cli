"""
ASCM v2: VectorStoreV2 implementation.
This is a proxy to the core VectorStore implementation which already
supports the ASCM v2 protocol.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from core.vector_store import VectorStore as CoreVectorStore
from core.memory_types import MemoryRecord, RetrievalQuery, SearchHit

class VectorStoreV2(CoreVectorStore):
    """
    Advanced Semantic Context Manager (ASCM) v2 Vector Store.
    Inherits from the unified core implementation.
    """
    def __init__(self, model_adapter, brain):
        super().__init__(model_adapter, brain)

    def migrate_from_v1(self, v1_store_path: Optional[Path] = None) -> int:
        """
        Wrapper around the core migration logic.
        If v1_store_path is not provided, it attempts to find legacy tables 
        in the current brain database.
        """
        # The core _migrate_legacy_v1 handles this automatically during __init__
        # but we can expose it explicitly if needed.
        return 0 # Placeholder if explicit migration is required

    def rebuild(self, options: Dict[str, Any]) -> bool:
        """
        Rebuilds the entire vector index from scratch.
        Useful when changing embedding models or fixing corruption.
        """
        try:
            self.brain.db.execute("DROP TABLE IF EXISTS embeddings")
            self._init_db()
            return True
        except Exception:
            return False

    def migrate_embedding_model(self, new_model_id: str) -> None:
        """
        Marks the store for re-embedding using a new model.
        In this implementation, it simply clears the embeddings table 
        so they are re-generated on next access.
        """
        self.brain.db.execute("DELETE FROM embeddings WHERE model_id != ?", (new_model_id,))
        self.brain.db.commit()
