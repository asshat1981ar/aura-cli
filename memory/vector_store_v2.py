"""
ASCM v2: VectorStoreV2 implementation.
Thin subclass of the core VectorStore that delegates all logic to the parent.
New code should import this class directly; core.vector_store re-exports it
for convenience but the canonical source is here.
"""

from __future__ import annotations

from typing import Any, Dict

from core.vector_store import VectorStore as CoreVectorStore


class VectorStoreV2(CoreVectorStore):
    """
    Advanced Semantic Context Manager (ASCM) v2 Vector Store.

    Inherits the full implementation from core.vector_store.VectorStore,
    which already implements the ASCM v2 protocol (multi-model embeddings,
    namespace partitioning, provenance-aware search, legacy migration).
    """

    def __init__(self, model_adapter, brain):
        super().__init__(model_adapter, brain)

    def rebuild(self, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Rebuild embeddings for the active model. Delegates to parent implementation."""
        return super().rebuild(options)

    def migrate_embedding_model(self, new_model_id: str) -> Dict[str, Any]:
        """Re-embed all records under a new model identifier. Delegates to parent."""
        return super().migrate_embedding_model(new_model_id)
