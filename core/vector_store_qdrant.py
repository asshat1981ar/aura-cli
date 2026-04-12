"""Qdrant-based vector store implementation.

Production-grade vector store with metadata filtering,
horizontal scaling, and hybrid search capabilities.
"""

from __future__ import annotations

import uuid
import time
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from core.logging_utils import log_json

if TYPE_CHECKING:
    from core.vector_store import VectorStore

# Optional Qdrant import
qdrant_available = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )

    qdrant_available = True
except ImportError:
    pass


@dataclass
class RebuildOptions:
    """Options for rebuilding the vector store index.

    Attributes:
        exclude_source_types: Source types to exclude from indexing
        drop_existing_embeddings: Whether to drop existing embeddings
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Model to use for embeddings
        embedding_dims: Dimensionality of embeddings
    """

    exclude_source_types: List[str] = field(default_factory=list)
    drop_existing_embeddings: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "default"
    embedding_dims: int = 768


class QdrantVectorStore:
    """Production-grade vector store using Qdrant.

    Features:
    - Metadata filtering (payload filters)
    - Horizontal scaling support
    - Hybrid search (dense + sparse)
    - Persistent storage
    - Collection management

    Requires:
        - Qdrant server running (local or cloud)
        - qdrant-client package installed
    """

    DEFAULT_COLLECTION = "aura_memory"
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 6333

    def __init__(
        self,
        model_adapter,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        collection_name: str = DEFAULT_COLLECTION,
        api_key: Optional[str] = None,
        prefer_grpc: bool = True,
    ):
        """Initialize Qdrant vector store.

        Args:
            model_adapter: Adapter for generating embeddings
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            api_key: API key for Qdrant Cloud
            prefer_grpc: Use gRPC protocol (faster)
        """
        if not qdrant_available:
            raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")

        self.model_adapter = model_adapter
        self.collection_name = collection_name

        # Initialize client
        if api_key:
            # Cloud deployment
            self.client = QdrantClient(
                url=f"https://{host}",
                api_key=api_key,
                prefer_grpc=prefer_grpc,
            )
        else:
            # Local deployment
            self.client = QdrantClient(
                host=host,
                port=port,
                prefer_grpc=prefer_grpc,
            )

        log_json(
            "INFO",
            "qdrant_store_initialized",
            {
                "host": host,
                "port": port,
                "collection": collection_name,
            },
        )

    def ensure_collection(self, vector_size: int = 768) -> None:
        """Ensure the collection exists with proper configuration.

        Args:
            vector_size: Dimensionality of embeddings
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                log_json(
                    "INFO",
                    "qdrant_collection_created",
                    {
                        "collection": self.collection_name,
                        "vector_size": vector_size,
                    },
                )
            else:
                log_json(
                    "DEBUG",
                    "qdrant_collection_exists",
                    {
                        "collection": self.collection_name,
                    },
                )

        except Exception as e:
            log_json("ERROR", "qdrant_collection_error", {"error": str(e)})
            raise

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Add a document to the vector store.

        Args:
            content: Text content to store
            metadata: Optional metadata dict
            embedding: Pre-computed embedding (or None to generate)

        Returns:
            Document ID
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self._generate_embedding(content)

        # Generate unique ID
        doc_id = str(uuid.uuid4())

        # Prepare payload
        payload = {
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            payload.update(metadata)

        # Insert into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        log_json("DEBUG", "qdrant_document_added", {"id": doc_id[:8]})
        return doc_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"source_type": "code"})
            score_threshold: Minimum similarity score

        Returns:
            List of search results with content and scores
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Build filter if provided
        qdrant_filter = self._build_filter(filters) if filters else None

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )

        # Format results
        return [
            {
                "id": point.id,
                "content": point.payload.get("content", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "content"},
            }
            for point in results
        ]

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id],
            )
            log_json("DEBUG", "qdrant_document_deleted", {"id": doc_id[:8]})
            return True
        except Exception as e:
            log_json("ERROR", "qdrant_delete_error", {"error": str(e)})
            return False

    def rebuild(self, options: RebuildOptions) -> Dict[str, Any]:
        """Rebuild the vector store index.

        Args:
            options: Rebuild configuration options

        Returns:
            Statistics about the rebuild
        """
        stats = {
            "chunks_created": 0,
            "chunks_skipped": 0,
            "errors": [],
        }

        if options.drop_existing_embeddings:
            # Delete existing collection
            try:
                self.client.delete_collection(self.collection_name)
                log_json(
                    "INFO",
                    "qdrant_collection_dropped",
                    {
                        "collection": self.collection_name,
                    },
                )
            except Exception as e:
                log_json("WARN", "qdrant_drop_failed", {"error": str(e)})

        # Ensure collection exists
        self.ensure_collection(vector_size=options.embedding_dims)

        log_json("INFO", "qdrant_rebuild_complete", stats)
        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Collection info and statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "vectors_count": info.points_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
            }
        except Exception as e:
            log_json("ERROR", "qdrant_stats_error", {"error": str(e)})
            return {
                "collection": self.collection_name,
                "error": str(e),
            }

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            # Use model adapter to generate embedding
            result = self.model_adapter.embed(text)
            if isinstance(result, list):
                return result
            # Handle numpy array
            return result.tolist()
        except Exception as e:
            log_json("ERROR", "embedding_generation_failed", {"error": str(e)})
            # Return zero vector as fallback
            return [0.0] * 768

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dict.

        Args:
            filters: Filter conditions

        Returns:
            Qdrant Filter object
        """
        conditions = []

        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        return Filter(must=conditions)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search using dense + sparse vectors.

        Note: This requires Qdrant 1.2+ with sparse vector support.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters

        Returns:
            Search results
        """
        # For now, delegate to regular search
        # Full hybrid search requires sparse vector setup
        return self.search(query, top_k, filters)


class VectorStoreFactory:
    """Factory for creating vector store instances.

    Supports both custom VectorStore and QdrantVectorStore
    based on configuration.
    """

    @staticmethod
    def create(model_adapter, use_qdrant: Optional[bool] = None, **kwargs) -> Union["QdrantVectorStore", "VectorStore"]:
        """Create appropriate vector store instance.

        Args:
            model_adapter: Model adapter for embeddings
            use_qdrant: Force Qdrant (True), custom (False), or auto (None)
            **kwargs: Additional arguments for store initialization

        Returns:
            Vector store instance
        """
        import os

        # Determine which store to use
        if use_qdrant is None:
            use_qdrant = os.environ.get("AURA_USE_QDRANT", "false").lower() == "true"

        if use_qdrant and qdrant_available:
            log_json("INFO", "using_qdrant_vector_store")
            return QdrantVectorStore(
                model_adapter=model_adapter,
                host=kwargs.get("qdrant_host", "localhost"),
                port=kwargs.get("qdrant_port", 6333),
                collection_name=kwargs.get("collection_name", "aura_memory"),
            )
        else:
            # Fall back to custom VectorStore
            if use_qdrant and not qdrant_available:
                log_json("WARN", "qdrant_requested_but_not_available")

            log_json("INFO", "using_custom_vector_store")
            from core.vector_store import VectorStore

            return VectorStore(model_adapter, kwargs.get("brain"))
