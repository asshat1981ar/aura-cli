"""
Data types and interfaces for Advanced Semantic Context Manager (ASCM) v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union
import numpy as np

@dataclass
class MemoryRecord:
    """A unit of semantic memory."""
    id: str
    content: str
    source_type: str  # 'file', 'memory', 'goal', 'output'
    source_ref: str   # e.g., 'core/orchestrator.py:45'
    created_at: float
    updated_at: float
    goal_id: Optional[str] = None
    agent_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    token_count: int = 0
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    content_hash: str = ""
    embedding: Optional[bytes] = None  # Store binary blob of numpy array

@dataclass
class RetrievalQuery:
    """Parameters for a semantic search."""
    query_text: str
    k: int = 5
    min_score: float = 0.7
    filters: Dict[str, Any] = field(default_factory=dict)
    recency_bias: float = 0.0  # 0.0 to 1.0
    dedupe_key: Optional[str] = "content_hash"
    budget_tokens: int = 4000

@dataclass
class SearchHit:
    """A single result from a semantic search."""
    record_id: str
    content: str
    score: float
    source_ref: str
    metadata: Dict[str, Any]
    explanation: str  # Why this was retrieved

@dataclass
class ContextBundle:
    """The assembled context for an agent."""
    goal: str
    goal_type: str
    snippets: List[Dict[str, Any]] # Provenance-rich snippets
    related_insights: List[str]
    memory: List[str]
    files: List[str]
    budget_report: Dict[str, int]
    trace: Dict[str, Any]

class EmbeddingProvider(Protocol):
    """Interface for embedding generation."""
    def embed(self, texts: List[str]) -> List[np.ndarray]: ...
    def model_id(self) -> str: ...
    def dimensions(self) -> int: ...
    def healthcheck(self) -> bool: ...

class VectorStoreV2(Protocol):
    """Interface for the persistent vector store."""
    def upsert(self, records: List[MemoryRecord]) -> Dict[str, int]: ...
    def search(self, query: RetrievalQuery) -> List[SearchHit]: ...
    def delete(self, ids: List[str]) -> int: ...
    def stats(self) -> Dict[str, Any]: ...
    def rebuild(self, options: Dict[str, Any]) -> bool: ...
    def migrate_embedding_model(self, new_model_id: str) -> None: ...
