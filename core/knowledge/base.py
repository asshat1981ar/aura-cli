"""Core knowledge base for agent learning and insight sharing."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json


class KnowledgeCategory(Enum):
    """Categories of knowledge entries."""
    LESSON_LEARNED = "lesson_learned"
    PATTERN = "pattern"
    BEST_PRACTICE = "best_practice"
    ANTI_PATTERN = "anti_pattern"
    TOOL_INSIGHT = "tool_insight"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    ERROR_PATTERN = "error_pattern"
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    TESTING = "testing"
    GENERAL = "general"


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge in the system."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source: str = "unknown"  # Agent or process that created this
    category: KnowledgeCategory = KnowledgeCategory.GENERAL
    confidence: float = 0.8
    tags: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: Optional[float] = None
    embedding: Optional[List[float]] = None
    
    def record_access(self):
        """Record an access to this entry."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class KnowledgeQuery:
    """Query for retrieving knowledge."""
    query_text: str = ""
    categories: Optional[List[KnowledgeCategory]] = None
    tags: Optional[List[str]] = None
    min_confidence: float = 0.5
    max_results: int = 10
    recency_weight: float = 0.3  # Weight recency in ranking
    semantic_weight: float = 0.7  # Weight semantic similarity
    min_recency_days: Optional[int] = None  # Only entries from last N days


@dataclass
class KnowledgeResult:
    """Result from a knowledge query."""
    entry: KnowledgeEntry
    semantic_score: float = 0.0
    recency_score: float = 0.0
    composite_score: float = 0.0
    rank: int = 0


class KnowledgeBase:
    """Central knowledge repository for agent learning."""
    
    def __init__(self, store=None, embedding_provider=None):
        """
        Initialize the knowledge base.
        
        Args:
            store: Optional KnowledgeStore instance
            embedding_provider: Optional embedding provider
        """
        self.store = store
        self.embedding = embedding_provider
        self._cache: Dict[str, KnowledgeEntry] = {}
        self._access_stats: Dict[str, int] = {}
        
        # Lazy initialization
        if self.store is None:
            try:
                from memory.knowledge_store import KnowledgeStore
                self.store = KnowledgeStore()
            except ImportError:
                log_json("WARN", "knowledge_store_not_available")
                self.store = None
        
        if self.embedding is None:
            try:
                from memory.embedding_provider import LocalEmbeddingProvider
                self.embedding = LocalEmbeddingProvider()
            except ImportError:
                log_json("WARN", "embedding_provider_not_available")
                self.embedding = None
    
    async def add(self, entry: KnowledgeEntry) -> str:
        """
        Add knowledge to the base with embedding.
        
        Args:
            entry: Knowledge entry to add
            
        Returns:
            Entry ID
        """
        # Generate embedding if provider available
        if self.embedding and not entry.embedding:
            try:
                entry.embedding = await self._generate_embedding(entry.content)
            except Exception as e:
                log_json("WARN", "embedding_generation_failed", {
                    "entry_id": entry.entry_id,
                    "error": str(e)
                })
        
        # Store
        if self.store:
            await self.store.save(entry)
        
        # Update cache
        self._cache[entry.entry_id] = entry
        
        log_json("INFO", "knowledge_added", {
            "entry_id": entry.entry_id,
            "category": entry.category.value,
            "source": entry.source,
            "has_embedding": entry.embedding is not None
        })
        
        return entry.entry_id
    
    async def query(self, query: KnowledgeQuery) -> List[KnowledgeResult]:
        """
        Semantic search for relevant knowledge.
        
        Args:
            query: Knowledge query
            
        Returns:
            List of knowledge results sorted by relevance
        """
        if not self.store:
            log_json("WARN", "knowledge_store_not_available_for_query")
            return []
        
        # Generate query embedding
        query_embedding = None
        if self.embedding:
            try:
                query_embedding = await self._generate_embedding(query.query_text)
            except Exception as e:
                log_json("WARN", "query_embedding_failed", {"error": str(e)})
        
        # Search store
        candidates = await self.store.search(
            query_text=query.query_text,
            query_embedding=query_embedding,
            categories=query.categories,
            tags=query.tags,
            min_confidence=query.min_confidence,
            min_recency_days=query.min_recency_days,
            limit=query.max_results * 3  # Fetch extra for re-ranking
        )
        
        if not candidates:
            return []
        
        # Calculate scores and re-rank
        results = self._calculate_scores(candidates, query, query_embedding)
        
        # Sort by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)
        
        # Assign ranks and limit results
        for i, result in enumerate(results[:query.max_results]):
            result.rank = i + 1
            result.entry.record_access()
            self._access_stats[result.entry.entry_id] = \
                self._access_stats.get(result.entry.entry_id, 0) + 1
        
        log_json("INFO", "knowledge_query_completed", {
            "query": query.query_text[:50],
            "candidates": len(candidates),
            "returned": min(len(results), query.max_results),
            "has_embedding": query_embedding is not None
        })
        
        return results[:query.max_results]
    
    async def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a specific knowledge entry by ID."""
        # Check cache first
        if entry_id in self._cache:
            entry = self._cache[entry_id]
            entry.record_access()
            return entry
        
        # Query store
        if self.store:
            entry = await self.store.get(entry_id)
            if entry:
                self._cache[entry_id] = entry
                entry.record_access()
            return entry
        
        return None
    
    async def update(self, entry: KnowledgeEntry) -> bool:
        """Update an existing knowledge entry."""
        if not self.store:
            return False
        
        # Update embedding if content changed
        if self.embedding:
            try:
                entry.embedding = await self._generate_embedding(entry.content)
            except Exception as e:
                log_json("WARN", "update_embedding_failed", {"error": str(e)})
        
        success = await self.store.update(entry)
        if success:
            self._cache[entry.entry_id] = entry
        
        return success
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        if not self.store:
            return False
        
        success = await self.store.delete(entry_id)
        if success:
            self._cache.pop(entry_id, None)
            self._access_stats.pop(entry_id, None)
        
        return success
    
    async def find_related(self, entry_id: str, 
                          max_results: int = 5) -> List[KnowledgeResult]:
        """Find entries related to a given entry."""
        entry = await self.get(entry_id)
        if not entry:
            return []
        
        # Use entry content as query
        query = KnowledgeQuery(
            query_text=entry.content,
            categories=[entry.category],
            max_results=max_results + 1,  # +1 to exclude self
            min_confidence=0.3
        )
        
        results = await self.query(query)
        
        # Filter out the original entry
        return [r for r in results if r.entry.entry_id != entry_id][:max_results]
    
    async def get_popular(self, category: Optional[KnowledgeCategory] = None,
                         limit: int = 10) -> List[KnowledgeEntry]:
        """Get most accessed knowledge entries."""
        entries = []
        
        for entry_id, count in sorted(
            self._access_stats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit * 2]:
            entry = await self.get(entry_id)
            if entry:
                if category is None or entry.category == category:
                    entries.append(entry)
                    
            if len(entries) >= limit:
                break
        
        return entries
    
    async def get_recent(self, category: Optional[KnowledgeCategory] = None,
                        limit: int = 10,
                        days: Optional[int] = None) -> List[KnowledgeEntry]:
        """Get recently created knowledge entries."""
        if not self.store:
            return []
        
        return await self.store.get_recent(
            category=category,
            limit=limit,
            days=days
        )
    
    async def consolidate(self, 
                         similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """Consolidate similar knowledge entries."""
        if not self.store:
            return {"consolidated": 0, "removed": 0}
        
        try:
            from core.knowledge.consolidation import KnowledgeConsolidator
            
            consolidator = KnowledgeConsolidator(self.embedding)
            
            # Get all entries
            all_entries = await self.store.get_all()
            
            # Consolidate
            result = await consolidator.consolidate(
                all_entries, 
                similarity_threshold
            )
            
            # Update store
            for merged in result.merged_entries:
                await self.store.save(merged)
            
            for removed_id in result.removed_entry_ids:
                await self.store.delete(removed_id)
                self._cache.pop(removed_id, None)
            
            log_json("INFO", "knowledge_consolidated", {
                "merged": len(result.merged_entries),
                "removed": len(result.removed_entry_ids),
                "remaining": result.remaining_count
            })
            
            return {
                "consolidated": len(result.merged_entries),
                "removed": len(result.removed_entry_ids),
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            log_json("ERROR", "consolidation_failed", {"error": str(e)})
            return {"consolidated": 0, "removed": 0, "error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        stats = {
            "cached_entries": len(self._cache),
            "total_accesses": sum(self._access_stats.values()),
        }
        
        if self.store:
            try:
                store_stats = self.store.get_statistics()
                stats.update(store_stats)
            except Exception:
                pass
        
        return stats
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.embedding:
            raise RuntimeError("No embedding provider available")
        
        # Handle different embedding provider interfaces
        if hasattr(self.embedding, 'embed'):
            return await self.embedding.embed(text)
        elif hasattr(self.embedding, 'get_embedding'):
            result = self.embedding.get_embedding(text)
            if hasattr(result, '__await__'):
                return await result
            return result
        else:
            raise RuntimeError("Embedding provider has no embed method")
    
    def _calculate_scores(
        self, 
        candidates: List[KnowledgeEntry],
        query: KnowledgeQuery,
        query_embedding: Optional[List[float]]
    ) -> List[KnowledgeResult]:
        """Calculate composite scores for candidates."""
        results = []
        
        current_time = time.time()
        
        for entry in candidates:
            # Semantic similarity score
            semantic_score = 0.0
            if query_embedding and entry.embedding:
                semantic_score = self._cosine_similarity(
                    query_embedding, 
                    entry.embedding
                )
            
            # Recency score (exponential decay)
            recency_score = 0.5  # Default mid-point
            if entry.created_at:
                age_days = (current_time - entry.created_at) / (24 * 3600)
                # Score decays over 30 days
                recency_score = max(0.0, 1.0 - (age_days / 30))
            
            # Confidence boost
            confidence_boost = entry.confidence * 0.1
            
            # Composite score
            composite = (
                semantic_score * query.semantic_weight +
                recency_score * query.recency_weight +
                confidence_boost
            )
            
            results.append(KnowledgeResult(
                entry=entry,
                semantic_score=semantic_score,
                recency_score=recency_score,
                composite_score=composite
            ))
        
        return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


# Global knowledge base instance
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base
