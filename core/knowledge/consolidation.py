"""Knowledge consolidation for merging and deduplicating entries."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.knowledge.base import KnowledgeCategory, KnowledgeEntry
from core.logging_utils import log_json


@dataclass
class ConsolidationResult:
    """Result of knowledge consolidation."""
    merged_entries: List[KnowledgeEntry]
    removed_entry_ids: List[str]
    remaining_count: int
    similarity_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clusters: List[List[str]] = field(default_factory=list)


@dataclass
class EntryCluster:
    """A cluster of similar knowledge entries."""
    cluster_id: str
    entry_ids: List[str]
    centroid: Optional[List[float]] = None
    representative_id: Optional[str] = None


class KnowledgeConsolidator:
    """Merges and deduplicates knowledge entries."""
    
    def __init__(self, embedding_provider=None):
        """
        Initialize the consolidator.
        
        Args:
            embedding_provider: Provider for generating embeddings
        """
        self.embedding = embedding_provider
        self.similarity_threshold = 0.85
        self.min_cluster_size = 2
        self.max_cluster_size = 10
        
    async def consolidate(
        self, 
        entries: List[KnowledgeEntry],
        similarity_threshold: Optional[float] = None
    ) -> ConsolidationResult:
        """
        Consolidate entries by merging similar ones.
        
        Args:
            entries: Knowledge entries to consolidate
            similarity_threshold: Override default similarity threshold
            
        Returns:
            ConsolidationResult with merged entries and removed IDs
        """
        threshold = similarity_threshold or self.similarity_threshold
        
        if len(entries) < 2:
            return ConsolidationResult(
                merged_entries=entries,
                removed_entry_ids=[],
                remaining_count=len(entries)
            )
        
        log_json("INFO", "consolidation_started", {
            "entry_count": len(entries),
            "threshold": threshold
        })
        
        # Ensure all entries have embeddings
        entries = await self._ensure_embeddings(entries)
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(entries)
        
        # Cluster similar entries
        clusters = self._cluster_entries(entries, similarity_matrix, threshold)
        
        # Merge clusters
        merged_entries = []
        removed_ids = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Singleton cluster, keep as-is
                entry = next(e for e in entries if e.entry_id == cluster[0])
                merged_entries.append(entry)
            else:
                # Merge cluster
                cluster_entries = [e for e in entries if e.entry_id in cluster]
                merged = self._merge_cluster(cluster_entries, similarity_matrix)
                merged_entries.append(merged)
                
                # Mark originals for removal (except representative)
                for entry in cluster_entries:
                    if entry.entry_id != merged.entry_id:
                        removed_ids.append(entry.entry_id)
        
        log_json("INFO", "consolidation_completed", {
            "original_count": len(entries),
            "merged_count": len(merged_entries),
            "removed_count": len(removed_ids),
            "clusters_formed": len([c for c in clusters if len(c) > 1])
        })
        
        return ConsolidationResult(
            merged_entries=merged_entries,
            removed_entry_ids=removed_ids,
            remaining_count=len(merged_entries),
            similarity_matrix=similarity_matrix,
            clusters=clusters
        )
    
    async def _ensure_embeddings(
        self, 
        entries: List[KnowledgeEntry]
    ) -> List[KnowledgeEntry]:
        """Generate embeddings for entries that don't have them."""
        if not self.embedding:
            return entries
        
        for entry in entries:
            if entry.embedding is None:
                try:
                    if hasattr(self.embedding, 'embed'):
                        entry.embedding = await self.embedding.embed(entry.content)
                    elif hasattr(self.embedding, 'get_embedding'):
                        result = self.embedding.get_embedding(entry.content)
                        if hasattr(result, '__await__'):
                            entry.embedding = await result
                        else:
                            entry.embedding = result
                except Exception as e:
                    log_json("WARN", "embedding_generation_failed", {
                        "entry_id": entry.entry_id,
                        "error": str(e)
                    })
        
        return entries
    
    def _calculate_similarity_matrix(
        self, 
        entries: List[KnowledgeEntry]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate pairwise cosine similarity between entries."""
        matrix: Dict[str, Dict[str, float]] = {}
        
        for i, entry_a in enumerate(entries):
            matrix[entry_a.entry_id] = {}
            
            for j, entry_b in enumerate(entries):
                if i == j:
                    matrix[entry_a.entry_id][entry_b.entry_id] = 1.0
                elif i < j:
                    # Calculate similarity
                    sim = self._calculate_similarity(entry_a, entry_b)
                    matrix[entry_a.entry_id][entry_b.entry_id] = sim
                    
                    # Symmetric
                    if entry_b.entry_id not in matrix:
                        matrix[entry_b.entry_id] = {}
                    matrix[entry_b.entry_id][entry_a.entry_id] = sim
        
        return matrix
    
    def _calculate_similarity(
        self, 
        a: KnowledgeEntry, 
        b: KnowledgeEntry
    ) -> float:
        """Calculate similarity between two entries."""
        scores = []
        
        # Embedding similarity (primary)
        if a.embedding and b.embedding:
            embedding_sim = self._cosine_similarity(a.embedding, b.embedding)
            scores.append((embedding_sim, 0.6))  # Weight: 60%
        
        # Category match
        if a.category == b.category:
            scores.append((1.0, 0.15))
        else:
            scores.append((0.0, 0.15))
        
        # Tag overlap
        if a.tags and b.tags:
            intersection = set(a.tags) & set(b.tags)
            union = set(a.tags) | set(b.tags)
            tag_sim = len(intersection) / len(union) if union else 0
            scores.append((tag_sim, 0.15))
        
        # Source similarity
        if a.source == b.source:
            scores.append((1.0, 0.1))
        else:
            scores.append((0.0, 0.1))
        
        # Weighted average
        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(s * w for s, w in scores)
        return weighted_sum / total_weight
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _cluster_entries(
        self,
        entries: List[KnowledgeEntry],
        similarity_matrix: Dict[str, Dict[str, float]],
        threshold: float
    ) -> List[List[str]]:
        """Cluster entries using agglomerative clustering."""
        if not entries:
            return []
        
        # Start with each entry as its own cluster
        clusters: List[Set[str]] = [{e.entry_id} for e in entries]
        entry_map = {e.entry_id: e for e in entries}
        
        # Iteratively merge closest clusters
        while True:
            best_merge = None
            best_similarity = threshold
            
            # Find closest pair of clusters
            for i, cluster_a in enumerate(clusters):
                for j, cluster_b in enumerate(clusters[i+1:], i+1):
                    # Calculate average linkage similarity
                    similarities = []
                    for id_a in cluster_a:
                        for id_b in cluster_b:
                            sim = similarity_matrix.get(id_a, {}).get(id_b, 0)
                            similarities.append(sim)
                    
                    if similarities:
                        avg_sim = sum(similarities) / len(similarities)
                        if avg_sim > best_similarity:
                            best_similarity = avg_sim
                            best_merge = (i, j)
            
            if best_merge is None:
                break
            
            # Merge clusters
            i, j = best_merge
            new_cluster = clusters[i] | clusters[j]
            
            # Check size constraint
            if len(new_cluster) > self.max_cluster_size:
                continue
            
            # Remove old clusters and add merged
            clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
            clusters.append(new_cluster)
        
        # Convert to sorted lists
        return [sorted(list(c)) for c in clusters]
    
    def _merge_cluster(
        self,
        entries: List[KnowledgeEntry],
        similarity_matrix: Dict[str, Dict[str, float]]
    ) -> KnowledgeEntry:
        """Merge a cluster of entries into a single entry."""
        if len(entries) == 1:
            return entries[0]
        
        # Select representative (entry with highest average similarity to others)
        avg_similarities = []
        for entry in entries:
            similarities = [
                similarity_matrix.get(entry.entry_id, {}).get(other.entry_id, 0)
                for other in entries
                if other.entry_id != entry.entry_id
            ]
            avg_sim = sum(similarities) / len(similarities) if similarities else 0
            avg_similarities.append((entry, avg_sim))
        
        representative = max(avg_similarities, key=lambda x: x[1])[0]
        
        # Aggregate metadata
        all_tags: Set[str] = set()
        all_sources: Set[str] = set()
        max_confidence = 0.0
        all_context: Dict[str, Any] = {}
        
        for entry in entries:
            all_tags.update(entry.tags)
            all_sources.add(entry.source)
            max_confidence = max(max_confidence, entry.confidence)
            all_context.update(entry.context)
        
        # Build merged content
        content_parts = [representative.content]
        
        # Add unique insights from other entries
        for entry in entries:
            if entry.entry_id != representative.entry_id:
                # Check if content adds something new
                if not self._content_overlap(representative.content, entry.content):
                    content_parts.append(f"\n\nAdditional insight: {entry.content}")
        
        merged_content = "\n".join(content_parts)
        
        # Create merged entry
        return KnowledgeEntry(
            entry_id=f"merged_{representative.entry_id}",
            content=merged_content,
            source=f"merged({','.join(sorted(all_sources))})",
            category=representative.category,
            confidence=min(1.0, max_confidence + 0.05),  # Slight boost for consensus
            tags=sorted(all_tags),
            related_entries=[],
            context={
                **all_context,
                "merged_from": [e.entry_id for e in entries],
                "merge_count": len(entries),
                "original_representative": representative.entry_id
            }
        )
    
    def _content_overlap(self, content_a: str, content_b: str) -> bool:
        """Check if content_b is mostly contained in content_a."""
        # Simple heuristic: check if significant portion of words overlap
        words_a = set(content_a.lower().split())
        words_b = set(content_b.lower().split())
        
        if not words_b:
            return True
        
        overlap = len(words_a & words_b)
        return overlap / len(words_b) > 0.7
    
    def find_duplicates(
        self, 
        entries: List[KnowledgeEntry],
        threshold: Optional[float] = None
    ) -> List[Tuple[KnowledgeEntry, KnowledgeEntry, float]]:
        """
        Find potential duplicate entries.
        
        Returns:
            List of (entry_a, entry_b, similarity) tuples
        """
        threshold = threshold or self.similarity_threshold
        duplicates = []
        
        for i, entry_a in enumerate(entries):
            for entry_b in entries[i+1:]:
                sim = self._calculate_similarity(entry_a, entry_b)
                if sim >= threshold:
                    duplicates.append((entry_a, entry_b, sim))
        
        # Sort by similarity descending
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        return duplicates
