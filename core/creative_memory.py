"""Unified memory system for creative patterns using AURA's Brain.

Replaces the creative system's in-memory Pattern Memory with AURA's
persistent SQLite + NetworkX graph storage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from core.logging_utils import log_json
from core.exceptions import RecallError, StorageError


@dataclass
class CreativePattern:
    """A creative pattern with metadata."""
    id: str
    content: str
    domain: str
    technique: str
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    related_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_memory_text(self) -> str:
        """Convert to text for brain storage."""
        return f"""Creative Pattern [{self.id}]:
Domain: {self.domain}
Technique: {self.technique}
Content: {self.content}
Success Rate: {self.success_rate:.2f} ({self.usage_count} uses)
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreativePattern":
        """Create from dictionary."""
        return cls(**data)


class CreativePatternMemory:
    """Persistent pattern memory backed by AURA's Brain.
    
    This replaces the creative system's in-memory Pattern Memory with
    AURA's persistent storage, enabling:
    - Cross-session pattern recall
    - Semantic search via vector similarity
    - Pattern relationship tracking
    - Success rate analytics
    
    Example:
        >>> from memory.brain import Brain
        >>> brain = Brain()
        >>> memory = CreativePatternMemory(brain)
        >>> 
        >>> # Store a pattern
        >>> pattern = CreativePattern(
        ...     id="scamper-substitute-001",
        ...     content="Replace REST with GraphQL",
        ...     domain="api_design",
        ...     technique="SCAMPER"
        ... )
        >>> memory.record(pattern)
        >>> 
        >>> # Recall patterns
        >>> patterns = memory.recall("api_design", "reduce latency")
    """
    
    # Tag prefix for creative patterns in brain
    PATTERN_TAG = "creative_pattern"
    
    def __init__(self, brain: Any):
        """Initialize with AURA Brain instance.
        
        Args:
            brain: AURA Brain instance for storage
        """
        self.brain = brain
        self._cache: Dict[str, CreativePattern] = {}
        self._domains: Set[str] = set()
    
    def record(
        self,
        pattern: CreativePattern,
        track_usage: bool = False,
    ) -> None:
        """Persist pattern to AURA's brain.
        
        Args:
            pattern: The creative pattern to store
            track_usage: Whether this is a usage update
            
        Raises:
            StorageError: If storage fails
        """
        try:
            # Update usage if tracking
            if track_usage:
                pattern.usage_count += 1
                pattern.last_used = datetime.now().isoformat()
            
            # Store in brain with tagging
            memory_text = pattern.to_memory_text()
            metadata = {
                "type": self.PATTERN_TAG,
                "pattern_id": pattern.id,
                "domain": pattern.domain,
                "technique": pattern.technique,
                "success_rate": pattern.success_rate,
                "usage_count": pattern.usage_count,
                "created_at": pattern.created_at,
                "last_used": pattern.last_used,
                "related_patterns": json.dumps(pattern.related_patterns),
            }
            
            self.brain.remember(
                text=memory_text,
                tags=[self.PATTERN_TAG, pattern.domain, pattern.technique],
                metadata=metadata,
            )
            
            # Update cache and domains
            self._cache[pattern.id] = pattern
            self._domains.add(pattern.domain)
            
            log_json(
                "INFO",
                "creative_pattern_recorded",
                {"pattern_id": pattern.id, "domain": pattern.domain}
            )
            
        except Exception as e:
            raise StorageError(f"Failed to store pattern {pattern.id}: {e}") from e
    
    def recall(
        self,
        domain: str,
        query: str,
        top_k: int = 5,
        min_success_rate: float = 0.0,
    ) -> List[CreativePattern]:
        """Recall patterns using semantic search.
        
        Args:
            domain: Pattern domain to search
            query: Search query for semantic matching
            top_k: Maximum patterns to return
            min_success_rate: Minimum success rate filter
            
        Returns:
            List of matching creative patterns
            
        Raises:
            RecallError: If recall fails
        """
        try:
            # Use brain's semantic search
            search_query = f"{self.PATTERN_TAG} {domain}: {query}"
            
            if hasattr(self.brain, 'recall_with_budget'):
                memories = self.brain.recall_with_budget(
                    query=search_query,
                    max_tokens=4000,
                )
            elif hasattr(self.brain, 'search'):
                memories = self.brain.search(
                    query=search_query,
                    limit=top_k * 2,  # Get extra for filtering
                )
            else:
                memories = []
            
            # Parse and filter patterns
            patterns = []
            for memory in memories:
                pattern = self._parse_memory(memory)
                if pattern and pattern.domain == domain:
                    if pattern.success_rate >= min_success_rate:
                        patterns.append(pattern)
            
            # Sort by success rate and usage
            patterns.sort(
                key=lambda p: (p.success_rate, p.usage_count),
                reverse=True
            )
            
            log_json(
                "INFO",
                "creative_patterns_recalled",
                {
                    "domain": domain,
                    "query": query,
                    "found": len(patterns),
                }
            )
            
            return patterns[:top_k]
            
        except Exception as e:
            raise RecallError(f"Failed to recall patterns: {e}") from e
    
    def cross_pollinate(
        self,
        source_domain: str,
        target_domain: str,
        analogy_query: str = "",
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find patterns from source domain applicable to target.
        
        Uses analogy detection to find cross-domain inspiration.
        
        Args:
            source_domain: Domain to draw patterns from
            target_domain: Domain to apply patterns to
            analogy_query: Optional query for analogy matching
            top_k: Maximum analogies to return
            
        Returns:
            List of pattern analogies with relevance scores
        """
        # Get patterns from source domain
        source_patterns = self.recall(
            domain=source_domain,
            query=analogy_query or "*",
            top_k=10,
        )
        
        analogies = []
        for pattern in source_patterns:
            # Calculate analogy relevance (simplified)
            relevance = self._calculate_analogy_relevance(
                pattern, target_domain
            )
            
            if relevance > 0.3:  # Minimum threshold
                analogies.append({
                    "pattern": pattern,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "relevance": relevance,
                    "adaptation_hint": self._generate_adaptation_hint(
                        pattern, target_domain
                    ),
                })
        
        # Sort by relevance
        analogies.sort(key=lambda a: a["relevance"], reverse=True)
        
        return analogies[:top_k]
    
    def update_success_rate(
        self,
        pattern_id: str,
        success: bool,
    ) -> None:
        """Update pattern success rate based on implementation result.
        
        Args:
            pattern_id: Pattern to update
            success: Whether implementation succeeded
        """
        try:
            # Get existing pattern
            pattern = self._cache.get(pattern_id)
            if not pattern and hasattr(self.brain, 'get'):
                memory = self.brain.get(pattern_id)
                pattern = self._parse_memory(memory) if memory else None
            
            if pattern:
                # Update success rate with exponential moving average
                old_rate = pattern.success_rate
                alpha = 0.3  # Learning rate
                new_success = 1.0 if success else 0.0
                pattern.success_rate = (alpha * new_success) + ((1 - alpha) * old_rate)
                
                # Re-store updated pattern
                self.record(pattern, track_usage=True)
        
        except Exception as e:
            log_json("WARN", "pattern_success_update_failed", {"error": str(e)})
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a domain.
        
        Args:
            domain: Domain to analyze
            
        Returns:
            Statistics dictionary
        """
        try:
            patterns = self.recall(domain, "*", top_k=1000)
            
            if not patterns:
                return {"domain": domain, "pattern_count": 0}
            
            return {
                "domain": domain,
                "pattern_count": len(patterns),
                "avg_success_rate": sum(p.success_rate for p in patterns) / len(patterns),
                "total_usage": sum(p.usage_count for p in patterns),
                "techniques": list(set(p.technique for p in patterns)),
            }
        
        except Exception as e:
            log_json("WARN", "domain_stats_failed", {"error": str(e)})
            return {"domain": domain, "error": str(e)}
    
    def list_domains(self) -> List[str]:
        """List all known pattern domains."""
        return sorted(self._domains)
    
    def _parse_memory(self, memory: Any) -> Optional[CreativePattern]:
        """Parse brain memory into CreativePattern."""
        try:
            if isinstance(memory, dict):
                metadata = memory.get("metadata", {})
                if metadata.get("type") == self.PATTERN_TAG:
                    return CreativePattern(
                        id=metadata.get("pattern_id", "unknown"),
                        content=memory.get("text", ""),
                        domain=metadata.get("domain", "general"),
                        technique=metadata.get("technique", "unknown"),
                        success_rate=metadata.get("success_rate", 0.0),
                        usage_count=metadata.get("usage_count", 0),
                        created_at=metadata.get("created_at"),
                        last_used=metadata.get("last_used"),
                        related_patterns=json.loads(
                            metadata.get("related_patterns", "[]")
                        ),
                    )
            return None
        except Exception:
            return None
    
    def _calculate_analogy_relevance(
        self,
        pattern: CreativePattern,
        target_domain: str,
    ) -> float:
        """Calculate how relevant a pattern is to target domain."""
        # Simplified relevance scoring
        base_score = pattern.success_rate * 0.5
        
        # Boost for high-usage patterns
        usage_boost = min(pattern.usage_count / 10, 0.3)
        
        return base_score + usage_boost
    
    def _generate_adaptation_hint(
        self,
        pattern: CreativePattern,
        target_domain: str,
    ) -> str:
        """Generate hint for adapting pattern to target domain."""
        return (
            f"Consider applying '{pattern.technique}' technique from "
            f"{pattern.domain} to {target_domain}: {pattern.content[:100]}..."
        )


def create_creative_memory(brain: Any) -> CreativePatternMemory:
    """Factory function for CreativePatternMemory.
    
    Args:
        brain: AURA Brain instance
        
    Returns:
        Configured CreativePatternMemory
    """
    return CreativePatternMemory(brain=brain)
