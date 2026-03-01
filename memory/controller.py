import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.logging_utils import log_json
from memory.store import MemoryStore

class MemoryTier(Enum):
    WORKING = "working" # Volatile, per-task
    SESSION = "session" # Per-run, flushed on exit
    PROJECT = "project" # Persistent, long-term

@dataclass
class MemoryEntry:
    content: Any
    tier: MemoryTier
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MemoryController:
    """
    Unified Control Plane: Centralized memory authority for AURA.
    Manages explicit tiers, persistence policies, and garbage collection.
    
    Tiers:
    - WORKING: In-memory only, very short-term (e.g., current cycle scratchpad).
    - SESSION: In-memory only, medium-term (e.g., across multiple cycles in one run).
    - PROJECT: Persistent (via MemoryStore), long-term (e.g., past summaries, task hierarchy).
    """
    def __init__(self, store: Optional[MemoryStore] = None, max_working_size: int = 50, max_session_size: int = 500):
        self.tiers: Dict[MemoryTier, List[MemoryEntry]] = {
            MemoryTier.WORKING: [],
            MemoryTier.SESSION: [],
            MemoryTier.PROJECT: [] # Mirrored from store if available
        }
        self.persistent_store = store
        self.max_working_size = max_working_size
        self.max_session_size = max_session_size

    def set_store(self, store: MemoryStore):
        """Attaches a persistent store to the controller."""
        self.persistent_store = store
        log_json("INFO", "memory_controller_store_attached")

    def store(self, tier: MemoryTier, content: Any, metadata: Optional[Dict[str, Any]] = None):
        """Stores a piece of information in the specified tier."""
        entry = MemoryEntry(content=content, tier=tier, metadata=metadata or {})
        
        # 1. Store in memory
        self.tiers[tier].append(entry)
        
        # 2. Persist to disk if PROJECT tier
        if tier == MemoryTier.PROJECT and self.persistent_store:
            # Always wrap in a standard envelope for consistency
            record = {"content": content}
            if metadata:
                record["metadata"] = metadata
            self.persistent_store.put("project_memory", record)
        
        log_json("INFO", "memory_stored", details={"tier": tier.value, "content_snippet": str(content)[:50]})
        
        # Immediate GC for volatile tiers
        if tier == MemoryTier.WORKING:
            self._gc_tier(tier, self.max_working_size)
        elif tier == MemoryTier.SESSION:
            self._gc_tier(tier, self.max_session_size)

    def retrieve(self, tier: MemoryTier, limit: int = 100) -> List[Any]:
        """Retrieves recent entries from a tier."""
        if tier == MemoryTier.PROJECT and self.persistent_store:
            # Query from disk for PROJECT tier
            records = self.persistent_store.query("project_memory", limit=limit)
            return [r.get("content", r) for r in records]
            
        return [e.content for e in self.tiers[tier][-limit:]]

    def _gc_tier(self, tier: MemoryTier, max_size: int):
        """Enforces size limits on a memory tier."""
        if len(self.tiers[tier]) > max_size:
            removed = len(self.tiers[tier]) - max_size
            self.tiers[tier] = self.tiers[tier][-max_size:]
            log_json("INFO", "memory_gc_executed", details={"tier": tier.value, "removed_count": removed})

    def flush(self, tier: MemoryTier):
        """Clears all entries in a tier."""
        self.tiers[tier] = []
        log_json("INFO", "memory_flushed", details={"tier": tier.value})

    def checkpoint(self, brain_instance: Any = None):
        """
        Persists SESSION memory to the PROJECT tier (disk).
        Optionally records to the semantic Brain if provided.
        """
        for entry in self.tiers[MemoryTier.SESSION]:
            self.store(MemoryTier.PROJECT, entry.content, entry.metadata)
            if brain_instance and hasattr(brain_instance, "remember"):
                brain_instance.remember(entry.content)
        
        self.flush(MemoryTier.SESSION)
        log_json("INFO", "memory_checkpoint_complete")

# Global Controller Instance
memory_controller = MemoryController()
