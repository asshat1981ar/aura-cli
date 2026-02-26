import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.logging_utils import log_json

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
    """
    def __init__(self, max_working_size: int = 50, max_session_size: int = 500):
        self.tiers: Dict[MemoryTier, List[MemoryEntry]] = {
            MemoryTier.WORKING: [],
            MemoryTier.SESSION: [],
            MemoryTier.PROJECT: []
        }
        self.max_working_size = max_working_size
        self.max_session_size = max_session_size

    def store(self, tier: MemoryTier, content: Any, metadata: Optional[Dict[str, Any]] = None):
        """Stores a piece of information in the specified tier."""
        entry = MemoryEntry(content=content, tier=tier, metadata=metadata or {})
        self.tiers[tier].append(entry)
        
        log_json("INFO", "memory_stored", details={"tier": tier.value, "content_snippet": str(content)[:50]})
        
        # Immediate GC for volatile tiers
        if tier == MemoryTier.WORKING:
            self._gc_tier(tier, self.max_working_size)
        elif tier == MemoryTier.SESSION:
            self._gc_tier(tier, self.max_session_size)

    def retrieve(self, tier: MemoryTier, limit: int = 100) -> List[Any]:
        """Retrieves recent entries from a tier."""
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

    def checkpoint(self, project_db_adapter: Any):
        """
        Persists SESSION memory to the PROJECT tier (and thus to disk).
        """
        for entry in self.tiers[MemoryTier.SESSION]:
            # Explicitly pass PROJECT tier to avoid default SESSION recursion
            project_db_adapter.remember(entry.content, tier=MemoryTier.PROJECT)
            self.tiers[MemoryTier.PROJECT].append(entry)
        
        self.flush(MemoryTier.SESSION)
        log_json("INFO", "memory_checkpoint_complete")

# Global Controller Instance
memory_controller = MemoryController()
