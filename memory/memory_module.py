"""Base interface for the modular memory architecture.

Defines the MemoryEntry data class and the MemoryModule abstract base class
that all memory modules (working, episodic, semantic, procedural) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """A single memory record returned by any memory module.

    Attributes:
        id: Unique identifier for this entry.
        content: The textual content of the memory.
        metadata: Arbitrary key-value metadata attached to the entry.
        timestamp: UNIX timestamp when the entry was created.
        score: Relevance score assigned during search/retrieval (0.0 = unscored).
    """

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = 0.0
    score: float = 0.0


class MemoryModule(ABC):
    """Base interface for all memory modules.

    Every memory module provides a uniform CRUD + search surface so that
    the MemoryManager can orchestrate them without knowing implementation
    details.
    """

    @abstractmethod
    def write(self, content: str, metadata: Optional[dict] = None) -> str:
        """Store a new memory entry.

        Args:
            content: The textual content to store.
            metadata: Optional key-value metadata to attach.

        Returns:
            The unique ID of the newly created entry.
        """

    @abstractmethod
    def read(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve relevant entries for *query*.

        Args:
            query: A search string or key used to find matching entries.
            top_k: Maximum number of entries to return.

        Returns:
            A list of matching MemoryEntry objects, ordered by relevance.
        """

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """Semantic or keyword search across all stored entries.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            A list of MemoryEntry objects ranked by relevance score.
        """

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Remove an entry by its ID.

        Args:
            entry_id: The unique identifier of the entry to remove.

        Returns:
            True if the entry was found and deleted, False otherwise.
        """

    @abstractmethod
    def clear(self) -> None:
        """Wipe all entries from this module."""

    @abstractmethod
    def stats(self) -> dict:
        """Return module statistics.

        Returns:
            A dict with at minimum ``count`` (int) and ``type`` (str) keys.
            Implementations may add module-specific keys (e.g. ``storage_bytes``).
        """
