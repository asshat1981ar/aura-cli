"""Abstract base for credential storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class CredentialStore(ABC):
    """Interface for secure credential storage.

    Implementations must handle arbitrary key names and secret values
    without crashing, raising documented exceptions for invalid input.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Retrieve a secret by key. Returns None if not found."""

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store a secret under the given key."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a secret. Returns True if it existed."""

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all stored key names."""

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        return self.get(key) is not None
