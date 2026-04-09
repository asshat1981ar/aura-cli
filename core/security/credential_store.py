"""Abstract base class for credential storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class CredentialStore(ABC):
    """Interface for secure credential storage."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Retrieve a credential by key. Returns None if not found."""

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store a credential."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a credential. Returns True if it existed."""

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all stored credential keys."""
