"""Abstract credential store interface.

Defines the contract for credential storage backends.
All implementations must support set/get/delete/list_keys operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CredentialStore(ABC):
    """Abstract base class for credential storage backends."""

    @abstractmethod
    def set(self, name: str, secret: str) -> None:
        """Store a credential.

        Args:
            name: Credential identifier (non-empty string).
            secret: Secret value to store.

        Raises:
            ValueError: If name is empty.
        """

    @abstractmethod
    def get(self, name: str) -> str | None:
        """Retrieve a stored credential.

        Args:
            name: Credential identifier.

        Returns:
            The secret value, or None if not found.
        """

    @abstractmethod
    def delete(self, name: str) -> None:
        """Delete a stored credential.

        Does not raise if the credential does not exist.

        Args:
            name: Credential identifier.
        """

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all stored credential names.

        Returns:
            List of credential identifiers.
        """
