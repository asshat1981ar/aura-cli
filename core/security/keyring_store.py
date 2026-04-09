"""Keyring-backed credential store — delegates to the OS keyring."""

from __future__ import annotations

from typing import Optional

from core.security.credential_store import CredentialStore

_SERVICE_NAME = "aura-cli"


class KeyringStore(CredentialStore):
    """Stores credentials using the OS keyring (via the `keyring` library).

    Falls back to FileStore if the keyring backend is unavailable.
    """

    def __init__(self, service_name: str = _SERVICE_NAME) -> None:
        self._service = service_name
        try:
            import keyring as _kr
            self._kr = _kr
        except ImportError:
            raise ImportError(
                "keyring package is required for KeyringStore. "
                "Install with: pip install keyring"
            )
        # Track keys ourselves since keyring has no list API
        self._keys: set[str] = set()

    def get(self, key: str) -> Optional[str]:
        if not key:
            raise ValueError("Key must not be empty")
        try:
            result = self._kr.get_password(self._service, key)
            if result is not None:
                self._keys.add(key)
            return result
        except Exception:
            return None

    def set(self, key: str, value: str) -> None:
        if not key:
            raise ValueError("Key must not be empty")
        if not isinstance(value, str):
            raise TypeError(f"Value must be a string, got {type(value).__name__}")
        self._kr.set_password(self._service, key, value)
        self._keys.add(key)

    def delete(self, key: str) -> bool:
        if not key:
            raise ValueError("Key must not be empty")
        try:
            self._kr.delete_password(self._service, key)
            self._keys.discard(key)
            return True
        except Exception:
            return False

    def list_keys(self) -> list[str]:
        return sorted(self._keys)
