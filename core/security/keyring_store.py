"""System keyring credential store — preferred backend when available."""

from __future__ import annotations

from core.security.credential_store import CredentialStore

try:
    import keyring as _keyring

    _KEYRING_AVAILABLE = True
except ImportError:
    _keyring = None  # type: ignore[assignment]
    _KEYRING_AVAILABLE = False

_SERVICE_NAME = "aura-cli"


class KeyringStore(CredentialStore):
    """Store credentials via the OS keyring (e.g. macOS Keychain, GNOME Keyring)."""

    def __init__(self, service_name: str = _SERVICE_NAME) -> None:
        if not _KEYRING_AVAILABLE:
            raise RuntimeError("keyring package is not installed")
        self._service = service_name
        self._keys: set[str] = set()

    def get(self, key: str) -> str | None:
        value = _keyring.get_password(self._service, key)
        if value is not None:
            self._keys.add(key)
        return value

    def set(self, key: str, value: str) -> None:
        _keyring.set_password(self._service, key, value)
        self._keys.add(key)

    def delete(self, key: str) -> bool:
        try:
            _keyring.delete_password(self._service, key)
            self._keys.discard(key)
            return True
        except _keyring.errors.PasswordDeleteError:
            return False

    def list_keys(self) -> list[str]:
        return sorted(self._keys)
