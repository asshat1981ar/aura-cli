"""OS keyring-backed credential store.

Uses the system keyring (macOS Keychain, Windows Credential Locker,
Linux Secret Service) to store credentials securely.
"""

from __future__ import annotations

from core.security.credential_store import CredentialStore

try:
    import keyring as _keyring

    KEYRING_AVAILABLE = True
except ImportError:
    _keyring = None  # type: ignore[assignment]
    KEYRING_AVAILABLE = False


_SERVICE_NAME = "aura-cli"
_INDEX_KEY = "__aura_cli_keys__"
_INDEX_SEP = "\x00"


class KeyringStore(CredentialStore):
    """Credential store backed by the OS keyring."""

    def __init__(self, service_name: str = _SERVICE_NAME) -> None:
        if not KEYRING_AVAILABLE:
            raise RuntimeError(
                "keyring package not installed. Install with: pip install keyring"
            )
        self._service = service_name

    def set(self, name: str, secret: str) -> None:
        if not name:
            raise ValueError("Credential name must not be empty")
        _keyring.set_password(self._service, name, secret)
        self._add_to_index(name)

    def get(self, name: str) -> str | None:
        return _keyring.get_password(self._service, name)

    def delete(self, name: str) -> None:
        try:
            _keyring.delete_password(self._service, name)
        except _keyring.errors.PasswordDeleteError:
            pass
        self._remove_from_index(name)

    def list_keys(self) -> list[str]:
        raw = _keyring.get_password(self._service, _INDEX_KEY)
        if not raw:
            return []
        return [k for k in raw.split(_INDEX_SEP) if k]

    def _add_to_index(self, name: str) -> None:
        keys = set(self.list_keys())
        keys.add(name)
        _keyring.set_password(
            self._service, _INDEX_KEY, _INDEX_SEP.join(sorted(keys))
        )

    def _remove_from_index(self, name: str) -> None:
        keys = set(self.list_keys())
        keys.discard(name)
        if keys:
            _keyring.set_password(
                self._service, _INDEX_KEY, _INDEX_SEP.join(sorted(keys))
            )
        else:
            try:
                _keyring.delete_password(self._service, _INDEX_KEY)
            except _keyring.errors.PasswordDeleteError:
                pass
