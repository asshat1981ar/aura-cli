"""Factory for selecting the best available credential store backend."""

from __future__ import annotations

import warnings

from core.security.credential_store import CredentialStore
from core.security.file_store import FileStore

_FALLBACK_WARNING = (
    "System keyring is not available; credentials will be stored in a local file. "
    "Install a keyring backend for better security."
)


def get_credential_store() -> CredentialStore:
    """Return the best available credential store.

    Tries the system keyring first; falls back to file-based storage
    with a user-visible warning.
    """
    try:
        import keyring
        import keyring.backends.fail

        backend = keyring.get_keyring()
        if isinstance(backend, keyring.backends.fail.Keyring):
            raise RuntimeError("fail backend")

        from core.security.keyring_store import KeyringStore

        return KeyringStore()
    except Exception:
        warnings.warn(_FALLBACK_WARNING, UserWarning, stacklevel=2)
        return FileStore()
