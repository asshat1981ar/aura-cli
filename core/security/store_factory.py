"""Auto-detection factory for credential stores.

Attempts KeyringStore first; falls back to FileStore if the keyring
is unavailable or unusable.
"""

from __future__ import annotations

from pathlib import Path

from core.security.credential_store import CredentialStore


def create_credential_store(
    store_dir: Path | str | None = None,
) -> CredentialStore:
    """Create the best available credential store.

    Attempts to use the OS keyring first. If keyring is not available
    or not functional, falls back to Fernet-encrypted file store.

    Args:
        store_dir: Directory for file-based fallback storage.

    Returns:
        A CredentialStore instance.
    """
    try:
        from core.security.keyring_store import KEYRING_AVAILABLE, KeyringStore

        if KEYRING_AVAILABLE:
            store = KeyringStore()
            # Smoke-test: try a no-op to verify the backend works
            store.list_keys()
            return store
    except Exception:
        pass

    from core.security.file_store import FileStore

    return FileStore(store_dir=store_dir)
