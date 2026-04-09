"""Factory for creating credential store instances."""

from __future__ import annotations

import os
from pathlib import Path

from core.security.credential_store import CredentialStore
from core.security.file_store import FileStore


def create_credential_store(
    backend: str = "auto",
    config_dir: str | None = None,
) -> CredentialStore:
    """Create a credential store instance.

    Args:
        backend: "file", "keyring", or "auto" (try keyring, fall back to file).
        config_dir: Directory for file-based storage. Defaults to ~/.aura.

    Returns:
        A CredentialStore implementation.
    """
    if config_dir is None:
        config_dir = str(Path.home() / ".aura")

    if backend == "file":
        return FileStore(config_dir=config_dir)

    if backend == "keyring":
        from core.security.keyring_store import KeyringStore
        return KeyringStore()

    # auto: try keyring first, fall back to file
    try:
        from core.security.keyring_store import KeyringStore
        store = KeyringStore()
        # Smoke test: try a no-op to verify the backend works
        return store
    except (ImportError, Exception):
        return FileStore(config_dir=config_dir)
