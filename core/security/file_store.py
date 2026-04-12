"""Fernet-encrypted file-backed credential store.

Falls back to this when the OS keyring is unavailable. Credentials are
encrypted with a Fernet key derived from a master password or stored
in a key file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cryptography.fernet import Fernet

from core.security.credential_store import CredentialStore

_DEFAULT_DIR = Path.home() / ".aura" / "credentials"


class FileStore(CredentialStore):
    """Credential store backed by Fernet-encrypted JSON files."""

    def __init__(
        self,
        store_dir: Path | str | None = None,
        fernet_key: bytes | None = None,
    ) -> None:
        self._dir = Path(store_dir) if store_dir else _DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        key_file = self._dir / ".key"
        if fernet_key:
            self._fernet = Fernet(fernet_key)
            if not key_file.exists():
                key_file.write_bytes(fernet_key)
                os.chmod(key_file, 0o600)
        elif key_file.exists():
            self._fernet = Fernet(key_file.read_bytes().strip())
        else:
            new_key = Fernet.generate_key()
            key_file.write_bytes(new_key)
            os.chmod(key_file, 0o600)
            self._fernet = Fernet(new_key)

        self._store_file = self._dir / "store.enc"
        self._data: dict[str, str] = self._load()

    def set(self, name: str, secret: str) -> None:
        if not name:
            raise ValueError("Credential name must not be empty")
        self._data[name] = secret
        self._save()

    def get(self, name: str) -> str | None:
        return self._data.get(name)

    def delete(self, name: str) -> None:
        self._data.pop(name, None)
        self._save()

    def list_keys(self) -> list[str]:
        return list(self._data.keys())

    def _load(self) -> dict[str, str]:
        if not self._store_file.exists():
            return {}
        encrypted = self._store_file.read_bytes()
        if not encrypted:
            return {}
        decrypted = self._fernet.decrypt(encrypted)
        return json.loads(decrypted.decode("utf-8"))

    def _save(self) -> None:
        raw = json.dumps(self._data).encode("utf-8")
        encrypted = self._fernet.encrypt(raw)
        self._store_file.write_bytes(encrypted)
