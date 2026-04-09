"""File-based credential store — fallback when system keyring is unavailable."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from core.security.credential_store import CredentialStore


class FileStore(CredentialStore):
    """Store credentials in a JSON file with restricted permissions.

    Used as a fallback when system keyring is not available.
    Credentials are stored in ``<config_dir>/credentials.json`` with
    mode 0600 (owner read/write only).
    """

    _FILENAME = "credentials.json"

    def __init__(self, config_dir: str | None = None) -> None:
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".aura")
        self._dir = Path(config_dir)
        self._path = self._dir / self._FILENAME

    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        with open(self._path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, str]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)

    def get(self, key: str) -> str | None:
        return self._read().get(key)

    def set(self, key: str, value: str) -> None:
        data = self._read()
        data[key] = value
        self._write(data)

    def delete(self, key: str) -> bool:
        data = self._read()
        if key not in data:
            return False
        del data[key]
        self._write(data)
        return True

    def list_keys(self) -> list[str]:
        return list(self._read().keys())
