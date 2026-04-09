"""File-based credential store — stores secrets as JSON on disk."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from core.security.credential_store import CredentialStore

# Key validation: alphanumeric, hyphens, underscores, dots; max 512 chars
_KEY_RE = re.compile(r"^[a-zA-Z0-9._-]{1,512}$")

# Characters that could enable path traversal or injection
_UNSAFE_CHARS = frozenset("/\\:\x00")

_STORE_FILENAME = "credentials.json"


class FileStore(CredentialStore):
    """Stores credentials in a JSON file within a config directory.

    Security properties:
    - Key names are validated against a strict allowlist pattern
    - Path traversal attempts are rejected
    - The backing file is a single JSON object (not one-file-per-key)
    - Corrupt backing files are handled gracefully
    """

    def __init__(self, config_dir: str) -> None:
        self._config_dir = Path(config_dir)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self._config_dir / _STORE_FILENAME

    def _validate_key(self, key: str) -> None:
        """Reject keys that could cause filesystem or injection issues."""
        if not key:
            raise ValueError("Key must not be empty")
        if any(c in key for c in _UNSAFE_CHARS):
            raise ValueError(f"Key contains unsafe characters: {key!r}")
        if ".." in key:
            raise ValueError(f"Key contains path traversal sequence: {key!r}")
        if len(key) > 512:
            raise ValueError(f"Key too long ({len(key)} > 512)")

    def _read_store(self) -> dict[str, str]:
        """Read the backing JSON file. Returns empty dict on corruption."""
        if not self._store_path.exists():
            return {}
        try:
            data = self._store_path.read_text(encoding="utf-8")
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                return {}
            return {k: v for k, v in parsed.items() if isinstance(v, str)}
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return {}

    def _write_store(self, data: dict[str, str]) -> None:
        """Atomically write the store file."""
        tmp_path = self._store_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(self._store_path)
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def get(self, key: str) -> Optional[str]:
        self._validate_key(key)
        return self._read_store().get(key)

    def set(self, key: str, value: str) -> None:
        self._validate_key(key)
        if not isinstance(value, str):
            raise TypeError(f"Value must be a string, got {type(value).__name__}")
        store = self._read_store()
        store[key] = value
        self._write_store(store)

    def delete(self, key: str) -> bool:
        self._validate_key(key)
        store = self._read_store()
        if key not in store:
            return False
        del store[key]
        self._write_store(store)
        return True

    def list_keys(self) -> list[str]:
        return list(self._read_store().keys())
