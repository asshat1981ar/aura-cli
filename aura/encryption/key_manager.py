"""Encryption key management."""

import base64
import getpass
import os
from pathlib import Path
from typing import Optional

from .models import EncryptionConfig, KeySource


class KeyManager:
    """Manage encryption keys from various sources."""

    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()

    def get_key(self) -> bytes:
        """Get the encryption key from configured source."""
        if self.config.key_source == KeySource.ENV_VAR:
            return self._get_from_env()
        elif self.config.key_source == KeySource.KEYCHAIN:
            return self._get_from_keychain()
        elif self.config.key_source == KeySource.FILE:
            return self._get_from_file()
        elif self.config.key_source == KeySource.PROMPT:
            return self._get_from_prompt()
        else:
            raise ValueError(f"Unknown key source: {self.config.key_source}")

    def _get_from_env(self) -> bytes:
        """Get key from environment variable."""
        key = os.environ.get(self.config.key_env_var)
        if not key:
            raise KeyError(f"Encryption key not found in environment variable {self.config.key_env_var}")
        return self._normalize_key(key)

    def _get_from_keychain(self) -> bytes:
        """Get key from system keychain."""
        try:
            import keyring

            key = keyring.get_password(
                self.config.keychain_service or "aura-cli",
                self.config.keychain_account or "default",
            )
            if not key:
                raise KeyError("Key not found in keychain")
            return self._normalize_key(key)
        except ImportError:
            raise RuntimeError("keyring package not installed")

    def _get_from_file(self) -> bytes:
        """Get key from file."""
        if not self.config.key_file:
            raise ValueError("key_file not configured")

        path = Path(self.config.key_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Key file not found: {path}")

        key = path.read_text().strip()
        return self._normalize_key(key)

    def _get_from_prompt(self) -> bytes:
        """Get key from user prompt."""
        key = getpass.getpass("Enter encryption key: ")
        if not key:
            raise ValueError("No key provided")
        return self._normalize_key(key)

    def _normalize_key(self, key: str) -> bytes:
        """Normalize key to 32 bytes for AES-256."""
        key_bytes = key.encode() if isinstance(key, str) else key

        # If key is base64 encoded, decode it
        try:
            decoded = base64.b64decode(key_bytes)
            if len(decoded) == 32:
                return decoded
        except Exception:
            pass

        # Otherwise, use SHA-256 to derive 32-byte key
        import hashlib

        return hashlib.sha256(key_bytes).digest()

    def generate_key(self) -> str:
        """Generate a new random encryption key."""
        import secrets

        key = secrets.token_bytes(32)
        return base64.b64encode(key).decode()

    def save_key_to_file(self, key: str, path: str, mode: int = 0o600):
        """Save key to file with restricted permissions."""
        path_obj = Path(path).expanduser()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(key)
        os.chmod(path_obj, mode)
