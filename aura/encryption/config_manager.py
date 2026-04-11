"""Configuration manager with field-level encryption."""

import json
from pathlib import Path
from typing import Any, Optional, Set

from .crypto import CryptoEngine
from .key_manager import KeyManager
from .models import EncryptedConfig, EncryptedValue, EncryptionConfig


class EncryptedConfigManager:
    """Manage configuration with encrypted sensitive fields."""

    DEFAULT_SENSITIVE_KEYS = {
        "password",
        "secret",
        "token",
        "key",
        "api_key",
        "private_key",
        "credentials",
        "auth",
    }

    def __init__(
        self,
        config_path: str,
        encryption_config: Optional[EncryptionConfig] = None,
        sensitive_keys: Optional[Set[str]] = None,
    ):
        self.config_path = Path(config_path)
        self.encryption_config = encryption_config or EncryptionConfig()
        self.sensitive_keys = sensitive_keys or self.DEFAULT_SENSITIVE_KEYS.copy()
        self._key_manager = KeyManager(self.encryption_config)
        self._cache: Optional[dict] = None

    def load(self) -> dict:
        """Load and decrypt configuration."""
        if self._cache is not None:
            return self._cache

        if not self.config_path.exists():
            return {}

        try:
            data = json.loads(self.config_path.read_text())
            encrypted_config = EncryptedConfig.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

        # Decrypt fields if encryption is configured
        if encrypted_config.encrypted_fields:
            key = self._key_manager.get_key()
            engine = CryptoEngine(key, self.encryption_config.algorithm)

            for field in encrypted_config.encrypted_fields:
                if field in encrypted_config.data:
                    encrypted_value = EncryptedValue.from_dict(encrypted_config.data[field])
                    encrypted_config.data[field] = engine.decrypt(encrypted_value)

        self._cache = encrypted_config.data
        return self._cache

    def save(self, data: dict, encrypt: bool = True):
        """Save configuration with optional encryption."""
        encrypted_config = EncryptedConfig(data=data.copy())

        if encrypt:
            key = self._key_manager.get_key()
            engine = CryptoEngine(key, self.encryption_config.algorithm)

            for key_name in list(data.keys()):
                if self._is_sensitive(key_name):
                    encrypted = engine.encrypt(str(data[key_name]))
                    encrypted_config.data[key_name] = encrypted.to_dict()
                    encrypted_config.encrypted_fields.add(key_name)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(encrypted_config.to_dict(), indent=2))
        self._cache = data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        config = self.load()
        return config.get(key, default)

    def set(self, key: str, value: Any, encrypt: Optional[bool] = None):
        """Set a configuration value."""
        config = self.load()
        config[key] = value

        # Auto-encrypt if key looks sensitive
        should_encrypt = encrypt if encrypt is not None else self._is_sensitive(key)
        self.save(config, encrypt=should_encrypt)

    def delete(self, key: str):
        """Delete a configuration key."""
        config = self.load()
        if key in config:
            del config[key]
            self.save(config)

    def rotate_key(self, new_key: bytes, backup: bool = True):
        """Rotate encryption key.

        Args:
            new_key: The new encryption key (32 bytes)
            backup: If True, create a backup of the original config file

        Raises:
            ValueError: If the new_key is not 32 bytes
            RuntimeError: If key rotation fails
        """
        if len(new_key) != 32:
            raise ValueError("New key must be exactly 32 bytes")

        # Create backup if requested
        if backup and self.config_path.exists():
            backup_path = self.config_path.with_suffix(".json.backup")
            backup_path.write_text(self.config_path.read_text())

        try:
            # Load with old key
            old_key = self._key_manager.get_key()
            old_engine = CryptoEngine(old_key, self.encryption_config.algorithm)

            data = json.loads(self.config_path.read_text())
            encrypted_config = EncryptedConfig.from_dict(data)

            # Decrypt all fields
            decrypted_data = encrypted_config.data.copy()
            for field in encrypted_config.encrypted_fields:
                if field in decrypted_data:
                    encrypted_value = EncryptedValue.from_dict(decrypted_data[field])
                    decrypted_data[field] = old_engine.decrypt(encrypted_value)

            # Re-encrypt with new key
            new_engine = CryptoEngine(new_key, self.encryption_config.algorithm)
            new_config = EncryptedConfig(data=decrypted_data)

            for field in encrypted_config.encrypted_fields:
                if field in decrypted_data:
                    encrypted = new_engine.encrypt(str(decrypted_data[field]))
                    new_config.data[field] = encrypted.to_dict()
                    new_config.encrypted_fields.add(field)

            self.config_path.write_text(json.dumps(new_config.to_dict(), indent=2))
            self._cache = None
        except Exception as e:
            raise RuntimeError(f"Key rotation failed: {e}")

    def _is_sensitive(self, key: str) -> bool:
        """Check if a key should be encrypted."""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.sensitive_keys)

    def add_sensitive_key(self, key: str):
        """Add a key pattern to the sensitive list."""
        self.sensitive_keys.add(key.lower())

    def remove_sensitive_key(self, key: str):
        """Remove a key pattern from the sensitive list."""
        self.sensitive_keys.discard(key.lower())
