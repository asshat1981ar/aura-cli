"""Tests for config manager."""

import json
import os
import pytest
from pathlib import Path

from aura.encryption.config_manager import EncryptedConfigManager
from aura.encryption.models import EncryptionConfig, EncryptionAlgorithm, KeySource


class TestEncryptedConfigManager:
    @pytest.fixture
    def key(self):
        return b"x" * 32

    @pytest.fixture
    def config_file(self, tmp_path, key, monkeypatch):
        monkeypatch.setenv("AURA_ENCRYPTION_KEY", key.decode())
        return tmp_path / "config.json"

    @pytest.fixture
    def manager(self, config_file):
        return EncryptedConfigManager(
            config_path=str(config_file),
            encryption_config=EncryptionConfig(
                algorithm=EncryptionAlgorithm.AES_GCM,
                key_source=KeySource.ENV_VAR,
            ),
        )

    def test_load_nonexistent_file(self, tmp_path):
        manager = EncryptedConfigManager(config_path=str(tmp_path / "new.json"))
        assert manager.load() == {}

    def test_save_and_load_unencrypted(self, manager):
        data = {"setting": "value", "number": 42}
        manager.save(data, encrypt=False)

        loaded = manager.load()
        assert loaded["setting"] == "value"
        assert loaded["number"] == 42

    def test_save_and_load_encrypted(self, manager):
        data = {"api_key": "secret123", "setting": "value"}
        manager.save(data, encrypt=True)

        # Check file has encrypted fields
        saved = json.loads(Path(manager.config_path).read_text())
        assert "api_key" in saved["encrypted_fields"]
        assert "ciphertext" in saved["data"]["api_key"]

        # Load and decrypt
        loaded = manager.load()
        assert loaded["api_key"] == "secret123"
        assert loaded["setting"] == "value"

    def test_get_value(self, manager):
        manager.save({"key": "value"}, encrypt=False)

        assert manager.get("key") == "value"
        assert manager.get("missing") is None
        assert manager.get("missing", "default") == "default"

    def test_set_value(self, manager):
        manager.save({"existing": "value"}, encrypt=False)
        manager.set("new_key", "new_value")

        assert manager.get("existing") == "value"
        assert manager.get("new_key") == "new_value"

    def test_set_auto_encrypts_sensitive(self, manager):
        manager.save({}, encrypt=False)
        manager.set("api_key", "secret123")

        # Check file has encrypted api_key
        saved = json.loads(Path(manager.config_path).read_text())
        assert "api_key" in saved["encrypted_fields"]

    def test_set_no_encrypt_for_non_sensitive(self, manager):
        manager.save({}, encrypt=False)
        manager.set("regular_setting", "value")

        # Check file has no encrypted fields
        saved = json.loads(Path(manager.config_path).read_text())
        assert saved["encrypted_fields"] == []

    def test_delete_key(self, manager):
        manager.save({"key1": "value1", "key2": "value2"}, encrypt=False)
        manager.delete("key1")

        assert manager.get("key1") is None
        assert manager.get("key2") == "value2"

    def test_is_sensitive(self, manager):
        assert manager._is_sensitive("api_key") is True
        assert manager._is_sensitive("password") is True
        assert manager._is_sensitive("SECRET_TOKEN") is True
        assert manager._is_sensitive("MY_AUTH_HEADER") is True
        assert manager._is_sensitive("regular_setting") is False

    def test_add_sensitive_key(self, manager):
        manager.add_sensitive_key("custom_secret")
        assert "custom_secret" in manager.sensitive_keys

    def test_remove_sensitive_key(self, manager):
        manager.remove_sensitive_key("api_key")
        assert "api_key" not in manager.sensitive_keys

    def test_load_uses_cache(self, manager):
        manager.save({"key": "value"}, encrypt=False)
        first_load = manager.load()

        # Modify file directly
        Path(manager.config_path).write_text(json.dumps({"key": "modified"}))

        # Should return cached value
        second_load = manager.load()
        assert second_load == first_load

    def test_rotate_key(self, manager, monkeypatch, tmp_path):
        """Test key rotation - new key must match what KeyManager would produce."""
        import hashlib

        old_key_str = "old_key_32bytes_for_testing!!"
        new_key_str = "new_key_32bytes_for_testing!!"

        # Keys after normalization (SHA-256)
        old_key_normalized = hashlib.sha256(old_key_str.encode()).digest()
        new_key_normalized = hashlib.sha256(new_key_str.encode()).digest()

        # Save with old key
        monkeypatch.setenv("AURA_ENCRYPTION_KEY", old_key_str)
        manager.save({"api_key": "secret123"}, encrypt=True)

        # Rotate to new key (pass normalized key)
        manager.rotate_key(new_key_normalized)

        # Create new manager with new key (simulating env var change)
        monkeypatch.setenv("AURA_ENCRYPTION_KEY", new_key_str)
        new_manager = EncryptedConfigManager(
            config_path=manager.config_path,
            encryption_config=EncryptionConfig(
                algorithm=EncryptionAlgorithm.AES_GCM,
                key_source=KeySource.ENV_VAR,
            ),
        )
        loaded = new_manager.load()
        assert loaded["api_key"] == "secret123"
