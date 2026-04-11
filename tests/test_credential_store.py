"""
Tests for the credential store module.

Security Issue #427: Secure credential storage tests
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest import mock

import pytest

# Mock keyring before import
mock_keyring = mock.MagicMock()
mock_keyring_available = True

with mock.patch.dict("sys.modules", {"keyring": mock_keyring}):
    from core.credential_store import (
        CredentialStore,
        CredentialError,
        get_credential_store,
        secure_store,
        secure_retrieve,
        secure_delete,
        KEYRING_AVAILABLE,
        CRYPTO_AVAILABLE,
    )


class TestCredentialStore:
    """Test cases for CredentialStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_keyring_store(self):
        """Mock keyring storage backend."""
        store = {}

        def set_password(service, key, value):
            store[(service, key)] = value

        def get_password(service, key):
            return store.get((service, key))

        def delete_password(service, key):
            if (service, key) in store:
                del store[(service, key)]

        mock_keyring.set_password = set_password
        mock_keyring.get_password = get_password
        mock_keyring.delete_password = delete_password
        mock_keyring.KeyringError = Exception
        mock_keyring.PasswordSetError = Exception

        return store

    @pytest.fixture
    def credential_store(self, temp_dir, mock_keyring_store):
        """Create a CredentialStore instance with mocked dependencies."""
        fallback_path = temp_dir / "credentials.enc"
        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=True,
            enable_fallback=True,
        )
        return store

    def test_initialization(self, temp_dir):
        """Test credential store initialization."""
        fallback_path = temp_dir / "credentials.enc"
        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
        )

        assert store.app_name == "test-aura"
        assert store.fallback_path == fallback_path
        assert store.enable_keyring == KEYRING_AVAILABLE
        assert store.enable_fallback == CRYPTO_AVAILABLE

    def test_is_sensitive_key(self, credential_store):
        """Test detection of sensitive keys."""
        # Known sensitive keys
        assert credential_store.is_sensitive_key("api_key")
        assert credential_store.is_sensitive_key("openai_api_key")
        assert credential_store.is_sensitive_key("anthropic_api_key")
        assert credential_store.is_sensitive_key("github_token")

        # Pattern-based detection
        assert credential_store.is_sensitive_key("my_service_api_key")
        assert credential_store.is_sensitive_key("aws_secret")
        assert credential_store.is_sensitive_key("db_password")
        assert credential_store.is_sensitive_key("auth_token")

        # Non-sensitive keys
        assert not credential_store.is_sensitive_key("model_name")
        assert not credential_store.is_sensitive_key("timeout")
        assert not credential_store.is_sensitive_key("enable_feature")

    def test_store_and_retrieve_keyring(self, credential_store, mock_keyring_store):
        """Test storing and retrieving via keyring."""
        # Skip if keyring not available
        if not credential_store.enable_keyring:
            pytest.skip("Keyring not available in this environment")

        # Store
        result = credential_store.store("api_key", "secret123")
        assert result is True

        # Verify in keyring
        assert mock_keyring_store[("test-aura", "api_key")] == "secret123"

        # Retrieve
        value = credential_store.retrieve("api_key")
        assert value == "secret123"

    def test_store_and_retrieve_fallback(self, temp_dir):
        """Test storing and retrieving via fallback file."""
        fallback_path = temp_dir / "credentials.enc"

        # Disable keyring to force fallback
        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        # Store
        result = store.store("api_key", "secret456")
        assert result is True

        # Verify fallback file exists
        assert fallback_path.exists()

        # Retrieve
        value = store.retrieve("api_key")
        assert value == "secret456"

    def test_retrieve_priority_environment(self, credential_store, mock_keyring_store, monkeypatch):
        """Test that environment variables take priority."""
        # Store in keyring
        credential_store.store("api_key", "keyring_value")

        # Set environment variable
        monkeypatch.setenv("AURA_API_KEY", "env_value")

        # Should get env value
        value = credential_store.retrieve("api_key")
        assert value == "env_value"

        # Clean up
        monkeypatch.delenv("AURA_API_KEY")

    def test_retrieve_priority_legacy_env(self, credential_store, mock_keyring_store, monkeypatch):
        """Test legacy environment variable names."""
        # Store in keyring
        credential_store.store("openai_api_key", "keyring_value")

        # Set legacy env variable (uppercase key name)
        monkeypatch.setenv("OPENAI_API_KEY", "legacy_env_value")

        # Should get legacy env value
        value = credential_store.retrieve("openai_api_key")
        assert value == "legacy_env_value"

        # Clean up
        monkeypatch.delenv("OPENAI_API_KEY")

    def test_delete(self, credential_store, mock_keyring_store):
        """Test deleting credentials."""
        # Store and verify
        credential_store.store("api_key", "to_delete")
        assert credential_store.retrieve("api_key") == "to_delete"

        # Delete
        result = credential_store.delete("api_key")
        assert result is True

        # Verify deleted
        assert credential_store.retrieve("api_key") is None

    def test_exists(self, credential_store, mock_keyring_store):
        """Test checking credential existence."""
        # Initially doesn't exist
        assert credential_store.exists("api_key") is False

        # Store
        credential_store.store("api_key", "exists")

        # Now exists
        assert credential_store.exists("api_key") is True

    def test_store_empty_value(self, credential_store):
        """Test that empty values are not stored."""
        result = credential_store.store("api_key", "")
        assert result is False

    def test_retrieve_not_found(self, credential_store):
        """Test retrieving non-existent key."""
        value = credential_store.retrieve("nonexistent_key")
        assert value is None

    def test_list_stored_keys_fallback(self, temp_dir):
        """Test listing keys in fallback storage."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        # Store multiple keys
        store.store("key1", "value1")
        store.store("key2", "value2")
        store.store("key3", "value3")

        # List keys
        keys = store.list_stored_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_get_storage_info(self, credential_store):
        """Test getting storage information."""
        info = credential_store.get_storage_info()

        assert info["app_name"] == "test-aura"
        assert "keyring_available" in info
        assert "fallback_available" in info
        assert "fallback_path" in info
        assert "fallback_exists" in info
        assert "stored_keys_count" in info

    def test_fallback_encryption_integrity(self, temp_dir):
        """Test that fallback encryption/decryption preserves data."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        # Store various types of values
        test_data = {
            "simple_key": "simple_value",
            "key_with_special_chars": "val!@#$%^&*()",
            "unicode_key": "unicode_value_ñ中文🚀",
            "long_key": "x" * 1000,
        }

        for key, value in test_data.items():
            store.store(key, value)

        # Retrieve and verify
        for key, expected_value in test_data.items():
            actual_value = store.retrieve(key)
            assert actual_value == expected_value, f"Mismatch for key: {key}"

    def test_keyring_failure_fallback(self, temp_dir, mock_keyring_store):
        """Test fallback when keyring fails."""
        fallback_path = temp_dir / "credentials.enc"

        # Make keyring raise an error
        def failing_set_password(service, key, value):
            raise Exception("Keyring locked")

        mock_keyring.set_password = failing_set_password

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=True,  # Try keyring first
            enable_fallback=True,  # Fallback to file
        )

        # Should fall back to file storage
        result = store.store("api_key", "fallback_value")
        assert result is True

        # Verify in fallback
        value = store.retrieve("api_key")
        assert value == "fallback_value"


class TestConvenienceFunctions:
    """Test convenience module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        get_credential_store.cache_clear()
        yield
        get_credential_store.cache_clear()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    def test_secure_store_and_retrieve(self, temp_dir, monkeypatch):
        """Test convenience store/retrieve functions."""
        # Mock the fallback path
        monkeypatch.setenv("AURA_CREDENTIALS_PATH", str(temp_dir / "creds.enc"))

        # Store
        result = secure_store("test_key", "test_value")
        assert result is True

        # Retrieve
        value = secure_retrieve("test_key")
        assert value == "test_value"

        # Delete
        secure_delete("test_key")

        # Verify deleted
        assert secure_retrieve("test_key") is None


class TestCredentialStoreIntegration:
    """Integration tests for credential store."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    def test_migration_scenario(self, temp_dir, monkeypatch):
        """Test a credential migration scenario."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        # Simulate reading from config file (plaintext)
        plaintext_config = {
            "api_key": "sk-openrouter-12345",
            "openai_api_key": "sk-openai-67890",
            "model_name": "gpt-4",  # Not sensitive
        }

        # Migrate sensitive keys to credential store
        migrated = {}
        for key, value in plaintext_config.items():
            if store.is_sensitive_key(key):
                if store.store(key, value):
                    migrated[key] = value

        # Verify migration
        assert len(migrated) == 2
        assert store.retrieve("api_key") == "sk-openrouter-12345"
        assert store.retrieve("openai_api_key") == "sk-openai-67890"

        # Simulate clearing from config
        for key in migrated:
            plaintext_config[key] = None

        # Config should now be safe to store
        assert plaintext_config["api_key"] is None
        assert plaintext_config["openai_api_key"] is None
        assert plaintext_config["model_name"] == "gpt-4"

    def test_multiple_keys_same_store(self, temp_dir):
        """Test storing multiple API keys in same store."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        # Store multiple provider keys
        providers = {
            "openrouter_api_key": "sk-or-123",
            "openai_api_key": "sk-oai-456",
            "anthropic_api_key": "sk-ant-789",
            "gemini_api_key": "sk-gem-abc",
        }

        for key, value in providers.items():
            store.store(key, value)

        # Verify all retrievable
        for key, expected in providers.items():
            actual = store.retrieve(key)
            assert actual == expected, f"Failed for {key}"


class TestCredentialStoreEdgeCases:
    """Edge case tests for credential store."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    def test_very_long_credential(self, temp_dir):
        """Test storing very long credentials."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        long_key = "k" * 100
        long_value = "v" * 10000

        store.store(long_key, long_value)
        retrieved = store.retrieve(long_key)

        assert retrieved == long_value

    def test_unicode_credentials(self, temp_dir):
        """Test storing unicode credentials."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        unicode_credentials = {
            "unicode_key_ñ": "value_中文",
            "emoji_key_🚀": "value_🎉",
            "mixed_key": "Hello 世界 Γειά σου",
        }

        for key, value in unicode_credentials.items():
            store.store(key, value)

        for key, expected in unicode_credentials.items():
            actual = store.retrieve(key)
            assert actual == expected

    def test_special_characters(self, temp_dir):
        """Test credentials with special characters."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        special_values = [
            'value"with"quotes',
            "value'with'apostrophes",
            "value\nwith\nnewlines",
            "value\twith\ttabs",
            "value\\with\\backslashes",
        ]

        for i, value in enumerate(special_values):
            key = f"special_{i}"
            store.store(key, value)
            retrieved = store.retrieve(key)
            assert retrieved == value, f"Failed for value: {repr(value)}"

    def test_concurrent_access_simulation(self, temp_dir):
        """Simulate concurrent access patterns."""
        fallback_path = temp_dir / "credentials.enc"

        store = CredentialStore(
            app_name="test-aura",
            fallback_path=fallback_path,
            enable_keyring=False,
            enable_fallback=True,
        )

        # Simulate multiple writes
        for i in range(10):
            store.store(f"key_{i}", f"value_{i}")

        # Simulate concurrent reads
        for i in range(10):
            value = store.retrieve(f"key_{i}")
            assert value == f"value_{i}"

        # Overwrite some keys
        for i in range(5):
            store.store(f"key_{i}", f"updated_value_{i}")

        # Verify updates
        for i in range(5):
            value = store.retrieve(f"key_{i}")
            assert value == f"updated_value_{i}"

        # Verify untouched keys
        for i in range(5, 10):
            value = store.retrieve(f"key_{i}")
            assert value == f"value_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
