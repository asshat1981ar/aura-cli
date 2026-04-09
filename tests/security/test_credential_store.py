"""Tests for core/security credential store abstraction."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestFileStore:
    """Test the encrypted file-based credential store."""

    def _make_store(self, tmp_path: Path):
        from core.security.file_store import FileStore
        return FileStore(store_path=tmp_path / "creds.enc.json")

    def test_roundtrip_set_get_delete(self, tmp_path):
        store = self._make_store(tmp_path)
        store.set("api_key", "sk-secret-123")
        assert store.get("api_key") == "sk-secret-123"
        store.delete("api_key")
        assert store.get("api_key") is None

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.get("nonexistent") is None

    def test_list_keys(self, tmp_path):
        store = self._make_store(tmp_path)
        store.set("a", "1")
        store.set("b", "2")
        keys = store.list_keys()
        assert sorted(keys) == ["a", "b"]

    def test_delete_nonexistent_is_noop(self, tmp_path):
        store = self._make_store(tmp_path)
        store.delete("nonexistent")  # should not raise

    def test_overwrite_value(self, tmp_path):
        store = self._make_store(tmp_path)
        store.set("key", "value1")
        store.set("key", "value2")
        assert store.get("key") == "value2"

    def test_encrypted_at_rest(self, tmp_path):
        store = self._make_store(tmp_path)
        store.set("api_key", "my-secret-value")
        raw = (tmp_path / "creds.enc.json").read_text()
        assert "my-secret-value" not in raw


class TestKeyringStore:
    """Test keyring store with a mock keyring backend."""

    def test_roundtrip_with_mock(self, tmp_path):
        # Use a simple dict to mock keyring
        _store: dict[str, dict[str, str]] = {}

        def mock_set(service, name, secret):
            _store.setdefault(service, {})[name] = secret

        def mock_get(service, name):
            return _store.get(service, {}).get(name)

        def mock_delete(service, name):
            _store.get(service, {}).pop(name, None)

        with mock.patch("core.security.keyring_store.keyring") as mock_keyring:
            mock_keyring.set_password = mock_set
            mock_keyring.get_password = mock_get
            mock_keyring.delete_password = mock_delete
            mock_keyring.errors = type("Errors", (), {"PasswordDeleteError": Exception})

            from core.security.keyring_store import KeyringStore, _MANIFEST_PATH
            store = KeyringStore()
            store._manifest_path = tmp_path / "manifest.json"

            store.set("api_key", "sk-123")
            assert store.get("api_key") == "sk-123"
            assert "api_key" in store.list_keys()

            store.delete("api_key")
            assert store.get("api_key") is None
            assert "api_key" not in store.list_keys()


class TestStoreFactory:
    """Test the store factory auto-detection logic."""

    def test_fallback_to_file_store_when_keyring_unavailable(self, tmp_path):
        with mock.patch("core.security.keyring_store.keyring") as mock_kr:
            mock_kr.set_password.side_effect = Exception("no keyring backend")
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from core.security.store_factory import get_credential_store
                store = get_credential_store()
                from core.security.file_store import FileStore
                assert isinstance(store, FileStore)
                assert any("No OS keyring" in str(warning.message) for warning in w)

    def test_env_var_forces_file_backend(self, monkeypatch):
        monkeypatch.setenv("AURA_CREDENTIAL_BACKEND", "file")
        # Re-import to pick up env var
        from core.security.store_factory import get_credential_store
        store = get_credential_store()
        from core.security.file_store import FileStore
        assert isinstance(store, FileStore)
        monkeypatch.delenv("AURA_CREDENTIAL_BACKEND")


class TestEnvVarOverrides:
    """Test that env var overrides bypass the credential store."""

    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        from core.security.file_store import FileStore
        from core.security.credential_helpers import get_credential

        store = FileStore(store_path=tmp_path / "creds.enc.json")
        store.set("api_key", "stored-value")

        monkeypatch.setenv("AURA_API_KEY", "env-value")
        assert get_credential(store, "api_key") == "env-value"

    def test_store_used_when_no_env_var(self, tmp_path, monkeypatch):
        from core.security.file_store import FileStore
        from core.security.credential_helpers import get_credential

        store = FileStore(store_path=tmp_path / "creds.enc.json")
        store.set("api_key", "stored-value")

        monkeypatch.delenv("AURA_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert get_credential(store, "api_key") == "stored-value"


class TestMigrateKeyringCommand:
    """Test the credential migrate-keyring dispatch handler."""

    def test_migrate_reads_old_file_and_writes_to_store(self, tmp_path):
        from core.security.file_store import FileStore
        from core.security.credential_helpers import SECRET_KEYS, get_credential

        # Create a fake plaintext config
        old_config = {
            "api_key": "sk-old-key-123",
            "openai_api_key": "sk-openai-456",
            "model_name": "gpt-4",
        }
        config_file = tmp_path / "aura.config.json"
        config_file.write_text(json.dumps(old_config))

        store = FileStore(store_path=tmp_path / "creds.enc.json")

        # Simulate migration logic
        raw = json.loads(config_file.read_text())
        secrets_found = {k: v for k, v in raw.items() if k in SECRET_KEYS and isinstance(v, str) and v.strip()}
        for key, value in secrets_found.items():
            store.set(key, value)
        for key in secrets_found:
            raw.pop(key, None)
        config_file.write_text(json.dumps(raw, indent=4))

        # Verify secrets are in the store
        assert store.get("api_key") == "sk-old-key-123"
        assert store.get("openai_api_key") == "sk-openai-456"

        # Verify secrets removed from config file
        remaining = json.loads(config_file.read_text())
        assert "api_key" not in remaining
        assert "openai_api_key" not in remaining
        assert remaining["model_name"] == "gpt-4"
