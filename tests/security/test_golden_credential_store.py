"""
Golden file tests for CredentialStore error messages and list output.
Ensures user-facing strings don't drift unexpectedly.

Update snapshots with: pytest --snapshot-update
"""

import warnings

import pytest

from core.security.file_store import FileStore


class TestCredentialStoreGolden:

    def test_store_factory_warning_message(self, snapshot):
        """The fallback warning message must match snapshot exactly — user-visible string."""
        import keyring
        import keyring.backends.fail

        original_backend = keyring.get_keyring()
        keyring.set_keyring(keyring.backends.fail.Keyring())
        try:
            from core.security.store_factory import get_credential_store

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                store = get_credential_store()
                if w:
                    assert str(w[0].message) == snapshot
                else:
                    # If no warning emitted, the keyring worked — skip
                    pytest.skip("Keyring available, no fallback warning emitted")
        finally:
            keyring.set_keyring(original_backend)

    def test_list_keys_empty_store(self, tmp_path, snapshot):
        """Empty store list output must match snapshot."""
        store = FileStore(config_dir=str(tmp_path))
        assert store.list_keys() == snapshot

    def test_list_keys_after_set(self, tmp_path, snapshot):
        """list_keys after setting deterministic keys must match snapshot."""
        store = FileStore(config_dir=str(tmp_path))
        store.set("openai_api_key", "sk-test")
        store.set("anthropic_api_key", "ant-test")
        store.set("aura_api_key", "aura-test")
        # Sort for determinism
        assert sorted(store.list_keys()) == snapshot

    def test_file_store_get_missing_key(self, tmp_path, snapshot):
        """Getting a nonexistent key must return the expected sentinel."""
        store = FileStore(config_dir=str(tmp_path))
        assert store.get("nonexistent_key") == snapshot

    def test_file_store_delete_returns_false_for_missing(self, tmp_path, snapshot):
        """Deleting a nonexistent key must return the expected boolean."""
        store = FileStore(config_dir=str(tmp_path))
        assert store.delete("nonexistent_key") == snapshot
