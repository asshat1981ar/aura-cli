"""Unit tests for credential store implementations."""

from __future__ import annotations

import pytest

from core.security.file_store import FileStore


@pytest.fixture()
def file_store(tmp_path):
    """Create a FileStore backed by a temporary directory."""
    return FileStore(store_dir=tmp_path / "creds")


class TestFileStore:
    """Tests for FileStore (Fernet-encrypted file backend)."""

    def test_set_and_get(self, file_store):
        file_store.set("token", "secret123")
        assert file_store.get("token") == "secret123"

    def test_get_missing_returns_none(self, file_store):
        assert file_store.get("nonexistent") is None

    def test_delete_existing(self, file_store):
        file_store.set("token", "secret123")
        file_store.delete("token")
        assert file_store.get("token") is None

    def test_delete_nonexistent_does_not_raise(self, file_store):
        file_store.delete("nonexistent")  # should not raise

    def test_list_keys_empty(self, file_store):
        assert file_store.list_keys() == []

    def test_list_keys_after_set(self, file_store):
        file_store.set("a", "1")
        file_store.set("b", "2")
        assert sorted(file_store.list_keys()) == ["a", "b"]

    def test_overwrite(self, file_store):
        file_store.set("token", "old")
        file_store.set("token", "new")
        assert file_store.get("token") == "new"

    def test_empty_name_raises(self, file_store):
        with pytest.raises(ValueError, match="empty"):
            file_store.set("", "secret")

    def test_persistence_across_instances(self, tmp_path):
        store_dir = tmp_path / "creds"
        store1 = FileStore(store_dir=store_dir)
        store1.set("token", "persisted")

        store2 = FileStore(store_dir=store_dir)
        assert store2.get("token") == "persisted"

    def test_unicode_secret(self, file_store):
        file_store.set("emoji", "Hello 🌍🎉")
        assert file_store.get("emoji") == "Hello 🌍🎉"
