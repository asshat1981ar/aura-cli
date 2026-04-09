"""Property-based tests for CredentialStore implementations.

Uses Hypothesis to verify algebraic properties of credential stores:
round-trip, overwrite, delete, non-interference, and key listing.

Inspired by Vikram et al. (2024) "Can LLMs Write Good Property-Based Tests?"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from core.security.file_store import FileStore

# Strategy for valid credential names: non-empty, no surrogate chars
_name_st = st.text(
    min_size=1,
    max_size=128,
    alphabet=st.characters(blacklist_categories=("Cs",)),
)

# Strategy for credential secrets: non-empty strings
_secret_st = st.text(min_size=1, max_size=4096)

# Strategy for Unicode secrets including emoji, CJK, RTL
_unicode_secret_st = st.text(
    min_size=1,
    max_size=512,
    alphabet=st.characters(
        min_codepoint=0,
        max_codepoint=0x10FFFF,
        blacklist_categories=("Cs",),
    ),
)


def _make_store() -> FileStore:
    """Create a fresh FileStore in a temporary directory."""
    tmp = tempfile.mkdtemp()
    return FileStore(store_dir=Path(tmp) / "creds")


# ---------------------------------------------------------------------------
# Property 1: Round-trip — set then get returns the original secret
# ---------------------------------------------------------------------------
class TestRoundTrip:
    @given(name=_name_st, secret=_secret_st)
    def test_set_get_roundtrip(self, name, secret):
        store = _make_store()
        store.set(name, secret)
        assert store.get(name) == secret


# ---------------------------------------------------------------------------
# Property 2: Delete removes entry
# ---------------------------------------------------------------------------
class TestDeleteRemoves:
    @given(name=_name_st, secret=_secret_st)
    def test_delete_removes_entry(self, name, secret):
        store = _make_store()
        store.set(name, secret)
        store.delete(name)
        assert store.get(name) is None


# ---------------------------------------------------------------------------
# Property 3: Overwrite — latest write wins
# ---------------------------------------------------------------------------
class TestOverwrite:
    @given(name=_name_st, secret1=_secret_st, secret2=_secret_st)
    def test_latest_write_wins(self, name, secret1, secret2):
        store = _make_store()
        store.set(name, secret1)
        store.set(name, secret2)
        assert store.get(name) == secret2


# ---------------------------------------------------------------------------
# Property 4: Non-interference — setting key A does not affect key B
# ---------------------------------------------------------------------------
class TestNonInterference:
    @given(
        name_a=_name_st,
        name_b=_name_st,
        secret_a=_secret_st,
        secret_b=_secret_st,
    )
    def test_setting_a_does_not_affect_b(self, name_a, name_b, secret_a, secret_b):
        assume(name_a != name_b)
        store = _make_store()
        store.set(name_a, secret_a)
        store.set(name_b, secret_b)
        # Re-setting A should not change B
        store.set(name_a, "overwritten")
        assert store.get(name_b) == secret_b


# ---------------------------------------------------------------------------
# Property 5: Empty secret handling — store.set(name, "") stores empty string
# ---------------------------------------------------------------------------
class TestEmptySecret:
    @given(name=_name_st)
    def test_empty_secret_stores_empty_string(self, name):
        store = _make_store()
        store.set(name, "")
        assert store.get(name) == ""


# ---------------------------------------------------------------------------
# Property 6: Unicode secrets survive round-trip through Fernet encryption
# ---------------------------------------------------------------------------
class TestUnicodeRoundTrip:
    @given(name=_name_st, secret=_unicode_secret_st)
    def test_unicode_secrets_roundtrip(self, name, secret):
        store = _make_store()
        store.set(name, secret)
        assert store.get(name) == secret


# ---------------------------------------------------------------------------
# Property 7: list_keys reflects all stored keys
# ---------------------------------------------------------------------------
class TestListKeys:
    @given(
        keys=st.lists(
            _name_st,
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
    def test_list_reflects_stored_keys(self, keys):
        store = _make_store()
        for k in keys:
            store.set(k, "value")
        stored = set(store.list_keys())
        assert set(keys).issubset(stored)


# ---------------------------------------------------------------------------
# Property 8: Idempotent delete — deleting a non-existent key does not raise
# ---------------------------------------------------------------------------
class TestIdempotentDelete:
    @given(name=_name_st)
    def test_delete_nonexistent_does_not_raise(self, name):
        store = _make_store()
        # Ensure key does not exist, then delete
        store.delete(name)
        # Deleting again should also not raise
        store.delete(name)
