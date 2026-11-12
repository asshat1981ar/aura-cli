"""
Fuzz tests for CredentialStore implementations.
Uses Hypothesis for property-based fuzzing of arbitrary byte inputs.

Run with: pytest tests/security/fuzz_credential_store.py -v
Coverage-guided fuzzing: python -m pytest tests/security/fuzz_credential_store.py --hypothesis-seed=random
"""

import glob
import tempfile

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from core.security.file_store import FileStore


# Strategy for keys that pass FileStore's allowlist validation
_safe_key_st = st.text(
    min_size=1,
    max_size=256,
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="-_.",
    ),
)


class TestFileStoreFuzz:

    @given(
        name=_safe_key_st,
        secret=st.binary(min_size=0, max_size=65536),
    )
    @settings(
        max_examples=500,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_arbitrary_binary_secret_no_crash(self, name, secret):
        """FileStore must not crash on arbitrary binary input — decode as utf-8 or handle gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(config_dir=tmpdir)
            try:
                store.set(name, secret.decode("utf-8", errors="replace"))
            except (ValueError, TypeError):
                pass  # Documented rejection is acceptable; crash/panic is not
            # The store itself must still be readable after the attempt
            store.list_keys()  # Must not raise

    @given(name=st.text(min_size=1, max_size=2048))
    @settings(
        max_examples=500,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_arbitrary_key_name_no_crash(self, name):
        """Key names with special chars (path traversal, null bytes, unicode) must not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(config_dir=tmpdir)
            try:
                store.set(name, "safe_secret")
                result = store.get(name)
                assert result == "safe_secret" or result is None
            except (ValueError, OSError):
                pass  # Documented rejection; no crash

    @given(data=st.binary(min_size=1, max_size=4096))
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_corrupted_store_file_no_crash(self, data):
        """FileStore must not crash if its backing file is corrupted/truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(config_dir=tmpdir)
            store.set("key", "value")
            # Corrupt the backing file
            files = glob.glob(f"{tmpdir}/*.json") + glob.glob(f"{tmpdir}/*.enc")
            for f in files:
                with open(f, "wb") as fp:
                    fp.write(data)
            # Must handle corruption gracefully
            try:
                result = store.get("key")
                assert result is None or isinstance(result, str)
            except (ValueError, OSError):
                pass  # Any documented exception is acceptable
            # list_keys must also survive
            store.list_keys()

    @given(
        key=_safe_key_st,
        value=st.text(min_size=0, max_size=4096),
    )
    @settings(
        max_examples=500,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_roundtrip_integrity(self, key, value):
        """Values stored must be retrievable unchanged (roundtrip property)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(config_dir=tmpdir)
            try:
                store.set(key, value)
                retrieved = store.get(key)
                assert retrieved == value, f"Roundtrip failed: stored {value!r}, got {retrieved!r}"
            except (ValueError, TypeError):
                pass  # Key rejected by validation is acceptable

    @given(
        keys=st.lists(
            _safe_key_st,
            min_size=1,
            max_size=50,
            unique=True,
        )
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_multiple_keys_isolation(self, keys):
        """Setting multiple keys must not cause cross-contamination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(config_dir=tmpdir)
            for i, key in enumerate(keys):
                try:
                    store.set(key, f"value_{i}")
                except (ValueError, TypeError):
                    pass

            for i, key in enumerate(keys):
                try:
                    result = store.get(key)
                    if result is not None:
                        assert result == f"value_{i}"
                except (ValueError, OSError):
                    pass

    @given(key=_safe_key_st)
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_delete_then_get_returns_none(self, key):
        """After deletion, get must return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(config_dir=tmpdir)
            try:
                store.set(key, "to_delete")
                store.delete(key)
                assert store.get(key) is None
            except (ValueError, TypeError):
                pass
