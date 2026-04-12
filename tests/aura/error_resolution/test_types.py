"""Tests for aura/error_resolution/types.py — CacheKey, ResolutionResult."""

import pytest
from aura.error_resolution.types import CacheKey, ResolutionConfidence, ResolutionResult


# ---------------------------------------------------------------------------
# ResolutionConfidence
# ---------------------------------------------------------------------------

class TestResolutionConfidence:
    def test_values(self):
        assert ResolutionConfidence.HIGH == "high"
        assert ResolutionConfidence.MEDIUM == "medium"
        assert ResolutionConfidence.LOW == "low"

    def test_is_string_enum(self):
        assert isinstance(ResolutionConfidence.HIGH, str)


# ---------------------------------------------------------------------------
# CacheKey
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_str_format(self):
        k = CacheKey("ValueError", "bad value", "abc123")
        assert str(k) == "ValueError:bad value:abc123"

    def test_hash_consistent(self):
        k = CacheKey("TypeError", "msg", "hash")
        assert hash(k) == hash(k)

    def test_equal_keys(self):
        k1 = CacheKey("T", "m", "h")
        k2 = CacheKey("T", "m", "h")
        assert k1 == k2

    def test_unequal_keys(self):
        k1 = CacheKey("T", "m", "h1")
        k2 = CacheKey("T", "m", "h2")
        assert k1 != k2

    def test_not_equal_to_non_cache_key(self):
        k = CacheKey("T", "m", "h")
        result = k.__eq__("not a cache key")
        assert result is NotImplemented

    def test_hashable_as_dict_key(self):
        k = CacheKey("T", "m", "h")
        d = {k: "value"}
        assert d[k] == "value"

    def test_hash_differs_for_different_keys(self):
        k1 = CacheKey("T", "m1", "h")
        k2 = CacheKey("T", "m2", "h")
        assert hash(k1) != hash(k2)


# ---------------------------------------------------------------------------
# ResolutionResult
# ---------------------------------------------------------------------------

def _make_result(**kwargs) -> ResolutionResult:
    defaults = dict(
        original_error="ImportError: no module named foo",
        explanation="Module foo is not installed",
        suggested_fix="pip install foo",
        confidence=ResolutionConfidence.HIGH,
        auto_applied=False,
        cache_hit=False,
        provider="gpt-4",
        execution_time_ms=120,
    )
    defaults.update(kwargs)
    return ResolutionResult(**defaults)


class TestResolutionResultToDict:
    def test_to_dict_keys(self):
        r = _make_result()
        d = r.to_dict()
        for key in ("original_error", "explanation", "suggested_fix", "confidence",
                    "auto_applied", "cache_hit", "provider", "execution_time_ms", "metadata"):
            assert key in d

    def test_confidence_serialized_as_string(self):
        r = _make_result(confidence=ResolutionConfidence.MEDIUM)
        assert r.to_dict()["confidence"] == "medium"

    def test_auto_applied_preserved(self):
        r = _make_result(auto_applied=True)
        assert r.to_dict()["auto_applied"] is True

    def test_metadata_included(self):
        r = _make_result(metadata={"model": "gpt-4", "tokens": 500})
        assert r.to_dict()["metadata"]["model"] == "gpt-4"

    def test_empty_metadata_default(self):
        r = _make_result()
        assert r.to_dict()["metadata"] == {}


class TestResolutionResultFromDict:
    def test_roundtrip(self):
        original = _make_result(confidence=ResolutionConfidence.LOW, cache_hit=True)
        d = original.to_dict()
        restored = ResolutionResult.from_dict(d)
        assert restored.original_error == original.original_error
        assert restored.confidence == ResolutionConfidence.LOW
        assert restored.cache_hit is True
        assert restored.provider == original.provider

    def test_missing_metadata_defaults_empty(self):
        d = _make_result().to_dict()
        del d["metadata"]
        r = ResolutionResult.from_dict(d)
        assert r.metadata == {}

    def test_confidence_enum_restored(self):
        d = _make_result(confidence=ResolutionConfidence.HIGH).to_dict()
        r = ResolutionResult.from_dict(d)
        assert isinstance(r.confidence, ResolutionConfidence)
        assert r.confidence == ResolutionConfidence.HIGH
