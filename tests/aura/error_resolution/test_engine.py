"""Tests for aura/error_resolution/engine.py — ErrorResolutionEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aura.error_resolution.engine import (
    ErrorResolutionEngine,
    _extract_module_name,
    _parse_provider_response,
)
from aura.error_resolution.cache import FourLayerCache
from aura.error_resolution.types import ResolutionConfidence


# ---------------------------------------------------------------------------
# _extract_module_name
# ---------------------------------------------------------------------------

class TestExtractModuleName:
    def test_extracts_simple_module(self):
        err = ModuleNotFoundError("No module named 'requests'")
        assert _extract_module_name(err) == "requests"

    def test_extracts_nested_module(self):
        err = ModuleNotFoundError("No module named 'some.nested.package'")
        assert _extract_module_name(err) == "some.nested.package"

    def test_fallback_when_no_match(self):
        err = Exception("some other error")
        assert _extract_module_name(err) == "the_package"


# ---------------------------------------------------------------------------
# _parse_provider_response
# ---------------------------------------------------------------------------

class TestParseProviderResponse:
    def test_parses_all_fields(self):
        raw = (
            "EXPLANATION: The module is missing.\n"
            "FIX: pip install foo\n"
            "CONFIDENCE: high\n"
        )
        explanation, fix, confidence = _parse_provider_response(raw)
        assert explanation == "The module is missing."
        assert fix == "pip install foo"
        assert confidence == ResolutionConfidence.HIGH

    def test_confidence_medium(self):
        raw = "EXPLANATION: x\nFIX: y\nCONFIDENCE: medium"
        _, _, confidence = _parse_provider_response(raw)
        assert confidence == ResolutionConfidence.MEDIUM

    def test_confidence_low(self):
        raw = "EXPLANATION: x\nFIX: y\nCONFIDENCE: low"
        _, _, confidence = _parse_provider_response(raw)
        assert confidence == ResolutionConfidence.LOW

    def test_invalid_confidence_defaults_low(self):
        raw = "EXPLANATION: x\nFIX: y\nCONFIDENCE: unknown_level"
        _, _, confidence = _parse_provider_response(raw)
        assert confidence == ResolutionConfidence.LOW

    def test_missing_fields_empty_strings(self):
        explanation, fix, confidence = _parse_provider_response("")
        assert explanation == ""
        assert fix == ""
        assert confidence == ResolutionConfidence.LOW


# ---------------------------------------------------------------------------
# ErrorResolutionEngine — known fixes
# ---------------------------------------------------------------------------

class TestKnownFixes:
    async def test_module_not_found_resolved(self):
        engine = ErrorResolutionEngine()
        result = await engine.resolve(ModuleNotFoundError("No module named 'requests'"))
        assert result.provider == "known_fix"
        assert "requests" in result.suggested_fix
        assert result.confidence == ResolutionConfidence.HIGH
        assert result.cache_hit is False

    async def test_module_name_interpolated(self):
        engine = ErrorResolutionEngine()
        result = await engine.resolve(ModuleNotFoundError("No module named 'numpy'"))
        assert "numpy" in result.suggested_fix

    async def test_file_not_found_resolved(self):
        engine = ErrorResolutionEngine()
        result = await engine.resolve(FileNotFoundError("file.txt not found"))
        assert result.provider == "known_fix"

    async def test_permission_error_resolved(self):
        engine = ErrorResolutionEngine()
        result = await engine.resolve(PermissionError("denied"))
        assert result.provider == "known_fix"

    async def test_connection_refused_resolved(self):
        engine = ErrorResolutionEngine()
        result = await engine.resolve(ConnectionRefusedError("connection refused"))
        assert result.provider == "known_fix"

    async def test_timeout_error_resolved(self):
        engine = ErrorResolutionEngine()
        result = await engine.resolve(TimeoutError("timed out"))
        assert result.provider == "known_fix"

    async def test_known_fix_cached(self):
        engine = ErrorResolutionEngine()
        err = FileNotFoundError("missing.txt")
        await engine.resolve(err)
        # Second call should be a cache hit
        result2 = await engine.resolve(err)
        assert result2.cache_hit is True
        assert result2.provider == "cache"


# ---------------------------------------------------------------------------
# ErrorResolutionEngine — cache hit
# ---------------------------------------------------------------------------

class TestCacheHit:
    async def test_cache_hit_returns_cached_result(self):
        engine = ErrorResolutionEngine()
        err = FileNotFoundError("x.txt")
        r1 = await engine.resolve(err)
        r2 = await engine.resolve(err)
        assert r2.cache_hit is True
        assert r2.explanation == r1.explanation
        assert r2.suggested_fix == r1.suggested_fix


# ---------------------------------------------------------------------------
# ErrorResolutionEngine — fallback (no provider, unknown error)
# ---------------------------------------------------------------------------

class TestFallback:
    async def test_unknown_error_returns_none_provider(self):
        engine = ErrorResolutionEngine()
        result = await engine.resolve(Exception("something weird happened"))
        assert result.provider == "none"
        assert result.confidence == ResolutionConfidence.LOW
        assert result.auto_applied is False


# ---------------------------------------------------------------------------
# ErrorResolutionEngine — AI provider
# ---------------------------------------------------------------------------

class TestProviderPath:
    async def test_provider_response_used(self):
        engine = ErrorResolutionEngine()
        mock_provider = AsyncMock()
        mock_provider.name = "mock_ai"
        mock_provider.suggest_fix.return_value = (
            "EXPLANATION: Explanation here.\nFIX: pip install bar\nCONFIDENCE: high\n"
        )
        engine.providers.get_primary = lambda: mock_provider

        result = await engine.resolve(Exception("custom error"))
        assert result.provider == "mock_ai"
        assert result.suggested_fix == "pip install bar"
        assert result.confidence == ResolutionConfidence.HIGH

    async def test_provider_failure_returns_error_result(self):
        engine = ErrorResolutionEngine()
        mock_provider = AsyncMock()
        mock_provider.name = "failing_ai"
        mock_provider.suggest_fix.side_effect = RuntimeError("AI down")
        engine.providers.get_primary = lambda: mock_provider

        result = await engine.resolve(Exception("any error"))
        assert result.provider == "error"
        assert result.confidence == ResolutionConfidence.LOW

    async def test_auto_apply_safe_fix(self):
        engine = ErrorResolutionEngine()
        mock_provider = AsyncMock()
        mock_provider.name = "mock_ai"
        mock_provider.suggest_fix.return_value = (
            "EXPLANATION: Install it.\nFIX: pip install foo\nCONFIDENCE: high\n"
        )
        engine.providers.get_primary = lambda: mock_provider

        result = await engine.resolve(Exception("some error"), auto_apply=True)
        assert result.auto_applied is True

    async def test_auto_apply_not_set_for_dangerous_fix(self):
        engine = ErrorResolutionEngine()
        mock_provider = AsyncMock()
        mock_provider.name = "mock_ai"
        mock_provider.suggest_fix.return_value = (
            "EXPLANATION: Delete it.\nFIX: rm -rf /\nCONFIDENCE: high\n"
        )
        engine.providers.get_primary = lambda: mock_provider

        result = await engine.resolve(Exception("some error"), auto_apply=True)
        assert result.auto_applied is False


# ---------------------------------------------------------------------------
# ErrorResolutionEngine — utilities
# ---------------------------------------------------------------------------

class TestEngineUtilities:
    def test_get_stats(self):
        engine = ErrorResolutionEngine()
        stats = engine.get_stats()
        assert "l1_cache_size" in stats
        assert "providers_available" in stats
        assert "known_fixes_count" in stats
        assert stats["known_fixes_count"] >= 5

    async def test_clear_cache(self):
        engine = ErrorResolutionEngine()
        await engine.resolve(FileNotFoundError("f.txt"))
        engine.clear_cache()
        # After clear, same error should not be a cache hit
        result = await engine.resolve(FileNotFoundError("f.txt"))
        assert result.cache_hit is False
