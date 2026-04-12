"""Tests for error resolution engine."""

import os
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from aura.error_resolution.engine import ErrorResolutionEngine
from aura.error_resolution.types import ResolutionConfidence, ResolutionResult


class TestErrorResolutionEngine:
    """Tests for the main resolution engine."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def engine(self, temp_db):
        """Create engine with temp cache."""
        from aura.error_resolution.cache import FourLayerCache

        cache = FourLayerCache(l2_path=temp_db)
        return ErrorResolutionEngine(cache=cache)

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, engine):
        """Should return cached result without calling provider."""
        error = ValueError("test error")

        # Pre-populate cache
        cached_result = ResolutionResult(
            original_error="test error",
            explanation="cached explanation",
            suggested_fix="cached fix",
            confidence=ResolutionConfidence.HIGH,
            auto_applied=False,
            cache_hit=True,
            provider="cache",
            execution_time_ms=10,
        )
        cache_key = engine.cache.make_key(error, None)
        engine.cache.set(cache_key, cached_result)

        # Resolve should use cache
        result = await engine.resolve(error)

        assert result.cache_hit is True
        assert result.provider == "cache"
        assert result.explanation == "cached explanation"

    @pytest.mark.asyncio
    async def test_known_fix_lookup(self, engine):
        """Should check known fixes before AI provider."""
        # ModuleNotFoundError has a known fix
        error = ModuleNotFoundError("No module named 'requests'")

        with patch.object(engine.providers, "get_primary") as mock_get_primary:
            # Should not call provider if known fix found
            mock_provider = AsyncMock()
            mock_get_primary.return_value = mock_provider

            result = await engine.resolve(error)

            # Provider should not be called for known fixes
            mock_provider.suggest_fix.assert_not_called()
            assert result.provider == "known_fix"
            assert "pip install" in result.suggested_fix

    @pytest.mark.asyncio
    async def test_provider_called_on_cache_miss(self, engine):
        """Should query AI provider when no cache hit."""
        error = ValueError("unique error not in cache")

        mock_provider = AsyncMock()
        mock_provider.name = "test_provider"
        mock_provider.suggest_fix.return_value = "EXPLANATION: Test explanation\nFIX: test fix command\nCONFIDENCE: high"

        with patch.object(engine.providers, "get_primary", return_value=mock_provider):
            result = await engine.resolve(error)

        mock_provider.suggest_fix.assert_called_once()
        assert result.provider == "test_provider"
        assert result.explanation == "Test explanation"
        assert result.suggested_fix == "test fix command"

    @pytest.mark.asyncio
    async def test_provider_error_handling(self, engine):
        """Should handle provider errors gracefully."""
        error = ValueError("test")

        mock_provider = AsyncMock()
        mock_provider.name = "failing_provider"
        mock_provider.suggest_fix.side_effect = Exception("API Error")

        with patch.object(engine.providers, "get_primary", return_value=mock_provider):
            result = await engine.resolve(error)

        assert result.provider == "error"
        assert "Could not resolve" in result.explanation
        assert result.confidence == ResolutionConfidence.LOW

    @pytest.mark.asyncio
    async def test_auto_apply_respects_safety(self, engine):
        """Should only auto-apply safe commands."""
        error = ValueError("test")

        # Safe command
        mock_provider = AsyncMock()
        mock_provider.name = "test"
        mock_provider.suggest_fix.return_value = (
            "EXPLANATION: Test\n"
            "FIX: pip install package\n"  # Safe
            "CONFIDENCE: high"
        )

        with patch.object(engine.providers, "get_primary", return_value=mock_provider):
            result = await engine.resolve(error, auto_apply=True)

        # pip install is safe, should be marked for auto-apply
        assert result.auto_applied is True

    @pytest.mark.asyncio
    async def test_auto_apply_blocked_for_dangerous(self, engine):
        """Should not auto-apply dangerous commands."""
        error = ValueError("test")

        mock_provider = AsyncMock()
        mock_provider.name = "test"
        mock_provider.suggest_fix.return_value = (
            "EXPLANATION: Test\n"
            "FIX: sudo rm -rf /\n"  # Dangerous!
            "CONFIDENCE: high"
        )

        with patch.object(engine.providers, "get_primary", return_value=mock_provider):
            result = await engine.resolve(error, auto_apply=True)

        # Should not auto-apply dangerous commands
        assert result.auto_applied is False

    @pytest.mark.asyncio
    async def test_auto_apply_blocked_for_low_confidence(self, engine):
        """Should not auto-apply low confidence results."""
        error = ValueError("test")

        mock_provider = AsyncMock()
        mock_provider.name = "test"
        mock_provider.suggest_fix.return_value = (
            "EXPLANATION: Unclear\n"
            "FIX: pip install package\n"  # Safe but low confidence
            "CONFIDENCE: low"
        )

        with patch.object(engine.providers, "get_primary", return_value=mock_provider):
            result = await engine.resolve(error, auto_apply=True)

        # Should not auto-apply low confidence even if safe
        assert result.auto_applied is False

    def test_clear_cache(self, engine):
        """Should clear all cache layers."""
        error = ValueError("test")
        cache_key = engine.cache.make_key(error, None)

        # Add to cache
        result = ResolutionResult(
            original_error="test",
            explanation="test",
            suggested_fix="test",
            confidence=ResolutionConfidence.HIGH,
            auto_applied=False,
            cache_hit=False,
            provider="test",
            execution_time_ms=100,
        )
        engine.cache.set(cache_key, result)

        # Verify it's there
        assert engine.cache.get(cache_key) is not None

        # Clear cache
        engine.clear_cache()

        # Verify it's gone
        assert engine.cache.get(cache_key) is None

    def test_get_stats(self, engine):
        """Should return engine statistics."""
        stats = engine.get_stats()

        assert "l1_cache_size" in stats
        assert "providers_available" in stats
        assert "known_fixes_count" in stats
        assert stats["l1_cache_size"] == 0
        assert stats["known_fixes_count"] > 0  # Built-in fixes

    @pytest.mark.asyncio
    async def test_result_cached_after_provider_call(self, engine):
        """Should cache result after successful provider call."""
        error = ValueError("cache test")

        mock_provider = AsyncMock()
        mock_provider.name = "test"
        mock_provider.suggest_fix.return_value = "EXPLANATION: Test\nFIX: fix\nCONFIDENCE: high"

        with patch.object(engine.providers, "get_primary", return_value=mock_provider):
            # First call should use provider
            result1 = await engine.resolve(error)

            # Second call should use cache
            result2 = await engine.resolve(error)

        # Provider should only be called once
        assert mock_provider.suggest_fix.call_count == 1

        # Second result should be from cache
        assert result2.cache_hit is True
        assert result2.provider == "cache"
