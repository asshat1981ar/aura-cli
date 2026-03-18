"""D3: Model adapter fallback, cooldown/recovery, and cache hit/miss tests.

Tests local model profile failover chains, cooldown behavior,
respond() fallback cascade, and cache layer interactions.
"""
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from core.model_adapter import ModelAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config(overrides=None):
    """Return a config.get side-effect that returns sensible defaults."""
    base = {}
    if overrides:
        base.update(overrides)

    def getter(key, default=None):
        return base.get(key, default)
    return getter


def _make_adapter(**config_overrides):
    """Create a ModelAdapter with mocked config."""
    with patch("core.model_adapter.config.get", side_effect=_mock_config(config_overrides)), \
         patch("core.embedding_service.config.get", side_effect=_mock_config(config_overrides)), \
         patch("core.model_adapter.log_json"), \
         patch("core.embedding_service.log_json"):
        return ModelAdapter()


# ---------------------------------------------------------------------------
# Local profile fallback chains
# ---------------------------------------------------------------------------

class TestLocalProfileFallback:
    """Test the _call_local_profile_with_fallbacks chain."""

    def test_primary_profile_success_clears_cooldown(self):
        adapter = _make_adapter()
        profiles = {
            "primary": {"provider": "openai_compatible", "model": "m1", "base_url": "http://localhost:8080/v1"},
        }
        with patch.object(adapter, "_get_local_profiles", return_value=profiles), \
             patch.object(adapter, "_call_local_profile_provider", return_value="ok"):
            result = adapter.call_local_profile("primary", "test")
            assert result == "ok"
            assert adapter._local_profile_cooldown_remaining("primary") == 0.0

    def test_fallback_on_primary_failure(self):
        adapter = _make_adapter()
        profiles = {
            "primary": {
                "provider": "openai_compatible", "model": "m1",
                "base_url": "http://localhost:8080/v1",
                "fallback_profiles": ["secondary"],
                "cooldown_seconds": 10,
            },
            "secondary": {
                "provider": "openai_compatible", "model": "m2",
                "base_url": "http://localhost:8081/v1",
            },
        }
        call_count = {"n": 0}

        def provider_side_effect(profile, prompt):
            call_count["n"] += 1
            if profile["model"] == "m1":
                raise ConnectionError("primary down")
            return "fallback response"

        with patch.object(adapter, "_get_local_profiles", return_value=profiles), \
             patch.object(adapter, "_call_local_profile_provider", side_effect=provider_side_effect), \
             patch("core.model_adapter.log_json"):
            result = adapter.call_local_profile("primary", "test")
            assert result == "fallback response"
            assert call_count["n"] == 2

    def test_fallback_chain_exhausted_raises(self):
        adapter = _make_adapter()
        profiles = {
            "a": {"provider": "openai_compatible", "model": "a", "base_url": "http://x/v1", "fallback_profiles": ["b"]},
            "b": {"provider": "openai_compatible", "model": "b", "base_url": "http://x/v1"},
        }

        def always_fail(profile, prompt):
            raise ConnectionError("down")

        with patch.object(adapter, "_get_local_profiles", return_value=profiles), \
             patch.object(adapter, "_call_local_profile_provider", side_effect=always_fail), \
             patch("core.model_adapter.log_json"):
            with pytest.raises(ConnectionError):
                adapter.call_local_profile("a", "test")

    def test_fallback_loop_detection(self):
        adapter = _make_adapter()
        profiles = {
            "a": {"provider": "openai_compatible", "model": "a", "base_url": "http://x/v1", "fallback_profiles": ["b"]},
            "b": {"provider": "openai_compatible", "model": "b", "base_url": "http://x/v1", "fallback_profiles": ["a"]},
        }

        def always_fail(profile, prompt):
            raise ConnectionError("down")

        with patch.object(adapter, "_get_local_profiles", return_value=profiles), \
             patch.object(adapter, "_call_local_profile_provider", side_effect=always_fail), \
             patch("core.model_adapter.log_json"):
            with pytest.raises((ConnectionError, RuntimeError)):
                adapter.call_local_profile("a", "test")


# ---------------------------------------------------------------------------
# Cooldown and recovery
# ---------------------------------------------------------------------------

class TestCooldownRecovery:
    """Test cooldown marking and automatic recovery."""

    def test_mark_unhealthy_sets_cooldown(self):
        adapter = _make_adapter()
        profile = {"cooldown_seconds": 5}
        adapter._mark_local_profile_unhealthy("p1", profile, ConnectionError("down"))
        assert adapter._local_profile_cooldown_remaining("p1") > 0
        assert "p1" in adapter._local_profile_cooldown_reasons

    def test_cooldown_expires(self):
        adapter = _make_adapter()
        profile = {"cooldown_seconds": 0.05}
        adapter._mark_local_profile_unhealthy("p1", profile, ConnectionError("down"))
        assert adapter._local_profile_cooldown_remaining("p1") > 0
        time.sleep(0.1)
        assert adapter._local_profile_cooldown_remaining("p1") == 0.0

    def test_clear_unhealthy_removes_cooldown(self):
        adapter = _make_adapter()
        profile = {"cooldown_seconds": 60}
        adapter._mark_local_profile_unhealthy("p1", profile, ConnectionError("down"))
        adapter._clear_local_profile_unhealthy("p1")
        assert adapter._local_profile_cooldown_remaining("p1") == 0.0
        assert "p1" not in adapter._local_profile_cooldown_reasons

    def test_cooldown_skips_to_fallback(self):
        adapter = _make_adapter()
        profiles = {
            "primary": {
                "provider": "openai_compatible", "model": "m1",
                "base_url": "http://localhost:8080/v1",
                "fallback_profiles": ["secondary"],
                "cooldown_seconds": 60,
            },
            "secondary": {
                "provider": "openai_compatible", "model": "m2",
                "base_url": "http://localhost:8081/v1",
            },
        }
        # Put primary in cooldown
        adapter._mark_local_profile_unhealthy("primary", profiles["primary"], ConnectionError("down"))

        with patch.object(adapter, "_get_local_profiles", return_value=profiles), \
             patch.object(adapter, "_call_local_profile_provider", return_value="from secondary"), \
             patch("core.model_adapter.log_json"):
            result = adapter.call_local_profile("primary", "test")
            assert result == "from secondary"

    def test_cooldown_no_fallback_raises(self):
        adapter = _make_adapter()
        profiles = {
            "primary": {
                "provider": "openai_compatible", "model": "m1",
                "base_url": "http://localhost:8080/v1",
                "cooldown_seconds": 60,
            },
        }
        adapter._mark_local_profile_unhealthy("primary", profiles["primary"], ConnectionError("down"))

        with patch.object(adapter, "_get_local_profiles", return_value=profiles), \
             patch("core.model_adapter.log_json"):
            with pytest.raises(RuntimeError, match="cooling down"):
                adapter.call_local_profile("primary", "test")


# ---------------------------------------------------------------------------
# respond() fallback cascade
# ---------------------------------------------------------------------------

class TestRespondFallback:
    """Test the respond() method's OpenAI → OpenRouter → Local cascade."""

    def test_respond_returns_cached(self):
        adapter = _make_adapter()
        adapter._cache.put("hello", "cached")
        result = adapter.respond("hello")
        assert result == "cached"

    def test_respond_uses_router_first(self):
        adapter = _make_adapter()
        mock_router = MagicMock()
        mock_router.route.return_value = "router response"
        adapter.router = mock_router
        with patch.object(adapter, "_call_with_timeout", return_value="router response"):
            result = adapter.respond("hello")
        assert result == "router response"

    def test_respond_falls_back_to_openai(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_call_with_timeout", return_value="openai response"), \
             patch("core.model_adapter.log_json"):
            result = adapter.respond("hello")
        assert result == "openai response"

    def test_respond_falls_through_to_local(self):
        adapter = _make_adapter()
        call_order = []

        def timeout_side_effect(fn, *args, **kwargs):
            name = fn.__name__
            call_order.append(name)
            if name in ("call_openai", "call_openrouter"):
                raise ConnectionError(f"{name} failed")
            return "local response"

        with patch.object(adapter, "_call_with_timeout", side_effect=timeout_side_effect), \
             patch("core.model_adapter.log_json"):
            result = adapter.respond("hello")
        assert result == "local response"
        assert "call_openai" in call_order
        assert "call_openrouter" in call_order
        assert "call_local" in call_order

    def test_respond_caches_successful_response(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_call_with_timeout", return_value="fresh response"), \
             patch("core.model_adapter.log_json"):
            adapter.respond("new prompt")
        assert adapter._cache.get("new prompt") == "fresh response"


# ---------------------------------------------------------------------------
# Profile config helpers
# ---------------------------------------------------------------------------

class TestProfileConfigHelpers:
    """Test _profile_timeout, _profile_retries, _profile_backoff edge cases."""

    def test_timeout_default(self):
        adapter = _make_adapter()
        assert adapter._profile_timeout({}, key="timeout", default=30.0) == 30.0

    def test_timeout_from_profile(self):
        adapter = _make_adapter()
        assert adapter._profile_timeout({"timeout": 45}, key="timeout", default=30.0) == 45.0

    def test_timeout_invalid_returns_default(self):
        adapter = _make_adapter()
        assert adapter._profile_timeout({"timeout": "not_a_number"}, key="timeout", default=30.0) == 30.0

    def test_timeout_zero_returns_default(self):
        adapter = _make_adapter()
        assert adapter._profile_timeout({"timeout": 0}, key="timeout", default=30.0) == 30.0

    def test_retries_default(self):
        adapter = _make_adapter()
        assert adapter._profile_retries({}) == 3

    def test_retries_from_profile(self):
        adapter = _make_adapter()
        assert adapter._profile_retries({"retries": 5}) == 5

    def test_retries_invalid_returns_default(self):
        adapter = _make_adapter()
        assert adapter._profile_retries({"retries": "bad"}) == 3

    def test_backoff_default(self):
        adapter = _make_adapter()
        assert adapter._profile_backoff({}) == 0.5

    def test_backoff_from_profile(self):
        adapter = _make_adapter()
        assert adapter._profile_backoff({"backoff_factor": 1.0}) == 1.0

    def test_fallback_profiles_list(self):
        adapter = _make_adapter()
        assert adapter._profile_fallbacks({"fallback_profiles": ["a", "b"]}) == ["a", "b"]

    def test_fallback_profile_single(self):
        adapter = _make_adapter()
        assert adapter._profile_fallbacks({"fallback_profile": "a"}) == ["a"]

    def test_fallback_none(self):
        adapter = _make_adapter()
        assert adapter._profile_fallbacks({}) == []
