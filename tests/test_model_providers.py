"""Tests for core/model_providers.py — ProvidersMixin (via ModelAdapter)."""

import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from core.model_adapter import ModelAdapter


class _TestableAdapter(ModelAdapter):
    """Subclass that makes LLM_TIMEOUT settable for tests."""

    _llm_timeout_override = 60

    @property
    def LLM_TIMEOUT(self) -> int:
        return self._llm_timeout_override


def _make_mixin(**kwargs):
    """Create a testable ModelAdapter instance (inherits ProvidersMixin)."""
    obj = _TestableAdapter.__new__(_TestableAdapter)
    obj._local_profile_cooldowns = {}
    obj._local_profile_cooldown_reasons = {}
    obj._llm_timeout_override = 60
    obj.mcp_server_url = "http://localhost:8001"
    obj.gemini_cli_path = None
    obj.codex_cli_path = None
    obj.copilot_cli_path = None
    obj._embedding_dims = None
    obj._embedding_profile_name = None
    obj._mem_cache = {}
    obj.cache_db = None
    obj.cache_ttl = 3600
    obj._momento = None
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


class TestMakeRequestWithRetries(unittest.TestCase):
    """Tests for _make_request_with_retries."""

    @patch("core.model_adapter.requests")
    def test_success_first_attempt(self, mock_requests):
        mixin = _make_mixin()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_requests.request.return_value = mock_resp
        mock_requests.exceptions.RequestException = Exception

        result = mixin._make_request_with_retries("POST", "http://x", {}, {})
        self.assertEqual(result, mock_resp)
        mock_requests.request.assert_called_once()

    @patch("core.model_adapter.requests")
    @patch("time.sleep")
    def test_retries_on_failure_then_succeeds(self, mock_sleep, mock_requests):
        mixin = _make_mixin()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_requests.exceptions.RequestException = Exception
        mock_requests.request.side_effect = [Exception("fail"), mock_resp]

        result = mixin._make_request_with_retries("POST", "http://x", {}, {}, retries=2, backoff_factor=0.1)
        self.assertEqual(result, mock_resp)
        self.assertEqual(mock_requests.request.call_count, 2)

    @patch("core.model_adapter.requests")
    @patch("time.sleep")
    def test_exhausts_retries_raises(self, mock_sleep, mock_requests):
        mixin = _make_mixin()
        mock_requests.exceptions.RequestException = Exception
        mock_requests.request.side_effect = Exception("always fails")

        with self.assertRaises(Exception):
            mixin._make_request_with_retries("POST", "http://x", {}, {}, retries=3, backoff_factor=0.01)
        self.assertEqual(mock_requests.request.call_count, 3)

    @patch("core.model_adapter.requests")
    @patch("time.sleep")
    def test_backoff_increases(self, mock_sleep, mock_requests):
        mixin = _make_mixin()
        mock_requests.exceptions.RequestException = Exception
        mock_requests.request.side_effect = [Exception("1"), Exception("2"), MagicMock()]

        mixin._make_request_with_retries("POST", "http://x", {}, {}, retries=3, backoff_factor=1.0)
        # sleep(1*2^0=1), sleep(1*2^1=2)
        calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertAlmostEqual(calls[0], 1.0)
        self.assertAlmostEqual(calls[1], 2.0)


class TestCallOpenRouter(unittest.TestCase):
    """Tests for call_openrouter."""

    @patch("core.model_providers.resolve_openrouter_api_key", return_value="test-key")
    @patch("core.model_providers.config")
    def test_no_key_raises(self, mock_config, mock_resolve):
        mock_resolve.return_value = None
        mixin = _make_mixin()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                mixin.call_openrouter("hello")

    @patch("core.model_providers.config")
    @patch("core.model_providers.resolve_openrouter_api_key", return_value="test-key")
    def test_successful_call(self, mock_resolve, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {"fast": "model/x"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        result = mixin.call_openrouter("hello")
        self.assertEqual(result, "hi")

    @patch("core.model_providers.config")
    @patch("core.model_providers.resolve_openrouter_api_key", return_value="test-key")
    def test_model_selection_from_list(self, mock_resolve, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {"code": ["model/a", "model/b"]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        result = mixin.call_openrouter("hello", route_key="code")
        self.assertEqual(result, "ok")
        call_payload = mixin._make_request_with_retries.call_args[0][3]
        self.assertIn(call_payload["model"], ["model/a", "model/b"])

    @patch("core.model_providers.config")
    @patch("core.model_providers.resolve_openrouter_api_key", return_value="test-key")
    def test_fallback_on_error_response(self, mock_resolve, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {"fast": "model/x", "fallback": "model/fallback"}
        error_resp = MagicMock()
        error_resp.json.return_value = {"error": "model down"}
        ok_resp = MagicMock()
        ok_resp.json.return_value = {"choices": [{"message": {"content": "fallback ok"}}]}
        mixin._make_request_with_retries = MagicMock(side_effect=[error_resp, ok_resp])

        result = mixin.call_openrouter("hello")
        self.assertEqual(result, "fallback ok")
        self.assertEqual(mixin._make_request_with_retries.call_count, 2)

    @patch("core.model_providers.config")
    @patch("core.model_providers.resolve_openrouter_api_key", return_value="test-key")
    def test_fallback_also_fails_raises(self, mock_resolve, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {"fast": "model/x", "fallback": "model/fallback"}
        error_resp = MagicMock()
        error_resp.json.return_value = {"error": "down"}
        mixin._make_request_with_retries = MagicMock(return_value=error_resp)

        with self.assertRaises(ValueError):
            mixin.call_openrouter("hello")

    @patch("core.model_providers.config")
    @patch("core.model_providers.resolve_openrouter_api_key", return_value="test-key")
    def test_default_model_when_no_route_key(self, mock_resolve, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "default"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        mixin.call_openrouter("hello")
        payload = mixin._make_request_with_retries.call_args[0][3]
        self.assertEqual(payload["model"], "google/gemini-2.0-flash-001")


class TestCallLocalProfile(unittest.TestCase):
    """Tests for call_local_profile and related helpers."""

    def test_unknown_profile_raises(self):
        mixin = _make_mixin()
        mixin._get_local_profiles = MagicMock(return_value={})
        with self.assertRaises(ValueError):
            mixin.call_local_profile("nonexistent", "prompt")

    def test_successful_profile_call(self):
        mixin = _make_mixin()
        profile = {"provider": "openai_compatible", "model": "test", "base_url": "http://localhost:8080/v1"}
        mixin._get_local_profiles = MagicMock(return_value={"fast": profile})
        mixin._call_local_profile_provider = MagicMock(return_value="result")
        mixin._clear_local_profile_unhealthy = MagicMock()

        result = mixin.call_local_profile("fast", "prompt")
        self.assertEqual(result, "result")

    def test_fallback_on_failure(self):
        mixin = _make_mixin()
        fast_profile = {"provider": "openai_compatible", "model": "fast-m", "fallback_profile": "slow"}
        slow_profile = {"provider": "openai_compatible", "model": "slow-m"}
        mixin._get_local_profiles = MagicMock(return_value={"fast": fast_profile, "slow": slow_profile})
        mixin._call_local_profile_provider = MagicMock(side_effect=[RuntimeError("fail"), "ok"])
        mixin._mark_local_profile_unhealthy = MagicMock()
        mixin._clear_local_profile_unhealthy = MagicMock()

        result = mixin.call_local_profile("fast", "prompt")
        self.assertEqual(result, "ok")

    def test_fallback_loop_detection(self):
        mixin = _make_mixin()
        profile_a = {"provider": "openai_compatible", "model": "a", "fallback_profile": "b"}
        profile_b = {"provider": "openai_compatible", "model": "b", "fallback_profile": "a"}
        mixin._get_local_profiles = MagicMock(return_value={"a": profile_a, "b": profile_b})
        mixin._call_local_profile_provider = MagicMock(side_effect=RuntimeError("fail"))
        mixin._mark_local_profile_unhealthy = MagicMock()

        with self.assertRaises(RuntimeError):
            mixin.call_local_profile("a", "prompt")

    def test_cooldown_skips_to_fallback(self):
        mixin = _make_mixin()
        mixin._local_profile_cooldowns["fast"] = time.time() + 60
        mixin._local_profile_cooldown_reasons["fast"] = "broken"
        fast_profile = {"provider": "openai_compatible", "model": "fast-m", "fallback_profile": "slow"}
        slow_profile = {"provider": "openai_compatible", "model": "slow-m"}
        mixin._get_local_profiles = MagicMock(return_value={"fast": fast_profile, "slow": slow_profile})
        mixin._call_local_profile_provider = MagicMock(return_value="slow result")
        mixin._clear_local_profile_unhealthy = MagicMock()

        result = mixin.call_local_profile("fast", "prompt")
        self.assertEqual(result, "slow result")

    def test_cooldown_no_fallback_raises(self):
        mixin = _make_mixin()
        mixin._local_profile_cooldowns["fast"] = time.time() + 60
        mixin._local_profile_cooldown_reasons["fast"] = "broken"
        fast_profile = {"provider": "openai_compatible", "model": "fast-m"}
        mixin._get_local_profiles = MagicMock(return_value={"fast": fast_profile})

        with self.assertRaises(RuntimeError):
            mixin.call_local_profile("fast", "prompt")


class TestProfileHelpers(unittest.TestCase):
    """Tests for profile helper methods."""

    def test_profile_timeout_valid(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._profile_timeout({"request_timeout_seconds": 30}, key="request_timeout_seconds", default=60.0), 30.0)

    def test_profile_timeout_invalid(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._profile_timeout({"request_timeout_seconds": "bad"}, key="request_timeout_seconds", default=60.0), 60.0)

    def test_profile_retries_valid(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._profile_retries({"retries": 5}), 5)

    def test_profile_retries_invalid(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._profile_retries({"retries": "bad"}), 3)

    def test_profile_backoff_valid(self):
        mixin = _make_mixin()
        self.assertAlmostEqual(mixin._profile_backoff({"backoff_factor": 1.5}), 1.5)

    def test_profile_backoff_negative(self):
        mixin = _make_mixin()
        self.assertAlmostEqual(mixin._profile_backoff({"backoff_factor": -1}), 0.5)

    def test_profile_fallbacks_list(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._profile_fallbacks({"fallback_profiles": ["a", "b"]}), ["a", "b"])

    def test_profile_fallbacks_single(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._profile_fallbacks({"fallback_profile": "a"}), ["a"])

    def test_profile_fallbacks_none(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._profile_fallbacks({}), [])

    def test_provider_dispatch_unsupported(self):
        mixin = _make_mixin()
        with self.assertRaises(ValueError):
            mixin._call_local_profile_provider({"provider": "unsupported"}, "prompt")

    def test_cooldown_remaining_zero_when_not_set(self):
        mixin = _make_mixin()
        self.assertEqual(mixin._local_profile_cooldown_remaining("fast"), 0.0)

    def test_cooldown_remaining_positive(self):
        mixin = _make_mixin()
        mixin._local_profile_cooldowns["fast"] = time.time() + 100
        remaining = mixin._local_profile_cooldown_remaining("fast")
        self.assertGreater(remaining, 0)

    def test_mark_and_clear_unhealthy(self):
        mixin = _make_mixin()
        profile = {"cooldown_seconds": 10}
        mixin._mark_local_profile_unhealthy("fast", profile, RuntimeError("err"))
        self.assertIn("fast", mixin._local_profile_cooldowns)
        self.assertIn("fast", mixin._local_profile_cooldown_reasons)
        mixin._clear_local_profile_unhealthy("fast")
        self.assertNotIn("fast", mixin._local_profile_cooldowns)
        self.assertNotIn("fast", mixin._local_profile_cooldown_reasons)


if __name__ == "__main__":
    unittest.main()
