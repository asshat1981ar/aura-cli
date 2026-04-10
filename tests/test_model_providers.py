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


class TestCallOpenAI(unittest.TestCase):
    """Tests for call_openai."""

    @patch("core.model_providers.resolve_openai_api_key", return_value=None)
    def test_no_key_raises(self, _):
        mixin = _make_mixin()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError, msg="Should raise when no OpenAI key"):
                mixin.call_openai("hello")

    @patch("core.model_providers.resolve_openai_api_key", return_value="sk-test")
    def test_successful_call(self, _):
        mixin = _make_mixin()
        mixin._log_telemetry = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "gpt answer"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        result = mixin.call_openai("hello")
        self.assertEqual(result, "gpt answer")
        mixin._log_telemetry.assert_called_once()

    @patch("core.model_providers.resolve_openai_api_key", return_value="sk-test")
    def test_sends_correct_url_and_model(self, _):
        mixin = _make_mixin()
        mixin._log_telemetry = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        mixin.call_openai("test")
        call_args = mixin._make_request_with_retries.call_args[0]
        self.assertEqual(call_args[0], "POST")
        self.assertIn("openai.com", call_args[1])
        self.assertEqual(call_args[3]["model"], "gpt-4o-mini")

    @patch("core.model_providers.resolve_openai_api_key", return_value=None)
    def test_falls_back_to_env_key(self, _):
        mixin = _make_mixin()
        mixin._log_telemetry = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "env key"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-sk-test"}):
            result = mixin.call_openai("hello")
        self.assertEqual(result, "env key")


class TestCallAnthropic(unittest.TestCase):
    """Tests for call_anthropic."""

    @patch("core.model_providers.resolve_anthropic_api_key", return_value=None)
    def test_no_key_raises(self, _):
        mixin = _make_mixin()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                mixin.call_anthropic("hello")

    @patch("core.model_providers.resolve_anthropic_api_key", return_value="ant-test")
    def test_successful_call(self, _):
        mixin = _make_mixin()
        mixin._log_telemetry = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"text": "claude answer"}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        result = mixin.call_anthropic("hello")
        self.assertEqual(result, "claude answer")

    @patch("core.model_providers.resolve_anthropic_api_key", return_value="ant-test")
    def test_sends_correct_headers(self, _):
        mixin = _make_mixin()
        mixin._log_telemetry = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"text": "ok"}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        mixin.call_anthropic("test")
        headers = mixin._make_request_with_retries.call_args[0][2]
        self.assertIn("x-api-key", headers)
        self.assertIn("anthropic-version", headers)

    @patch("core.model_providers.resolve_anthropic_api_key", return_value=None)
    def test_falls_back_to_env_key(self, _):
        mixin = _make_mixin()
        mixin._log_telemetry = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"text": "env claude"}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-ant-test"}):
            result = mixin.call_anthropic("hello")
        self.assertEqual(result, "env claude")


class TestCallLocal(unittest.TestCase):
    """Tests for call_local (legacy command-based path)."""

    @patch("core.model_providers.config")
    def test_not_configured_returns_message(self, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = None
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        result = mixin.call_local("hello")
        self.assertIn("not configured", result.lower())

    @patch("core.model_providers.config")
    @patch("subprocess.run")
    def test_command_success(self, mock_run, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = "my-local-model"
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        mock_proc = MagicMock()
        mock_proc.stdout = "local response"
        mock_run.return_value = mock_proc
        result = mixin.call_local("prompt")
        self.assertEqual(result, "local response")

    @patch("core.model_providers.config")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_command_not_found(self, mock_run, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = "nonexistent-model"
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        result = mixin.call_local("prompt")
        self.assertIn("not found", result.lower())

    @patch("core.model_providers.config")
    @patch("subprocess.run")
    def test_command_process_error(self, mock_run, mock_config):
        import subprocess

        mixin = _make_mixin()
        mock_config.get.return_value = "my-local-model"
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="oops")
        result = mixin.call_local("prompt")
        self.assertIn("Error", result)


class TestCallGeminiCLI(unittest.TestCase):
    """Tests for call_gemini."""

    def test_no_path_raises(self):
        mixin = _make_mixin(gemini_cli_path=None)
        with self.assertRaises(ValueError):
            mixin.call_gemini("hello")

    @patch("subprocess.run")
    def test_successful_call(self, mock_run):
        mixin = _make_mixin(gemini_cli_path="/usr/bin/gemini")
        mock_proc = MagicMock()
        mock_proc.stdout = "  gemini answer  "
        mock_run.return_value = mock_proc
        result = mixin.call_gemini("hello")
        self.assertEqual(result, "gemini answer")

    @patch("subprocess.run")
    def test_subprocess_error(self, mock_run):
        import subprocess

        mixin = _make_mixin(gemini_cli_path="/usr/bin/gemini")
        mock_run.side_effect = subprocess.CalledProcessError(1, "gemini", stderr="api error")
        with self.assertRaises(RuntimeError, msg="Should raise RuntimeError on CLI failure"):
            mixin.call_gemini("hello")


class TestCallCodexCLI(unittest.TestCase):
    """Tests for call_codex."""

    def test_no_path_raises(self):
        mixin = _make_mixin(codex_cli_path=None)
        with self.assertRaises(ValueError):
            mixin.call_codex("hello")

    @patch("subprocess.run")
    def test_successful_call(self, mock_run):
        mixin = _make_mixin(codex_cli_path="/usr/bin/codex")
        mock_proc = MagicMock()
        mock_proc.stdout = "  codex answer  "
        mock_run.return_value = mock_proc
        result = mixin.call_codex("hello")
        self.assertEqual(result, "codex answer")

    @patch("subprocess.run")
    def test_subprocess_error(self, mock_run):
        import subprocess

        mixin = _make_mixin(codex_cli_path="/usr/bin/codex")
        mock_run.side_effect = subprocess.CalledProcessError(1, "codex", stderr="fail")
        with self.assertRaises(RuntimeError):
            mixin.call_codex("hello")


class TestCallLocalOllama(unittest.TestCase):
    """Tests for _call_local_ollama."""

    def test_missing_model_raises(self):
        mixin = _make_mixin()
        with self.assertRaises(ValueError):
            mixin._call_local_ollama({"base_url": "http://localhost:11434"}, "prompt")

    def test_successful_response(self):
        mixin = _make_mixin()
        profile = {"model": "llama3", "base_url": "http://localhost:11434"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ollama output"}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        result = mixin._call_local_ollama(profile, "hello")
        self.assertEqual(result, "ollama output")

    def test_temperature_and_max_tokens_in_payload(self):
        mixin = _make_mixin()
        profile = {"model": "llama3", "temperature": 0.5, "max_tokens": 512}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        mixin._call_local_ollama(profile, "test")
        payload = mixin._make_request_with_retries.call_args[0][3]
        self.assertIn("options", payload)
        self.assertAlmostEqual(payload["options"]["temperature"], 0.5)
        self.assertEqual(payload["options"]["num_predict"], 512)

    def test_uses_default_base_url(self):
        mixin = _make_mixin()
        profile = {"model": "llama3"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "default url"}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        mixin._call_local_ollama(profile, "test")
        url = mixin._make_request_with_retries.call_args[0][1]
        self.assertIn("11434", url)


class TestCallLocalOpenAICompatible(unittest.TestCase):
    """Tests for _call_local_openai_compatible."""

    def test_missing_model_raises(self):
        mixin = _make_mixin()
        with self.assertRaises(ValueError):
            mixin._call_local_openai_compatible({"base_url": "http://localhost:8080/v1"}, "prompt")

    def test_successful_response(self):
        mixin = _make_mixin()
        profile = {"model": "local-model", "base_url": "http://localhost:8080/v1"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "local answer"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        result = mixin._call_local_openai_compatible(profile, "hello")
        self.assertEqual(result, "local answer")

    def test_api_key_added_to_headers(self):
        mixin = _make_mixin()
        profile = {"model": "local-model", "api_key": "secret"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        mixin._call_local_openai_compatible(profile, "hello")
        headers = mixin._make_request_with_retries.call_args[0][2]
        self.assertIn("Authorization", headers)
        self.assertIn("secret", headers["Authorization"])

    def test_base_url_auto_append_v1(self):
        mixin = _make_mixin()
        profile = {"model": "m", "base_url": "http://localhost:8080"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mixin._make_request_with_retries = MagicMock(return_value=mock_resp)

        mixin._call_local_openai_compatible(profile, "test")
        url = mixin._make_request_with_retries.call_args[0][1]
        self.assertIn("/v1/chat/completions", url)


class TestCallLocalCommandProfile(unittest.TestCase):
    """Tests for _call_local_command_profile."""

    @patch("subprocess.run")
    def test_command_with_prompt_placeholder(self, mock_run):
        mixin = _make_mixin()
        mock_proc = MagicMock()
        mock_proc.stdout = "cmd output"
        mock_run.return_value = mock_proc
        profile = {"command": "mymodel --input {prompt}"}

        result = mixin._call_local_command_profile(profile, "test input")
        self.assertEqual(result, "cmd output")
        called_cmd = mock_run.call_args[0][0]
        self.assertFalse(any("{prompt}" in part for part in called_cmd))

    @patch("subprocess.run")
    def test_command_via_stdin(self, mock_run):
        mixin = _make_mixin()
        mock_proc = MagicMock()
        mock_proc.stdout = "stdin output"
        mock_run.return_value = mock_proc
        profile = {"command": "mymodel --stream"}

        result = mixin._call_local_command_profile(profile, "hello")
        self.assertEqual(result, "stdin output")
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs.get("input"), "hello")

    def test_invalid_command_raises(self):
        mixin = _make_mixin()
        with self.assertRaises(ValueError):
            mixin._call_local_command_profile({"command": 123}, "prompt")


class TestResolveLocalProfileName(unittest.TestCase):
    """Tests for _resolve_local_profile_name."""

    @patch("core.model_providers.config")
    def test_resolves_existing_profile(self, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {"fast": "fast-profile"}
        mixin._get_local_profiles = MagicMock(return_value={"fast-profile": {}})

        result = mixin._resolve_local_profile_name("fast")
        self.assertEqual(result, "fast-profile")

    @patch("core.model_providers.config")
    def test_returns_none_for_unknown_route(self, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {}
        mixin._get_local_profiles = MagicMock(return_value={})

        result = mixin._resolve_local_profile_name("missing")
        self.assertIsNone(result)

    @patch("core.model_providers.config")
    def test_returns_none_if_profile_not_in_profiles(self, mock_config):
        mixin = _make_mixin()
        mock_config.get.return_value = {"fast": "nonexistent"}
        mixin._get_local_profiles = MagicMock(return_value={})

        result = mixin._resolve_local_profile_name("fast")
        self.assertIsNone(result)


class TestExecuteTool(unittest.TestCase):
    """Tests for _execute_tool."""

    @patch("core.model_adapter.requests")
    def test_known_mcp_tool_success(self, mock_requests):
        mixin = _make_mixin()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "data"}
        mock_resp.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_resp
        mock_requests.exceptions.RequestException = Exception

        result = mixin._execute_tool("get_repo", {"owner": "me"})
        import json as _json

        self.assertEqual(_json.loads(result), {"result": "data"})

    @patch("core.model_adapter.requests")
    def test_known_mcp_tool_request_error(self, mock_requests):
        mixin = _make_mixin()
        mock_requests.post.side_effect = Exception("connection refused")
        mock_requests.exceptions.RequestException = Exception

        result = mixin._execute_tool("get_repo", {"owner": "me"})
        self.assertIn("failed", result.lower())

    @patch("subprocess.run")
    def test_unknown_tool_uses_npx(self, mock_run):
        mixin = _make_mixin()
        mock_proc = MagicMock()
        mock_proc.stdout = "tool result"
        mock_run.return_value = mock_proc

        result = mixin._execute_tool("custom_tool", {"key": "val"})
        self.assertEqual(result, "tool result")
        called_cmd = mock_run.call_args[0][0]
        self.assertIn("npx", called_cmd)


if __name__ == "__main__":
    unittest.main()
