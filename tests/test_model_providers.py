"""High-coverage pytest tests for core.model_providers.ProvidersMixin.

Tests for HTTP retry logic, API provider calls (OpenAI, Anthropic, OpenRouter),
local profile dispatch, fallback mechanisms, and health tracking.
"""

from __future__ import annotations

import json
import subprocess
import time
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from core.model_adapter import ModelAdapter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mixin():
    """Return a ModelAdapter instance for testing."""
    adapter = ModelAdapter.__new__(ModelAdapter)
    adapter._local_profile_cooldowns = {}
    adapter._local_profile_cooldown_reasons = {}
    adapter._llm_timeout_override = 60
    adapter.mcp_server_url = "http://localhost:8001"
    adapter.gemini_cli_path = None
    adapter.codex_cli_path = None
    adapter.copilot_cli_path = None
    adapter._embedding_dims = None
    adapter._embedding_profile_name = None
    adapter._mem_cache = {}
    adapter.cache_db = None
    adapter.cache_ttl = 3600
    adapter._log_telemetry = MagicMock()
    return adapter


@pytest.fixture
def mock_requests():
    """Mock requests module via core.model_adapter."""
    with patch("core.model_adapter.requests") as mock_req:
        yield mock_req


@pytest.fixture
def mock_response(mock_requests):
    """Create a mock HTTP response."""
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": "test response"}}]}
    response.raise_for_status.return_value = None
    mock_requests.request.return_value = response
    mock_requests.exceptions.RequestException = Exception
    return response


@pytest.fixture
def mock_config():
    """Mock config module."""
    with patch("core.model_providers.config") as cfg:
        cfg.get.return_value = {}
        yield cfg


@pytest.fixture
def mock_auth():
    """Mock runtime_auth module with all key resolvers."""
    with patch("core.model_providers.resolve_anthropic_api_key") as mock_anthro, \
         patch("core.model_providers.resolve_openai_api_key") as mock_openai, \
         patch("core.model_providers.resolve_openrouter_api_key") as mock_router, \
         patch("core.model_providers.resolve_local_model_profiles") as mock_profiles:
        
        mock_anthro.return_value = "test-anthropic-key"
        mock_openai.return_value = "test-openai-key"
        mock_router.return_value = "test-router-key"
        mock_profiles.return_value = {}
        
        yield {
            "anthropic": mock_anthro,
            "openai": mock_openai,
            "router": mock_router,
            "profiles": mock_profiles,
        }


# =============================================================================
# TestMakeRequestWithRetries
# =============================================================================


class TestMakeRequestWithRetries:
    """Test _make_request_with_retries HTTP retry loop."""

    def test_success_first_attempt(self, mixin, mock_response, mock_requests):
        """Test successful request on first attempt."""
        mock_requests.request.return_value = mock_response
        
        result = mixin._make_request_with_retries(
            "POST", "https://api.test.com", {"key": "value"}, {"data": "payload"}
        )
        
        assert result == mock_response
        mock_requests.request.assert_called_once()

    def test_success_after_retries(self, mixin, mock_response, mock_requests):
        """Test successful request after initial failures."""
        mock_requests.exceptions.RequestException = Exception
        
        # Fail twice, then succeed
        mock_requests.request.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            mock_response,
        ]
        
        result = mixin._make_request_with_retries(
            "POST", "https://api.test.com", {}, {}, retries=3
        )
        
        assert result == mock_response
        assert mock_requests.request.call_count == 3

    def test_retries_exhausted(self, mixin, mock_requests):
        """Test raises exception when retries exhausted."""
        mock_requests.exceptions.RequestException = Exception
        mock_requests.request.side_effect = Exception("Persistent error")
        
        with pytest.raises(Exception, match="Persistent error"):
            mixin._make_request_with_retries(
                "POST", "https://api.test.com", {}, {}, retries=2
            )
        
        assert mock_requests.request.call_count == 2

    def test_backoff_timing(self, mixin, mock_response, mock_requests):
        """Test exponential backoff timing."""
        mock_requests.exceptions.RequestException = Exception
        mock_requests.request.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            mock_response,
        ]
        
        start = time.time()
        mixin._make_request_with_retries(
            "POST", "https://api.test.com", {}, {}, retries=3, backoff_factor=0.05
        )
        elapsed = time.time() - start
        
        # Expected sleep: 0.05 + 0.1 = 0.15s
        assert elapsed >= 0.1
        assert mock_requests.request.call_count == 3

    def test_custom_timeout(self, mixin, mock_response, mock_requests):
        """Test custom timeout parameter."""
        mixin._make_request_with_retries(
            "POST", "https://api.test.com", {}, {}, timeout=10
        )
        
        call_kwargs = mock_requests.request.call_args[1]
        assert call_kwargs["timeout"] == 10

    def test_retry_label_in_logging(self, mixin, mock_response, mock_requests):
        """Test retry_label is used in logging."""
        mock_requests.exceptions.RequestException = Exception
        mock_requests.request.side_effect = [
            Exception("Error"),
            mock_response,
        ]
        
        with patch("core.model_providers.log_json") as mock_log:
            mixin._make_request_with_retries(
                "POST", "https://api.test.com", {}, {}, retries=2, retry_label="test_call"
            )
            
            # Check that log_json was called with retry details
            assert mock_log.called

    def test_http_error_status_raises(self, mixin, mock_requests):
        """Test that HTTP error status codes raise an exception."""
        response = MagicMock()
        response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_requests.request.return_value = response
        mock_requests.exceptions.RequestException = Exception
        
        with pytest.raises(Exception):
            mixin._make_request_with_retries(
                "POST", "https://api.test.com", {}, {}, retries=1
            )

    def test_single_retry(self, mixin, mock_response, mock_requests):
        """Test with single retry attempt."""
        mock_requests.request.return_value = mock_response
        
        result = mixin._make_request_with_retries(
            "POST", "https://api.test.com", {}, {}, retries=1
        )
        
        assert result == mock_response
        mock_requests.request.assert_called_once()


# =============================================================================
# TestCallOpenAI
# =============================================================================


class TestCallOpenAI:
    """Test call_openai method."""

    def test_successful_call(self, mixin, mock_response, mock_requests):
        """Test successful OpenAI API call."""
        mock_requests.request.return_value = mock_response
        
        with patch("core.model_providers.resolve_openai_api_key", return_value="key"):
            result = mixin.call_openai("Generate code for X")
        
        assert result == "test response"
        call_args = mock_requests.request.call_args
        assert call_args[0][0] == "POST"
        assert "openai.com" in call_args[0][1]
        assert "gpt-4o-mini" in call_args[1]["json"]["model"]

    def test_openai_no_api_key(self, mixin, mock_requests):
        """Test raises error when OPENAI_API_KEY not set."""
        with patch("core.model_providers.resolve_openai_api_key", return_value=None), \
             patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                mixin.call_openai("test prompt")

    def test_openai_from_env(self, mixin, mock_response, mock_requests):
        """Test OpenAI key resolution from environment."""
        with patch("core.model_providers.resolve_openai_api_key", return_value=None), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            mock_requests.request.return_value = mock_response
            
            result = mixin.call_openai("test")
            
            assert result == "test response"

    def test_openai_headers(self, mixin, mock_response, mock_requests):
        """Test OpenAI headers are correct."""
        with patch("core.model_providers.resolve_openai_api_key", return_value="test-key"):
            mock_requests.request.return_value = mock_response
            
            mixin.call_openai("test prompt")
            
            call_kwargs = mock_requests.request.call_args[1]
            headers = call_kwargs["headers"]
            assert "Authorization" in headers
            assert headers["Content-Type"] == "application/json"

    def test_openai_telemetry_logging(self, mixin, mock_response, mock_requests):
        """Test telemetry is logged for OpenAI calls."""
        with patch("core.model_providers.resolve_openai_api_key", return_value="key"):
            mock_requests.request.return_value = mock_response
            mixin._log_telemetry = MagicMock()
            
            mixin.call_openai("test")
            
            mixin._log_telemetry.assert_called_once()
            call_args = mixin._log_telemetry.call_args[0]
            assert call_args[0] == "openai"

    def test_openai_payload_structure(self, mixin, mock_response, mock_requests):
        """Test OpenAI request payload structure."""
        with patch("core.model_providers.resolve_openai_api_key", return_value="key"):
            mock_requests.request.return_value = mock_response
            
            mixin.call_openai("my prompt")
            
            payload = mock_requests.request.call_args[1]["json"]
            assert payload["model"] == "gpt-4o-mini"
            assert payload["messages"][0]["role"] == "user"
            assert payload["messages"][0]["content"] == "my prompt"


# =============================================================================
# TestCallAnthropic
# =============================================================================


class TestCallAnthropic:
    """Test call_anthropic method."""

    def test_successful_call(self, mixin, mock_response, mock_requests):
        """Test successful Anthropic API call."""
        mock_response.json.return_value = {
            "content": [{"text": "anthropic response"}]
        }
        mock_requests.request.return_value = mock_response
        
        with patch("core.model_providers.resolve_anthropic_api_key", return_value="key"):
            result = mixin.call_anthropic("test prompt")
        
        assert result == "anthropic response"
        call_args = mock_requests.request.call_args
        assert "anthropic.com" in call_args[0][1]

    def test_anthropic_no_api_key(self, mixin, mock_requests):
        """Test raises error when ANTHROPIC_API_KEY not set."""
        with patch("core.model_providers.resolve_anthropic_api_key", return_value=None), \
             patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                mixin.call_anthropic("test")

    def test_anthropic_headers(self, mixin, mock_response, mock_requests):
        """Test Anthropic headers are correct."""
        mock_response.json.return_value = {"content": [{"text": "resp"}]}
        mock_requests.request.return_value = mock_response
        
        with patch("core.model_providers.resolve_anthropic_api_key", return_value="key"):
            mixin.call_anthropic("test")
            
            headers = mock_requests.request.call_args[1]["headers"]
            assert "x-api-key" in headers
            assert "anthropic-version" in headers
            assert headers["Content-Type"] == "application/json"

    def test_anthropic_payload(self, mixin, mock_response, mock_requests):
        """Test Anthropic request payload structure."""
        mock_response.json.return_value = {"content": [{"text": "resp"}]}
        mock_requests.request.return_value = mock_response
        
        with patch("core.model_providers.resolve_anthropic_api_key", return_value="key"):
            mixin.call_anthropic("my prompt")
            
            payload = mock_requests.request.call_args[1]["json"]
            assert payload["model"] == "claude-3-5-sonnet-latest"
            assert payload["max_tokens"] == 4096
            assert payload["messages"][0]["content"] == "my prompt"

    def test_anthropic_telemetry(self, mixin, mock_response, mock_requests):
        """Test telemetry logging for Anthropic calls."""
        mock_response.json.return_value = {"content": [{"text": "response"}]}
        mock_requests.request.return_value = mock_response
        mixin._log_telemetry = MagicMock()
        
        with patch("core.model_providers.resolve_anthropic_api_key", return_value="key"):
            mixin.call_anthropic("test")
            
            mixin._log_telemetry.assert_called_once()
            assert mixin._log_telemetry.call_args[0][0] == "anthropic"


# =============================================================================
# TestCallOpenRouter
# =============================================================================


class TestCallOpenRouter:
    """Test call_openrouter method."""

    def test_successful_call(self, mixin, mock_response, mock_requests, mock_config):
        """Test successful OpenRouter API call."""
        mock_requests.request.return_value = mock_response
        mock_config.get.side_effect = lambda key, default=None: {
            "model_routing": {"fast": "gpt-4"},
        }.get(key, default)
        
        with patch("core.model_providers.resolve_openrouter_api_key", return_value="key"):
            result = mixin.call_openrouter("test prompt")
        
        assert result == "test response"
        call_args = mock_requests.request.call_args
        assert "openrouter.ai" in call_args[0][1]

    def test_openrouter_no_api_key(self, mixin):
        """Test raises error when OpenRouter API key not set."""
        with patch("core.model_providers.resolve_openrouter_api_key", return_value=None), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter"):
                mixin.call_openrouter("test")

    def test_openrouter_route_key_selection(self, mixin, mock_response, mock_requests, mock_config):
        """Test model selection via route_key."""
        mock_requests.request.return_value = mock_response
        mock_config.get.side_effect = lambda key, default=None: {
            "model_routing": {"special": "custom/model-v1"},
        }.get(key, default)
        
        with patch("core.model_providers.resolve_openrouter_api_key", return_value="key"):
            mixin.call_openrouter("test", route_key="special")
            
            payload = mock_requests.request.call_args[1]["json"]
            assert payload["model"] == "custom/model-v1"

    def test_openrouter_fallback_on_error(self, mixin, mock_response, mock_requests, mock_config):
        """Test fallback model is used on primary model error."""
        error_response = MagicMock()
        error_response.json.return_value = {"error": {"message": "Model not available"}}
        success_response = MagicMock()
        success_response.json.return_value = {
            "choices": [{"message": {"content": "fallback response"}}]
        }
        
        mock_requests.request.side_effect = [error_response, success_response]
        mock_config.get.side_effect = lambda key, default=None: {
            "model_routing": {
                "fast": "unavailable/model",
                "fallback": "reliable/model",
            },
        }.get(key, default)
        
        with patch("core.model_providers.resolve_openrouter_api_key", return_value="key"):
            result = mixin.call_openrouter("test")
        
        assert result == "fallback response"
        assert mock_requests.request.call_count == 2

    def test_openrouter_fallback_failure(self, mixin, mock_requests, mock_config):
        """Test raises error when both primary and fallback fail."""
        error_response = MagicMock()
        error_response.json.return_value = {"error": {"message": "Model failed"}}
        
        mock_requests.request.return_value = error_response
        mock_config.get.side_effect = lambda key, default=None: {
            "model_routing": {"fast": "bad/model", "fallback": "bad/fallback"},
        }.get(key, default)
        
        with patch("core.model_providers.resolve_openrouter_api_key", return_value="key"):
            with pytest.raises(ValueError, match="primary and fallback"):
                mixin.call_openrouter("test")

    def test_openrouter_headers(self, mixin, mock_response, mock_requests, mock_config):
        """Test OpenRouter headers are set correctly."""
        mock_requests.request.return_value = mock_response
        mock_config.get.return_value = {}
        
        with patch("core.model_providers.resolve_openrouter_api_key", return_value="key"):
            mixin.call_openrouter("test")
            
            headers = mock_requests.request.call_args[1]["headers"]
            assert "Authorization" in headers
            assert "HTTP-Referer" in headers
            assert "X-Title" in headers


# =============================================================================
# TestCallLocal
# =============================================================================


class TestCallLocal:
    """Test call_local method."""

    def test_local_with_profile(self, mixin, mock_config):
        """Test call_local uses profile if available."""
        mixin._resolve_local_profile_name = MagicMock(return_value="default")
        mixin.call_local_profile = MagicMock(return_value="profile response")
        
        result = mixin.call_local("test prompt")
        
        assert result == "profile response"
        mixin.call_local_profile.assert_called_once_with("default", "test prompt")

    def test_local_fallback_to_command(self, mixin, mock_config):
        """Test fallback to legacy command when profile fails."""
        mixin._resolve_local_profile_name = MagicMock(return_value="profile")
        mixin.call_local_profile = MagicMock(side_effect=Exception("Profile error"))
        mock_config.get.return_value = "echo test"
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "command output"
            mock_run.return_value = result
            
            output = mixin.call_local("prompt")
        
        assert "command output" in output

    def test_local_no_profile_no_command(self, mixin, mock_config):
        """Test error message when no profile or command configured."""
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        mock_config.get.return_value = None
        
        result = mixin.call_local("test")
        
        assert "not configured" in result.lower()

    def test_local_command_execution(self, mixin, mock_config):
        """Test legacy local command execution."""
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        mock_config.get.return_value = "llama chat"
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "llama output\n"
            mock_run.return_value = result
            
            output = mixin.call_local("prompt text")
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "llama" in call_args
        assert "chat" in call_args
        assert "prompt text" in call_args

    def test_local_command_not_found(self, mixin, mock_config):
        """Test handles FileNotFoundError for missing command."""
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        mock_config.get.return_value = "nonexistent_command"
        
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = mixin.call_local("test")
        
        assert "not found" in result.lower()

    def test_local_command_failure(self, mixin, mock_config):
        """Test handles subprocess error."""
        mixin._resolve_local_profile_name = MagicMock(return_value=None)
        mock_config.get.return_value = "bad_command"
        
        error = subprocess.CalledProcessError(1, "bad", stderr="error text")
        with patch("subprocess.run", side_effect=error):
            result = mixin.call_local("test")
        
        assert "error text" in result


# =============================================================================
# TestCallLocalProfile
# =============================================================================


class TestCallLocalProfile:
    """Test local profile calling with fallbacks and cooldowns."""

    def test_successful_profile_call(self, mixin):
        """Test successful local profile call."""
        mixin._get_local_profiles = MagicMock(
            return_value={"profile1": {"provider": "openai_compatible"}}
        )
        mixin._call_local_profile_provider = MagicMock(return_value="profile output")
        
        result = mixin.call_local_profile("profile1", "test")
        
        assert result == "profile output"

    def test_unknown_profile(self, mixin):
        """Test raises error for unknown profile."""
        mixin._get_local_profiles = MagicMock(return_value={})
        
        with pytest.raises(ValueError, match="Unknown local model profile"):
            mixin.call_local_profile("unknown", "test")

    def test_profile_cooldown_active(self, mixin):
        """Test fallback when profile is in cooldown."""
        mixin._get_local_profiles = MagicMock(
            return_value={
                "cooldown_profile": {"provider": "openai_compatible", "fallback_profile": "fallback"},
                "fallback": {"provider": "openai_compatible"},
            }
        )
        mixin._local_profile_cooldowns["cooldown_profile"] = time.time() + 10
        mixin._call_local_profile_provider = MagicMock(return_value="fallback output")
        
        with patch("core.model_providers.log_json"):
            result = mixin.call_local_profile("cooldown_profile", "test")
        
        assert result == "fallback output"

    def test_profile_fallback_loop_detection(self, mixin):
        """Test detects and prevents fallback loops."""
        # Set up profiles that will cause attempted_profiles to accumulate
        # The method should raise when fallback_profile points to itself
        mixin._get_local_profiles = MagicMock(
            return_value={"p1": {"provider": "command", "command": "echo test", "fallback_profile": "p1"}}
        )
        mixin._get_local_routing = MagicMock(return_value={})
        
        # The loop detection happens in attempted_profiles set tracking
        # We need to call _call_local_profile_with_fallbacks with p1 twice
        with pytest.raises(RuntimeError, match="fallback loop"):
            mixin._call_local_profile_with_fallbacks("p1", "test", attempted_profiles={"p1"})

    def test_profile_error_triggers_fallback(self, mixin):
        """Test profile error triggers fallback."""
        mixin._get_local_profiles = MagicMock(
            return_value={
                "main": {"provider": "openai_compatible", "fallback_profile": "backup"},
                "backup": {"provider": "openai_compatible"},
            }
        )
        
        call_count = [0]
        def side_effect(profile, prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Main profile error")
            return "backup output"
        
        mixin._call_local_profile_provider = MagicMock(side_effect=side_effect)
        mixin._mark_local_profile_unhealthy = MagicMock()
        
        with patch("core.model_providers.log_json"):
            result = mixin.call_local_profile("main", "test")
        
        assert result == "backup output"
        mixin._mark_local_profile_unhealthy.assert_called_once()

    def test_cooldown_all_fallbacks_exhausted(self, mixin):
        """Test raises error when cooldown active and no fallbacks available."""
        mixin._get_local_profiles = MagicMock(
            return_value={"profile": {"provider": "openai_compatible"}}
        )
        mixin._local_profile_cooldowns["profile"] = time.time() + 10
        mixin._local_profile_cooldown_reasons["profile"] = "Previous error"
        
        with pytest.raises(RuntimeError, match="cooling down"):
            mixin.call_local_profile("profile", "test")


# =============================================================================
# TestLocalProfileProviders
# =============================================================================


class TestLocalProfileProviders:
    """Test local profile provider dispatch."""

    def test_openai_compatible_dispatch(self, mixin, mock_response):
        """Test openai_compatible provider dispatch."""
        profile = {
            "provider": "openai_compatible",
            "base_url": "http://localhost:8080/v1",
            "model": "llama-7b",
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "local response"}}]
        }
        
        result = mixin._call_local_profile_provider(profile, "test prompt")
        
        assert result == "local response"

    def test_ollama_dispatch(self, mixin, mock_response):
        """Test ollama provider dispatch."""
        profile = {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama2",
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {"response": "ollama output"}
        
        result = mixin._call_local_profile_provider(profile, "test")
        
        assert result == "ollama output"

    def test_command_dispatch(self, mixin):
        """Test command provider dispatch."""
        profile = {
            "provider": "command",
            "command": "llama-cli",
        }
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "command output"
            mock_run.return_value = result
            
            output = mixin._call_local_profile_provider(profile, "test")
        
        assert "command output" in output

    def test_unsupported_provider(self, mixin):
        """Test raises error for unsupported provider."""
        profile = {"provider": "unknown_provider"}
        
        with pytest.raises(ValueError, match="Unsupported"):
            mixin._call_local_profile_provider(profile, "test")


# =============================================================================
# TestOpenAICompatibleLocal
# =============================================================================


class TestOpenAICompatibleLocal:
    """Test local OpenAI-compatible provider calls."""

    def test_successful_call(self, mixin, mock_response):
        """Test successful OpenAI-compatible local call."""
        profile = {
            "base_url": "http://localhost:8080/v1",
            "model": "mistral",
            "temperature": 0.5,
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        
        result = mixin._call_local_openai_compatible(profile, "test prompt")
        
        assert result == "response"

    def test_missing_model_raises(self, mixin):
        """Test raises error when model is missing."""
        profile = {"base_url": "http://localhost:8080/v1"}
        
        with pytest.raises(ValueError, match="model"):
            mixin._call_local_openai_compatible(profile, "test")

    def test_api_key_header(self, mixin, mock_response):
        """Test API key is added to headers."""
        profile = {
            "base_url": "http://localhost:8080/v1",
            "model": "llama",
            "api_key": "secret-key",
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "resp"}}]
        }
        
        mixin._call_local_openai_compatible(profile, "test")
        
        # Call signature: (method, url, headers, payload, ...)
        call_args = mixin._make_request_with_retries.call_args[0]
        headers = call_args[2]
        assert "Authorization" in headers
        assert "secret-key" in headers["Authorization"]

    def test_extra_body_merged(self, mixin, mock_response):
        """Test extra_body parameters are merged into payload."""
        profile = {
            "base_url": "http://localhost:8080/v1",
            "model": "llama",
            "extra_body": {"top_p": 0.9, "frequency_penalty": 0.5},
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "resp"}}]
        }
        
        mixin._call_local_openai_compatible(profile, "test")
        
        # Call signature: (method, url, headers, payload, ...)
        call_args = mixin._make_request_with_retries.call_args[0]
        payload = call_args[3]
        assert payload["top_p"] == 0.9
        assert payload["frequency_penalty"] == 0.5


# =============================================================================
# TestOllamaLocal
# =============================================================================


class TestOllamaLocal:
    """Test local Ollama provider calls."""

    def test_successful_call(self, mixin, mock_response):
        """Test successful Ollama call."""
        profile = {
            "base_url": "http://localhost:11434",
            "model": "llama2",
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {"response": "ollama output"}
        
        result = mixin._call_local_ollama(profile, "test prompt")
        
        assert result == "ollama output"

    def test_missing_model_raises(self, mixin):
        """Test raises error when model is missing."""
        profile = {"base_url": "http://localhost:11434"}
        
        with pytest.raises(ValueError, match="model"):
            mixin._call_local_ollama(profile, "test")

    def test_temperature_in_options(self, mixin, mock_response):
        """Test temperature is added to options."""
        profile = {
            "base_url": "http://localhost:11434",
            "model": "llama2",
            "temperature": 0.7,
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {"response": "resp"}
        
        mixin._call_local_ollama(profile, "test")
        
        # Call signature: (method, url, headers, payload, ...)
        call_args = mixin._make_request_with_retries.call_args[0]
        payload = call_args[3]
        assert "options" in payload
        assert payload["options"]["temperature"] == 0.7

    def test_max_tokens_to_num_predict(self, mixin, mock_response):
        """Test max_tokens is converted to num_predict for Ollama."""
        profile = {
            "base_url": "http://localhost:11434",
            "model": "llama2",
            "max_tokens": 1000,
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {"response": "resp"}
        
        mixin._call_local_ollama(profile, "test")
        
        # Call signature: (method, url, headers, payload, ...)
        call_args = mixin._make_request_with_retries.call_args[0]
        payload = call_args[3]
        assert payload["options"]["num_predict"] == 1000

    def test_system_prompt(self, mixin, mock_response):
        """Test system prompt is added to payload."""
        profile = {
            "base_url": "http://localhost:11434",
            "model": "llama2",
            "system": "You are a helpful assistant.",
        }
        mixin._make_request_with_retries = MagicMock(return_value=mock_response)
        mock_response.json.return_value = {"response": "resp"}
        
        mixin._call_local_ollama(profile, "test")
        
        # Call signature: (method, url, headers, payload, ...)
        call_args = mixin._make_request_with_retries.call_args[0]
        payload = call_args[3]
        assert payload["system"] == "You are a helpful assistant."


# =============================================================================
# TestCommandLocal
# =============================================================================


class TestCommandLocal:
    """Test local command provider calls."""

    def test_successful_command_call(self, mixin):
        """Test successful command execution."""
        profile = {"command": "echo hello"}
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "hello\n"
            mock_run.return_value = result
            
            output = mixin._call_local_command_profile(profile, "test prompt")
        
        assert output == "hello"

    def test_prompt_replacement_in_command(self, mixin):
        """Test {prompt} is replaced in command."""
        profile = {"command": "llama -p {prompt}"}
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "output"
            mock_run.return_value = result
            
            mixin._call_local_command_profile(profile, "my test")
        
        call_args = mock_run.call_args[0][0]
        # Check that {prompt} was replaced with quoted prompt
        assert any("my test" in arg for arg in call_args)

    def test_command_as_list(self, mixin):
        """Test command as list with prompt template."""
        profile = {"command": ["python", "-c", "print('{prompt}')"]}
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "test"
            mock_run.return_value = result
            
            mixin._call_local_command_profile(profile, "my prompt")
        
        call_args = mock_run.call_args[0][0]
        # Check that {prompt} was replaced with quoted version
        assert "my prompt" in call_args[2]

    def test_stdin_input(self, mixin):
        """Test prompt passed via stdin when no {prompt} in command."""
        profile = {"command": "cat"}
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "output"
            mock_run.return_value = result
            
            mixin._call_local_command_profile(profile, "test input")
        
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["input"] == "test input"

    def test_invalid_command_format(self, mixin):
        """Test raises error for invalid command format."""
        profile = {"command": 123}  # Invalid: not string or list
        
        with pytest.raises(ValueError, match="command"):
            mixin._call_local_command_profile(profile, "test")


# =============================================================================
# TestLocalProfileHealthTracking
# =============================================================================


class TestLocalProfileHealthTracking:
    """Test health tracking and cooldown management."""

    def test_mark_profile_unhealthy(self, mixin):
        """Test marking profile as unhealthy starts cooldown."""
        profile = {"cooldown_seconds": 5}
        
        with patch("core.model_providers.log_json"):
            mixin._mark_local_profile_unhealthy("test_profile", profile, Exception("Error"))
        
        assert "test_profile" in mixin._local_profile_cooldowns
        remaining = mixin._local_profile_cooldown_remaining("test_profile")
        assert remaining > 0

    def test_cooldown_remaining(self, mixin):
        """Test cooldown_remaining calculation."""
        mixin._local_profile_cooldowns["profile"] = time.time() + 10
        
        remaining = mixin._local_profile_cooldown_remaining("profile")
        
        assert 9 < remaining <= 10

    def test_cooldown_expired(self, mixin):
        """Test expired cooldown returns 0."""
        mixin._local_profile_cooldowns["profile"] = time.time() - 1
        
        remaining = mixin._local_profile_cooldown_remaining("profile")
        
        assert remaining == 0

    def test_clear_unhealthy(self, mixin):
        """Test clearing unhealthy status."""
        mixin._local_profile_cooldowns["profile"] = time.time() + 10
        mixin._local_profile_cooldown_reasons["profile"] = "Test error"
        
        mixin._clear_local_profile_unhealthy("profile")
        
        assert "profile" not in mixin._local_profile_cooldowns
        assert "profile" not in mixin._local_profile_cooldown_reasons


# =============================================================================
# TestProfileHelpers
# =============================================================================


class TestProfileHelpers:
    """Test profile configuration helper methods."""

    def test_profile_timeout_default(self, mixin):
        """Test timeout with default value."""
        profile = {}
        
        result = mixin._profile_timeout(profile, key="timeout_seconds", default=30.0)
        
        assert result == 30.0

    def test_profile_timeout_custom(self, mixin):
        """Test custom timeout value."""
        profile = {"timeout_seconds": 90}
        
        result = mixin._profile_timeout(profile, key="timeout_seconds", default=30.0)
        
        assert result == 90.0

    def test_profile_timeout_invalid(self, mixin):
        """Test invalid timeout falls back to default."""
        profile = {"timeout_seconds": "invalid"}
        
        result = mixin._profile_timeout(profile, key="timeout_seconds", default=30.0)
        
        assert result == 30.0

    def test_profile_timeout_zero_uses_default(self, mixin):
        """Test zero timeout uses default."""
        profile = {"timeout_seconds": 0}
        
        result = mixin._profile_timeout(profile, key="timeout_seconds", default=30.0)
        
        assert result == 30.0

    def test_profile_retries_default(self, mixin):
        """Test default retry count."""
        profile = {}
        
        result = mixin._profile_retries(profile)
        
        assert result == 3

    def test_profile_retries_custom(self, mixin):
        """Test custom retry count."""
        profile = {"retries": 5}
        
        result = mixin._profile_retries(profile)
        
        assert result == 5

    def test_profile_retries_invalid(self, mixin):
        """Test invalid retries falls back to default."""
        profile = {"retries": "invalid"}
        
        result = mixin._profile_retries(profile)
        
        assert result == 3

    def test_profile_backoff_default(self, mixin):
        """Test default backoff factor."""
        profile = {}
        
        result = mixin._profile_backoff(profile)
        
        assert result == 0.5

    def test_profile_backoff_custom(self, mixin):
        """Test custom backoff factor."""
        profile = {"backoff_factor": 0.3}
        
        result = mixin._profile_backoff(profile)
        
        assert result == 0.3

    def test_profile_cooldown_seconds(self, mixin):
        """Test cooldown seconds extraction."""
        profile = {"cooldown_seconds": 15.0}
        
        result = mixin._profile_cooldown_seconds(profile)
        
        assert result == 15.0

    def test_profile_fallbacks_single(self, mixin):
        """Test single fallback_profile."""
        profile = {"fallback_profile": "backup"}
        
        result = mixin._profile_fallbacks(profile)
        
        assert result == ["backup"]

    def test_profile_fallbacks_multiple(self, mixin):
        """Test multiple fallback_profiles."""
        profile = {"fallback_profiles": ["backup1", "backup2"]}
        
        result = mixin._profile_fallbacks(profile)
        
        assert result == ["backup1", "backup2"]

    def test_profile_fallbacks_empty(self, mixin):
        """Test no fallbacks."""
        profile = {}
        
        result = mixin._profile_fallbacks(profile)
        
        assert result == []

    def test_profile_fallbacks_invalid_type(self, mixin):
        """Test invalid fallback type returns empty list."""
        profile = {"fallback_profiles": "invalid"}
        
        result = mixin._profile_fallbacks(profile)
        
        assert result == []


# =============================================================================
# TestLocalProfileResolution
# =============================================================================


class TestLocalProfileResolution:
    """Test local profile name resolution."""

    def test_get_local_profiles(self, mixin):
        """Test get_local_profiles calls resolver."""
        with patch("core.model_providers.resolve_local_model_profiles") as mock_resolve:
            mock_resolve.return_value = {"profile1": {}}
            
            result = mixin._get_local_profiles()
            
            assert result == {"profile1": {}}
            mock_resolve.assert_called_once()

    def test_get_local_routing(self, mixin, mock_config):
        """Test get_local_routing from config."""
        mock_config.get.return_value = {"embedding": "emb_profile", "fast": "fast_profile"}
        
        result = mixin._get_local_routing()
        
        assert result["embedding"] == "emb_profile"

    def test_resolve_local_profile_name(self, mixin):
        """Test resolving profile name by route key."""
        mixin._get_local_routing = MagicMock(return_value={"embedding": "emb_profile"})
        mixin._get_local_profiles = MagicMock(return_value={"emb_profile": {}})
        
        result = mixin._resolve_local_profile_name("embedding")
        
        assert result == "emb_profile"

    def test_resolve_profile_not_found(self, mixin):
        """Test returns None when profile not found."""
        mixin._get_local_routing = MagicMock(return_value={"embedding": "missing"})
        mixin._get_local_profiles = MagicMock(return_value={})
        
        result = mixin._resolve_local_profile_name("embedding")
        
        assert result is None

    def test_resolve_profile_route_key_not_mapped(self, mixin):
        """Test returns None when route key not in routing."""
        mixin._get_local_routing = MagicMock(return_value={})
        mixin._get_local_profiles = MagicMock(return_value={"profile1": {}})
        
        result = mixin._resolve_local_profile_name("unknown")
        
        assert result is None


# =============================================================================
# TestCallCliMethods (Gemini, Codex, Copilot)
# =============================================================================


class TestCallCliMethods:
    """Test CLI-based provider calls (Gemini, Codex, Copilot)."""

    def test_call_gemini_success(self, mixin):
        """Test successful Gemini CLI call."""
        mixin.gemini_cli_path = "/usr/bin/gemini"
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "gemini response\n"
            mock_run.return_value = result
            
            output = mixin.call_gemini("test prompt")
        
        assert output == "gemini response"

    def test_call_gemini_no_path(self, mixin):
        """Test raises error when Gemini CLI not configured."""
        mixin.gemini_cli_path = None
        
        with pytest.raises(ValueError, match="Gemini CLI"):
            mixin.call_gemini("test")

    def test_call_codex_success(self, mixin):
        """Test successful Codex CLI call."""
        mixin.codex_cli_path = "/usr/bin/codex"
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "codex response\n"
            mock_run.return_value = result
            
            output = mixin.call_codex("test prompt")
        
        assert output == "codex response"

    def test_call_codex_no_path(self, mixin):
        """Test raises error when Codex CLI not configured."""
        mixin.codex_cli_path = None
        
        with pytest.raises(ValueError, match="Codex CLI"):
            mixin.call_codex("test")

    def test_call_copilot_success(self, mixin):
        """Test successful Copilot CLI call."""
        mixin.copilot_cli_path = "/usr/bin/copilot"
        
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.stdout = "copilot response\n"
            mock_run.return_value = result
            
            output = mixin.call_copilot("test prompt")
        
        assert output == "copilot response"

    def test_call_copilot_no_path(self, mixin):
        """Test raises error when Copilot CLI not configured."""
        mixin.copilot_cli_path = None
        
        with pytest.raises(ValueError, match="Copilot CLI"):
            mixin.call_copilot("test")

    def test_cli_error_handling(self, mixin):
        """Test CLI error handling."""
        mixin.gemini_cli_path = "/usr/bin/gemini"
        
        error = subprocess.CalledProcessError(1, "gemini", stderr="CLI error")
        with patch("subprocess.run", side_effect=error):
            with pytest.raises(RuntimeError, match="failed"):
                mixin.call_gemini("test")
