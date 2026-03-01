from unittest.mock import patch, MagicMock

from core.runtime_auth import (
    resolve_openai_api_key,
    resolve_openrouter_api_key,
    runtime_provider_status,
)


def test_resolve_openrouter_api_key_prefers_explicit_provider_key():
    mock_config = MagicMock()
    # Mock global config to return None so we can test the passed-in keys
    mock_config.get.return_value = None
    
    with patch("core.runtime_auth.config", mock_config):
        # cli_arg has highest priority
        assert resolve_openrouter_api_key(cli_arg="cli-key", config_api_key="passed-key") == "cli-key"
        # then config_api_key arg (if global config is None)
        assert resolve_openrouter_api_key(config_api_key="passed-key") == "passed-key"

def test_resolve_openrouter_api_key_prefers_global_config_over_passed_arg():
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, d=None: {"api_key": "global-config-key"}.get(k, d)
    
    with patch("core.runtime_auth.config", mock_config):
        # Implementation currently prefers global config.get("api_key") over passed config_api_key arg
        assert resolve_openrouter_api_key(config_api_key="passed-key") == "global-config-key"


def test_resolve_openrouter_api_key_falls_back_to_config():
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, d=None: {"api_key": "config-key"}.get(k, d)
    
    with patch("core.runtime_auth.config", mock_config):
        assert resolve_openrouter_api_key() == "config-key"


def test_resolve_openai_api_key_ignores_placeholders():
    mock_config = MagicMock()
    mock_config.get.return_value = "YOUR_API_KEY_HERE"

    with patch("core.runtime_auth.config", mock_config):
        assert resolve_openai_api_key() is None


def test_runtime_provider_status_tracks_chat_and_embeddings():
    mock_config = MagicMock()
    def mock_get(k, d=None):
        return {
            "openai_api_key": "openai-key",
            "api_key": None,
            "local_model_command": None,
        }.get(k, d)
    mock_config.get.side_effect = mock_get

    with patch("core.runtime_auth.config", mock_config), \
         patch("core.runtime_auth.resolve_gemini_cli_path", return_value=None):
        status = runtime_provider_status()

    assert status["openai"] is True
    assert status["chat_ready"] is True
    assert status["embedding_ready"] is True


def test_runtime_provider_status_allows_local_only_chat():
    mock_config = MagicMock()
    def mock_get(k, d=None):
        return {
            "openai_api_key": None,
            "api_key": None,
            "local_model_command": "ollama run llama2",
        }.get(k, d)
    mock_config.get.side_effect = mock_get

    with patch("core.runtime_auth.config", mock_config), \
         patch("core.runtime_auth.resolve_gemini_cli_path", return_value=None):
        status = runtime_provider_status()

    assert status["local_model"] is True
    assert status["chat_ready"] is True
    assert status["embedding_ready"] is False
