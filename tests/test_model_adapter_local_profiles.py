from unittest.mock import MagicMock, patch

from core.model_adapter import ModelAdapter


def _config_get_with_profiles(key, default=None):
    if key == "semantic_memory":
        return {"embedding_model": "text-embedding-3-small"}
    if key == "local_model_profiles":
        return {
            "android_coder": {
                "provider": "openai_compatible",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "qwen2.5-coder-3b",
                "temperature": 0.1,
                "max_tokens": 256,
            },
            "android_planner": {
                "provider": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "model": "phi4-mini",
                "temperature": 0.2,
                "max_tokens": 128,
            },
        }
    if key == "local_model_routing":
        return {
            "code_generation": "android_coder",
            "planning": "android_planner",
            "analysis": "android_planner",
            "critique": "android_planner",
            "quality": "android_planner",
            "fast": "android_coder",
        }
    return default


def test_respond_for_role_uses_openai_compatible_profile():
    with patch("core.model_adapter.config.get", side_effect=_config_get_with_profiles), patch("core.model_adapter.log_json"), patch.object(ModelAdapter, "_make_request_with_retries") as mock_request:
        adapter = ModelAdapter()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": '{"aura_target":"core/demo.py","code":"print(1)"}'}}]}
        mock_request.return_value = mock_response

        result = adapter.respond_for_role("code_generation", "write code")

    assert result == '{"aura_target":"core/demo.py","code":"print(1)"}'
    payload = mock_request.call_args.args[3]
    assert payload["model"] == "qwen2.5-coder-3b"
    assert payload["max_tokens"] == 256


def test_respond_for_role_uses_ollama_profile():
    with patch("core.model_adapter.config.get", side_effect=_config_get_with_profiles), patch("core.model_adapter.log_json"), patch.object(ModelAdapter, "_make_request_with_retries") as mock_request:
        adapter = ModelAdapter()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '["step 1", "step 2"]'}
        mock_request.return_value = mock_response

        result = adapter.respond_for_role("planning", "plan the task")

    assert result == '["step 1", "step 2"]'
    payload = mock_request.call_args.args[3]
    assert payload["model"] == "phi4-mini"
    assert payload["options"]["num_predict"] == 128


def test_respond_for_role_falls_back_to_default_path_when_profile_missing():
    def _config_get(key, default=None):
        if key == "semantic_memory":
            return {"embedding_model": "text-embedding-3-small"}
        if key == "local_model_profiles":
            return {}
        if key == "local_model_routing":
            return {"planning": "missing_profile"}
        return default

    with patch("core.model_adapter.config.get", side_effect=_config_get), patch("core.model_adapter.log_json"), patch.object(ModelAdapter, "respond", return_value='["fallback"]') as mock_respond:
        adapter = ModelAdapter()
        result = adapter.respond_for_role("planning", "plan it")

    assert result == '["fallback"]'
    mock_respond.assert_called_once_with("plan it")


def test_local_profile_failure_uses_configured_fallback_profile():
    def _config_get(key, default=None):
        if key == "semantic_memory":
            return {"embedding_model": "text-embedding-3-small"}
        if key == "local_model_profiles":
            return {
                "android_coder": {
                    "provider": "openai_compatible",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "qwen2.5-coder-3b",
                },
                "android_planner": {
                    "provider": "openai_compatible",
                    "base_url": "http://127.0.0.1:8081/v1",
                    "model": "phi4-mini",
                    "fallback_profiles": ["android_coder"],
                    "cooldown_seconds": 15,
                },
            }
        if key == "local_model_routing":
            return {
                "planning": "android_planner",
                "fast": "android_coder",
            }
        return default

    seen_models = []

    def _call(profile, prompt):
        seen_models.append(profile.get("model"))
        if profile.get("model") == "phi4-mini":
            raise RuntimeError("planner endpoint down")
        return '["fallback step"]'

    with patch("core.model_adapter.config.get", side_effect=_config_get), patch("core.model_adapter.log_json"), patch.object(ModelAdapter, "_call_local_openai_compatible", side_effect=_call):
        adapter = ModelAdapter()
        result = adapter.respond_for_role("planning", "plan the task")

    assert result == '["fallback step"]'
    assert seen_models == ["phi4-mini", "qwen2.5-coder-3b"]
    assert "android_planner" in adapter._local_profile_cooldowns


def test_local_profile_cooldown_skips_failed_profile_on_next_call():
    def _config_get(key, default=None):
        if key == "semantic_memory":
            return {"embedding_model": "text-embedding-3-small"}
        if key == "local_model_profiles":
            return {
                "android_coder": {
                    "provider": "openai_compatible",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "qwen2.5-coder-3b",
                },
                "android_planner": {
                    "provider": "openai_compatible",
                    "base_url": "http://127.0.0.1:8081/v1",
                    "model": "phi4-mini",
                    "fallback_profiles": ["android_coder"],
                    "cooldown_seconds": 30,
                },
            }
        if key == "local_model_routing":
            return {
                "planning": "android_planner",
                "fast": "android_coder",
            }
        return default

    seen_models = []

    def _call(profile, prompt):
        seen_models.append(profile.get("model"))
        if profile.get("model") == "phi4-mini":
            raise RuntimeError("planner endpoint down")
        return '["fallback step"]'

    with patch("core.model_adapter.config.get", side_effect=_config_get), patch("core.model_adapter.log_json"), patch.object(ModelAdapter, "_call_local_openai_compatible", side_effect=_call):
        adapter = ModelAdapter()
        first = adapter.respond_for_role("planning", "plan the task")
        second = adapter.respond_for_role("planning", "plan the task again")

    assert first == '["fallback step"]'
    assert second == '["fallback step"]'
    assert seen_models == ["phi4-mini", "qwen2.5-coder-3b", "qwen2.5-coder-3b"]
