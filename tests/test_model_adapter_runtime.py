from unittest.mock import MagicMock, patch

import requests

from core.model_adapter import ModelAdapter


def test_model_adapter_normalizes_openai_embedding_model_from_config():
    def mock_get(key, default=None):
        if key == "semantic_memory":
            return {"embedding_model": "openai/text-embedding-3-small"}
        return default

    with patch("core.model_adapter.config.get", side_effect=mock_get), \
         patch("core.model_adapter.log_json"):
        adapter = ModelAdapter()
        assert adapter._embedding_model == "text-embedding-3-small"


def test_embed_disables_remote_provider_after_failure(monkeypatch):
    import requests
    from unittest.mock import MagicMock
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    
    response = MagicMock(status_code=400)
    error = requests.exceptions.HTTPError("bad request")
    error.response = response
    
    def mock_get(key, default=None):
        if key == "semantic_memory":
            return {"embedding_model": "text-embedding-3-small"}
        return default

    with patch("core.model_adapter.log_json"), \
         patch("core.model_adapter.resolve_openai_api_key", return_value="openai-key"), \
         patch("core.model_adapter.config.get", side_effect=mock_get):
        adapter = ModelAdapter()

        with patch("core.model_adapter.time.sleep"), \
             patch.object(adapter, "_make_request_with_retries", side_effect=error) as mock_request:
            first = adapter.embed(["alpha"])
            second = adapter.embed(["beta"])

        assert len(first) == 1
        assert len(second) == 1
        assert first[0].shape == second[0].shape == (adapter.dimensions(),)
        assert adapter._embedding_disabled is True
        assert mock_request.call_count == 1


def test_model_adapter_uses_local_embedding_profile():
    def mock_get(key, default=None):
        if key == "semantic_memory":
            return {"embedding_model": "local_profile:android_embeddings"}
        if key == "local_model_profiles":
            return {
                "android_embeddings": {
                    "provider": "openai_compatible",
                    "base_url": "http://127.0.0.1:8082/v1",
                    "embedding_model": "bge-small",
                    "embedding_dims": 3,
                }
            }
        if key == "local_model_routing":
            return {"embedding": "android_embeddings"}
        if key == "model_routing":
            return {"embedding": "openai/text-embedding-3-small"}
        return default

    with patch("core.model_adapter.config.get", side_effect=mock_get), \
         patch("core.model_adapter.log_json"), \
         patch.object(ModelAdapter, "_make_request_with_retries") as mock_request:
        adapter = ModelAdapter()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_request.return_value = mock_response

        vectors = adapter.embed(["alpha", "beta"])

    assert adapter.model_id() == "bge-small"
    assert adapter.dimensions() == 3
    assert len(vectors) == 2
    assert vectors[0].shape == (3,)
    payload = mock_request.call_args.args[3]
    assert payload["model"] == "bge-small"


def test_model_adapter_uses_builtin_local_embeddings():
    def mock_get(key, default=None):
        if key == "semantic_memory":
            return {"embedding_model": "local-tfidf-svd-50d"}
        if key == "model_routing":
            return {"embedding": "openai/text-embedding-3-small"}
        return default

    with patch("core.model_adapter.config.get", side_effect=mock_get), \
         patch("core.model_adapter.log_json"):
        adapter = ModelAdapter()
        vectors = adapter.embed(["alpha", "beta"])

    assert adapter.model_id() == "local-tfidf-svd-50d"
    assert adapter.dimensions() == 50
    assert len(vectors) == 2
    assert vectors[0].shape == (50,)
