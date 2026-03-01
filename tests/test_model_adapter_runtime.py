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
