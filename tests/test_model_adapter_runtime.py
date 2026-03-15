import concurrent.futures
import json
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
        if key == "openai_api_key":
            return "openai-key"
        # Fallback to defaults or simulated values for other keys
        if key == "model_name":
            return "test-model"
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


class FakeFuture:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.cancel_called = False

    def result(self, timeout=None):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        self.cancel_called = True
        return True


class FakeExecutor:
    def __init__(self, futures):
        self._futures = list(futures)
        self.submit_calls = []
        self.shutdown_calls = []

    def submit(self, fn, *args):
        self.submit_calls.append((fn, args))
        return self._futures.pop(0)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


def test_respond_reuses_one_timeout_executor_across_fallbacks():
    adapter = ModelAdapter()
    timeout_future = FakeFuture(error=concurrent.futures.TimeoutError())
    success_future = FakeFuture(result="ok")
    executor = FakeExecutor([timeout_future, success_future])

    with patch.object(adapter, "_new_timeout_executor", return_value=executor) as mock_new, \
         patch.object(adapter, "_save_to_cache"):
        response = adapter.respond("prompt")

    assert response == "ok"
    assert mock_new.call_count == 1
    assert [call[0].__name__ for call in executor.submit_calls] == [
        "call_openai",
        "call_openrouter",
    ]
    assert timeout_future.cancel_called is True
    assert executor.shutdown_calls == [(False, True)]


def test_call_with_timeout_closes_owned_executor_without_waiting():
    adapter = ModelAdapter()
    future = FakeFuture(result="ok")
    executor = FakeExecutor([future])

    with patch.object(adapter, "_new_timeout_executor", return_value=executor):
        response = adapter._call_with_timeout(lambda: "ignored")

    assert response == "ok"
    assert executor.shutdown_calls == [(False, True)]


def test_execute_tool_routes_bound_tool_to_named_mcp_server():
    adapter = ModelAdapter()
    adapter._mcp_tool_bindings = {"get_repo": {"server": "copilot"}}
    response = MagicMock()
    response.json.return_value = {"result": {"ok": True}}
    requests_mod = MagicMock()
    requests_mod.post.return_value = response
    requests_mod.exceptions.RequestException = requests.exceptions.RequestException

    with patch("core.model_adapter._require_requests", return_value=requests_mod), \
         patch("core.model_adapter.config.get_mcp_server_port", return_value=8007), \
         patch("core.model_adapter.log_json"):
        payload = json.loads(
            adapter._execute_tool("get_repo", {"owner": "octo", "repo": "aura"})
        )

    requests_mod.post.assert_called_once_with(
        "http://localhost:8007/call",
        json={
            "tool_name": "get_repo",
            "args": {"owner": "octo", "repo": "aura"},
        },
        timeout=60,
    )
    assert payload["result"]["ok"] is True


def test_execute_tool_returns_binding_error_for_unknown_server():
    adapter = ModelAdapter()
    adapter._mcp_tool_bindings = {"get_repo": {"server": "missing"}}

    with patch(
        "core.model_adapter.config.get_mcp_server_port",
        side_effect=ValueError("Unknown MCP server name 'missing'"),
    ):
        result = adapter._execute_tool("get_repo", {"owner": "octo"})

    assert "binding error" in result
    assert "missing" in result
