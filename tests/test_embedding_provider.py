"""Tests for memory/embedding_provider.py — EmbeddingResult, LocalEmbeddingProvider, OpenAIEmbeddingProvider."""

import json
import pytest
from unittest.mock import MagicMock, patch

from memory.embedding_provider import EmbeddingResult, LocalEmbeddingProvider, OpenAIEmbeddingProvider


class TestEmbeddingResult:
    def test_dataclass_fields(self):
        result = EmbeddingResult(
            vector=[0.1, 0.2],
            model_name="test-model",
            model_version="1.0",
            provider_type="local",
            dimensions=2,
        )
        assert result.vector == [0.1, 0.2]
        assert result.model_name == "test-model"
        assert result.model_version == "1.0"
        assert result.provider_type == "local"
        assert result.dimensions == 2


class TestLocalEmbeddingProvider:
    def test_dimensions_default(self):
        provider = LocalEmbeddingProvider()
        assert provider.dimensions() == 50

    def test_dimensions_custom(self):
        provider = LocalEmbeddingProvider(dims=128)
        assert provider.dimensions() == 128

    def test_embed_returns_list_of_lists(self):
        provider = LocalEmbeddingProvider(dims=10)
        result = provider.embed(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 10
        assert len(result[1]) == 10

    def test_embed_deterministic(self):
        provider = LocalEmbeddingProvider(dims=20)
        v1 = provider.embed(["same text"])[0]
        v2 = provider.embed(["same text"])[0]
        assert v1 == v2

    def test_embed_different_texts_differ(self):
        provider = LocalEmbeddingProvider(dims=20)
        v1 = provider.embed(["text one"])[0]
        v2 = provider.embed(["text two"])[0]
        assert v1 != v2

    def test_embed_empty_string(self):
        provider = LocalEmbeddingProvider(dims=10)
        result = provider.embed([""])
        assert len(result) == 1
        assert len(result[0]) == 10

    def test_embed_text_returns_embedding_result(self):
        provider = LocalEmbeddingProvider(dims=16)
        result = provider.embed_text("hello world")
        assert isinstance(result, EmbeddingResult)
        assert len(result.vector) == 16
        assert result.model_name == "local-sha256"
        assert result.model_version == "1.0"
        assert result.provider_type == "local"
        assert result.dimensions == 16

    def test_embed_text_deterministic(self):
        provider = LocalEmbeddingProvider(dims=16)
        r1 = provider.embed_text("same")
        r2 = provider.embed_text("same")
        assert r1.vector == r2.vector

    def test_embed_batch_length(self):
        provider = LocalEmbeddingProvider(dims=8)
        texts = ["a", "b", "c", "d"]
        results = provider.embed_batch(texts)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, EmbeddingResult)
            assert len(r.vector) == 8

    def test_embed_batch_empty(self):
        provider = LocalEmbeddingProvider()
        results = provider.embed_batch([])
        assert results == []

    def test_healthcheck_true(self):
        provider = LocalEmbeddingProvider()
        assert provider.healthcheck() is True

    def test_class_attributes(self):
        assert LocalEmbeddingProvider.model_name == "local-sha256"
        assert LocalEmbeddingProvider.model_version == "1.0"
        assert LocalEmbeddingProvider.provider_type == "local"


class TestOpenAIEmbeddingProvider:
    def _make_api_response(self, vectors):
        """Build a mock OpenAI API response payload."""
        data = [{"index": i, "embedding": vec} for i, vec in enumerate(vectors)]
        return {"data": data}

    def test_model_name_property(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-3-small")
        assert provider.model_name == "text-embedding-3-small"

    def test_dimensions_small_model(self):
        provider = OpenAIEmbeddingProvider(api_key="k", model="text-embedding-3-small")
        assert provider.dimensions() == 1536

    def test_dimensions_large_model(self):
        provider = OpenAIEmbeddingProvider(api_key="k", model="text-embedding-3-large")
        assert provider.dimensions() == 3072

    def test_embed_text_calls_api(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        fake_vector = [0.1, 0.2, 0.3]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_api_response([fake_vector])
        mock_resp.raise_for_status = MagicMock()

        with patch("memory.embedding_provider._requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = provider.embed_text("hello world")

        assert isinstance(result, EmbeddingResult)
        assert result.vector == fake_vector
        assert result.provider_type == "openai"
        assert result.model_name == "text-embedding-3-small"
        assert result.dimensions == 3

        # Verify API call payload
        call_kwargs = mock_requests.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["input"] == ["hello world"]
        assert payload["model"] == "text-embedding-3-small"
        # Authorization header
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"

    def test_embed_batch_calls_api(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_api_response(vectors)
        mock_resp.raise_for_status = MagicMock()

        with patch("memory.embedding_provider._requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            results = provider.embed_batch(["text one", "text two"])

        assert len(results) == 2
        assert results[0].vector == [0.1, 0.2]
        assert results[1].vector == [0.3, 0.4]
        for r in results:
            assert isinstance(r, EmbeddingResult)
            assert r.provider_type == "openai"

    def test_embed_compat_shim_returns_list_of_lists(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        vectors = [[0.5, 0.6]]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_api_response(vectors)
        mock_resp.raise_for_status = MagicMock()

        with patch("memory.embedding_provider._requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = provider.embed(["text"])

        assert result == [[0.5, 0.6]]

    def test_embed_batch_preserves_order(self):
        """API may return embeddings out of order by index; verify sorting."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        # Return in reverse order
        out_of_order_data = {
            "data": [
                {"index": 1, "embedding": [0.9, 0.9]},
                {"index": 0, "embedding": [0.1, 0.1]},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = out_of_order_data
        mock_resp.raise_for_status = MagicMock()

        with patch("memory.embedding_provider._requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            results = provider.embed_batch(["first", "second"])

        assert results[0].vector == [0.1, 0.1]
        assert results[1].vector == [0.9, 0.9]

    def test_healthcheck_success(self):
        provider = OpenAIEmbeddingProvider(api_key="k")
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_api_response([[0.1]])
        mock_resp.raise_for_status = MagicMock()

        with patch("memory.embedding_provider._requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            assert provider.healthcheck() is True

    def test_healthcheck_failure(self):
        provider = OpenAIEmbeddingProvider(api_key="k")
        with patch("memory.embedding_provider._requests") as mock_requests:
            mock_requests.post.side_effect = Exception("network error")
            assert provider.healthcheck() is False

    def test_provider_type(self):
        provider = OpenAIEmbeddingProvider(api_key="k")
        assert provider.provider_type == "openai"
