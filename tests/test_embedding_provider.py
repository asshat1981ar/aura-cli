import pytest
import numpy as np
from unittest.mock import MagicMock
from memory.embedding_provider import LocalEmbeddingProvider, OpenAIEmbeddingProvider

def test_local_embedding_provider_random_fallback():
    # Force sklearn missing behavior by checking internal flag if possible,
    # or just relying on the environment.
    # In this environment, sklearn is missing, so it should use random vectors.
    provider = LocalEmbeddingProvider()
    
    # Check basic properties
    assert provider.available()
    model_id = provider.model_id()
    assert model_id == "local-random" or model_id == "local-tfidf-svd-50d"
    assert provider.dimensions() == 50
    
    # Test embedding consistency
    text1 = "hello world"
    vec1 = provider.embed([text1])[0]
    vec2 = provider.embed([text1])[0]
    
    assert len(vec1) == 50
    assert np.allclose(vec1, vec2)
    
    # Test different texts produce different vectors (high probability)
    text2 = "goodbye world"
    vec3 = provider.embed([text2])[0]
    assert not np.allclose(vec1, vec3)

def test_openai_embedding_provider():
    mock_adapter = MagicMock()
    mock_adapter.embed.return_value = [np.array([0.1, 0.2, 0.3])]
    
    provider = OpenAIEmbeddingProvider(mock_adapter)
    
    assert provider.available()
    assert provider.model_id() == "text-embedding-3-small"
    assert provider.dimensions() == 1536
    
    vecs = provider.embed(["test"])
    assert len(vecs) == 1
    assert vecs[0] == [0.1, 0.2, 0.3]
    
    mock_adapter.embed.assert_called_with(["test"])

def test_provider_batching():
    provider = LocalEmbeddingProvider()
    texts = ["a", "b", "c"]
    vecs = provider.embed(texts)
    assert len(vecs) == 3
    assert len(vecs[0]) == 50
