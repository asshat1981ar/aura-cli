"""ONNX Runtime Mobile embedding provider for local, private inference.

Optimized for Android (NNAPI) and iOS (CoreML) with INT8 quantized models.
Includes automatic model downloading and caching from Hugging Face.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Mapping

import numpy as np
from .embedding_provider import EmbeddingProvider, EmbeddingResult, _MissingPackage

logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    import onnxruntime as ort
except ImportError:
    ort = _MissingPackage("onnxruntime-mobile")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = _MissingPackage("huggingface_hub")

try:
    from tokenizers import Tokenizer
except ImportError:
    tokenizer = _MissingPackage("tokenizers")


class ONNXEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using ONNX Runtime Mobile.

    Default model: Xenova/all-MiniLM-L6-v2 (INT8 quantized).
    Acceleration: NNAPI (Android), CoreML (iOS), CPU (Fallback).
    """

    model_version: str = "1.0-quantized"
    provider_type: str = "onnx"

    def __init__(
        self,
        model_id: str = "Xenova/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        use_acceleration: bool = True,
    ):
        self._model_id = model_id
        self._cache_dir = Path(cache_dir or os.path.expanduser("~/.aura/models/onnx"))
        self._use_acceleration = use_acceleration
        
        self._model_path: Optional[Path] = None
        self._tokenizer_path: Optional[Path] = None
        self._tokenizer: Optional[Any] = None
        self._session: Optional[Any] = None
        self._dims: int = 384 # Default for MiniLM-L6-v2

    @property
    def model_name(self) -> str:
        return self._model_id

    def dimensions(self) -> int:
        if self._session is None:
            return self._dims
        return self._dims

    def _ensure_model(self) -> Path:
        """Download model from HF if not cached."""
        if self._model_path:
            return self._model_path
            
        try:
            self._model_path = Path(hf_hub_download(
                repo_id=self._model_id,
                filename="onnx/model_quantized.onnx",
                cache_dir=str(self._cache_dir)
            ))
            return self._model_path
        except Exception as e:
            logger.error(f"Failed to download ONNX model: {e}")
            raise

    def _ensure_tokenizer(self) -> Path:
        """Download tokenizer from HF if not cached."""
        if self._tokenizer_path:
            return self._tokenizer_path
            
        try:
            self._tokenizer_path = Path(hf_hub_download(
                repo_id=self._model_id,
                filename="tokenizer.json",
                cache_dir=str(self._cache_dir)
            ))
            return self._tokenizer_path
        except Exception as e:
            logger.error(f"Failed to download tokenizer: {e}")
            raise

    def _init_resources(self):
        """Initialize tokenizer and ONNX session."""
        if self._session:
            return

        self._ensure_model()
        self._ensure_tokenizer()

        # Initialize Tokenizer
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(str(self._tokenizer_path))

        # Initialize ONNX Session with acceleration
        providers = ['CPUExecutionProvider']
        
        if self._use_acceleration:
            platform = sys.platform
            is_android = os.path.exists('/system/build.prop') or 'ANDROID_ROOT' in os.environ
            
            if is_android:
                logger.info("Detected Android: Enabling NNAPI acceleration")
                providers.insert(0, 'NNAPIExecutionProvider')
            elif platform == 'darwin':
                logger.info("Detected iOS/macOS: Enabling CoreML acceleration")
                providers.insert(0, 'CoreMLExecutionProvider')

        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(self._model_path), providers=providers)
        except Exception as e:
            logger.warning(f"Failed to initialize accelerated session ({e}). Falling back to CPU.")
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(self._model_path), providers=['CPUExecutionProvider'])

        # Update dimensions
        self._dims = self._session.get_outputs()[0].shape[-1]
        if isinstance(self._dims, str):
             self._dims = 384

    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2 Normalization."""
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norm, a_min=1e-9, a_max=None)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Legacy interface: returns list of raw float vectors."""
        results = self.embed_batch(list(texts))
        return [r.vector for r in results]

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single string and return a structured EmbeddingResult."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a batch of strings and return a list of EmbeddingResults."""
        if not texts:
            return []

        self._init_resources()

        # Tokenize
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        input_names = [i.name for i in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
            inputs["token_type_ids"] = token_type_ids

        # Run inference
        outputs = self._session.run(None, inputs)
        last_hidden_state = outputs[0]

        # Post-processing: Mean Pooling -> L2 Normalization
        embeddings = self._mean_pooling(last_hidden_state, attention_mask)
        embeddings = self._normalize(embeddings)

        return [
            EmbeddingResult(
                vector=vec.tolist(),
                model_name=self._model_id,
                model_version=self.model_version,
                provider_type=self.provider_type,
                dimensions=self._dims
            )
            for vec in embeddings
        ]

    def healthcheck(self) -> bool:
        """Verify the provider is operational by running a simple embedding."""
        try:
            self.embed_text("health")
            return True
        except Exception:
            return False
