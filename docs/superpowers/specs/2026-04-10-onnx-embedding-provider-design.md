# Design Spec: ONNX Embedding Provider (Mobile Optimized)

## Overview
This spec defines the implementation of a local embedding provider using ONNX Runtime Mobile. It is designed for high-performance, low-memory footprints on Android and iOS while maintaining compatibility with the AURA `ASCM v2` embedding contract.

## Summary
The `ONNXEmbeddingProvider` will leverage `onnxruntime-mobile` and INT8 quantized models to provide high-quality embeddings locally. It includes automatic model downloading from Hugging Face, platform-specific hardware acceleration (NNAPI/CoreML), and local caching for offline operation.

## Prerequisites & Refactoring
Before implementing the ONNX provider, the existing embedding infrastructure will be refactored for better type safety and extensibility:
1. **Define ABC**: Create `EmbeddingProvider` Abstract Base Class in `memory/embedding_provider.py` to enforce the contract (`embed_text`, `embed_batch`, `dimensions`, `healthcheck`).
2. **Update Type**: Update `EmbeddingResult.provider_type` annotation/comment to support `onnx`.
3. **Refactor existing providers**: Update `LocalEmbeddingProvider` and `OpenAIEmbeddingProvider` to inherit from the new ABC.

## Architecture

### 1. Core Component
- **File**: `memory/onnx_embedding_provider.py`
- **Class**: `ONNXEmbeddingProvider`
- **Contract**: Inherits from `EmbeddingProvider` ABC.

### 2. Dependencies
- `onnxruntime-mobile` (or `onnxruntime` if on desktop): Core inference engine.
- `huggingface_hub`: For model retrieval.
- `numpy`: Tensor operations and mean pooling.
- `tokenizers`: For text preprocessing.
- **Dependency Strategy**: Add as an optional extra in `pyproject.toml` under `onnx`. Use a helper to check for available runtime (mobile vs standard).

### 3. Model Management
- **Default Model**: `Xenova/all-MiniLM-L6-v2` (quantized to INT8).
- **Size**: ~22MB (compared to ~90MB for FP32).
- **Storage**: Configurable path via `aura.config.json` (defaults to `~/.aura/models/onnx/`).
- **Logic**: 
  - Check local cache.
  - If missing, download `model_quantized.onnx`, `tokenizer.json`, and `config.json` using `hf_hub_download`.
  - Handle download timeouts and display progress via `rich`.

### 4. Hardware Acceleration
The provider will dynamically select execution providers:
- **Android**: `NNAPIExecutionProvider` (if available).
- **iOS**: `CoreMLExecutionProvider` (if available).
- **Fallback**: `CPUExecutionProvider` (optimized for ARM/x86 via MLAS).

## Data Flow
1. **Input**: String or List of Strings.
2. **Preprocessing**: Tokenize text into `input_ids`, `attention_mask`, and `token_type_ids`.
3. **Inference**: Run ONNX session.
4. **Postprocessing**:
   - Apply mean pooling to the `last_hidden_state` using the `attention_mask`.
   - Normalize the resulting vector (L2 normalization).
5. **Output**: `EmbeddingResult` or raw List[float].

## Success Criteria
- [ ] `EmbeddingProvider` ABC is defined and enforced.
- [ ] Successfully loads `all-MiniLM-L6-v2` INT8 model.
- [ ] Produces 384-dimension vectors.
- [ ] Memory usage remains under 100MB during inference.
- [ ] Supports offline mode after initial download.
- [ ] Integration tests pass against `LoopOrchestrator`.

## Future Considerations
- Support for larger models (e.g., `BGE-small-en-v1.5`) via configuration.
- Custom delegate tuning for specific NPU hardware.
