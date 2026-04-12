# Local LLM Integration Design

**Date:** 2026-04-10  
**Status:** Approved  
**Target:** Linux CPU-only, 8 GB RAM, 8 cores

---

## Problem Statement

AURA's `local_model_profiles` and `local_model_routing` infrastructure is fully implemented in code but contains only placeholder Android profiles. No Linux/Ollama profiles are configured, so `respond_for_role()` always falls through to cloud (OpenRouter) or errors when offline. This design defines the model stack, profile configuration, and routing for a Linux CPU-only developer workstation.

---

## Architecture

### Backend: Ollama

Ollama is selected as the inference backend because:
- AURA has a dedicated `ollama` provider path (`_call_local_ollama` for chat, `_call_local_ollama_embeddings` for embeddings)
- Native GGUF support via embedded llama.cpp
- Automatic model management (`ollama pull`, model swap, context tracking)
- CPU inference works well on Linux x86-64

**Provider used in profiles:** `ollama` (dispatches to `_call_local_ollama` → `POST /api/generate`)

### Model Stack (8 GB RAM, balanced quality/speed)

| Role | Model | Ollama Tag | Size (Q4_K_M) | Speed (8-core CPU) |
|------|-------|-----------|----------------|---------------------|
| `code_generation`, `quality` | Qwen2.5-Coder-7B-Instruct | `qwen2.5-coder:7b` | ~4.7 GB | ~4–8 tok/s |
| `planning`, `analysis`, `critique`, `fast` | Phi-4-mini-Instruct | `phi4-mini` | ~2.8 GB | ~12–20 tok/s |
| `embedding` | nomic-embed-text v1.5 | `nomic-embed-text` | ~274 MB | <50ms/query |

**Rationale:**
- **Qwen2.5-Coder-7B**: 83% HumanEval — best-in-class 7B coder. Loaded only when generating or quality-checking code. Falls back to phi4-mini on failure.
- **Phi-4-mini (3.8B)**: 74.4% HumanEval, 128K context, structured JSON reliable. Handles planning/critique/analysis where reasoning depth matters more than raw code synthesis.
- **nomic-embed-text v1.5**: 137M params, 274 MB, 8192-token context, Matryoshka 768d embeddings. Near-free resource cost; SOTA local embedding for semantic memory.

Ollama auto-swaps the loaded model when a different profile is called. Only one model occupies RAM at a time.

---

## Configuration

### Profiles (`local_model_profiles`)

```json
"linux_coder_7b": {
  "provider": "ollama",
  "model": "qwen2.5-coder:7b",
  "base_url": "http://127.0.0.1:11434",
  "temperature": 0.2,
  "max_tokens": 4096,
  "request_timeout_seconds": 120,
  "cooldown_seconds": 60,
  "fallback_profiles": ["linux_fast_3b"]
},
"linux_fast_3b": {
  "provider": "ollama",
  "model": "phi4-mini",
  "base_url": "http://127.0.0.1:11434",
  "temperature": 0.3,
  "max_tokens": 2048,
  "request_timeout_seconds": 60,
  "cooldown_seconds": 30
},
"linux_embed": {
  "provider": "ollama",
  "model": "nomic-embed-text",
  "base_url": "http://127.0.0.1:11434",
  "embedding_model": "nomic-embed-text",
  "embedding_dims": 768
}
```

### Routing (`local_model_routing`)

```json
"code_generation": "linux_coder_7b",
"planning":        "linux_fast_3b",
"analysis":        "linux_fast_3b",
"critique":        "linux_fast_3b",
"quality":         "linux_coder_7b",
"fast":            "linux_fast_3b",
"embedding":       "linux_embed"
```

---

## Setup Commands

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (downloads once, ~8 GB total)
ollama pull qwen2.5-coder:7b       # ~4.7 GB
ollama pull phi4-mini              # ~2.8 GB
ollama pull nomic-embed-text       # ~274 MB

# Verify Ollama is running
curl http://127.0.0.1:11434/api/tags

# Test AURA local routing
aura run "write a python hello world"
```

---

## Fallback Behavior

1. `linux_coder_7b` fails → `linux_fast_3b` (configured via `fallback_profiles`)
2. `linux_fast_3b` fails + cooldown active → `respond_for_role` checks `model_routing` for the route key and calls OpenRouter (requires `primary_provider` defaults to `"openrouter"` — fixed in `core/model_adapter.py` line 264)
3. OpenRouter fails → `respond()` default chain (OpenAI → Anthropic → local command)

---

## Resource Budget

| State | RAM Used | Notes |
|-------|----------|-------|
| Ollama idle (no model loaded) | ~0.1 GB | Background daemon |
| Phi-4-mini loaded (3B) | ~3.2 GB | Planning/analysis/fast calls |
| qwen2.5-coder:7b loaded | ~5.5 GB | Code gen / quality calls |
| OS + AURA process | ~2.0 GB | Always present |
| **Peak (7B loaded)** | **~7.5 GB** | Within 8 GB limit with tight headroom |

⚠️ **Note:** On code generation calls, the system will run near 8 GB capacity. Close memory-heavy browser tabs or other applications before intensive coding sessions. The `cooldown_seconds: 60` and `fallback_profiles` ensure graceful recovery if OOM occurs.

---

## Embedding Integration

Runtime dispatch for embeddings:
1. `ModelAdapter.embed(texts)` → `_embed_with_local_profile()`
2. `_embed_with_local_profile()` reads `semantic_memory.embedding_model` = `local_profile:linux_embed`
3. Loads `linux_embed` profile (provider=`ollama`)
4. Calls `_call_local_ollama_embeddings()` → tries `POST /api/embed` first, falls back to `POST /api/embeddings`
5. Returns numpy float32 vectors (768d for nomic-embed-text v1.5)

The `semantic_memory.embedding_model` config key takes priority over `local_model_routing.embedding`. Both are now set to `linux_embed`.

---

## Testing

Existing test coverage in `tests/test_model_adapter_local_profiles.py`:
- `test_respond_for_role_uses_ollama_profile` — covers the ollama provider path
- `test_local_profile_failure_uses_configured_fallback_profile` — covers `fallback_profiles`
- `test_local_profile_cooldown_skips_failed_profile_on_next_call` — covers cooldown

**Gap:** no test covers the Ollama embedding path (`linux_embed` / `_call_local_ollama_embeddings` / `/api/embed`). A test should mock `POST /api/embed` and verify `embed(["hello"])` returns the correct numpy vectors when `semantic_memory.embedding_model = "local_profile:linux_embed"` with `provider = "ollama"`.

---

## Rejected Alternatives

| Option | Reason Rejected |
|--------|----------------|
| All-7B (Option A) | 7B always loaded = 5.5 GB + 2 GB OS = near OOM for fast/planning calls that don't need it |
| All-3B (Option C) | Wastes user's stated preference for quality on code gen |
| vLLM | NVIDIA GPU only; no CPU support |
| llama.cpp server (manual) | Worse DX than Ollama; no auto model management; adds maintenance burden |
