# Android Local Models For AURA

## Why This Exists

AURA already supports a single `local_model_command`, but that path is too weak
for serious software-development work on-device:

- it treats every task as one generic local prompt
- it does not distinguish planning from code generation
- it makes it easy for small local models to break JSON-heavy phases with prose
- it does not let us pair a code-specialized model with a separate reasoning model

This document recommends a practical Android stack for Termux/proot and maps it
to the new `local_model_profiles` + `local_model_routing` runtime support.


## Current AURA Weaknesses For Software Development

### 1. Planning Is Brittle On Small Local Models

`PlannerAgent` expects a strict JSON array. The previous local-model path simply
appended the prompt to a shell command, so smaller models often answered with
explanation text instead of the required structure.

### 2. Code Generation And Planning Need Different Models

The repo already has distinct roles like:

- `planning`
- `analysis`
- `critique`
- `code_generation`
- `quality`

but the old local path had no way to route those roles to different models.

### 3. The Default Cloud Stack Is Not Phone-Friendly

Current defaults are remote-first and optimized for availability, not offline or
low-latency Termux work. That is fine for the desktop path, but it leaves a
gap for on-device development.


## Recommended Android-Feasible Model Stack

These are the models I recommend prioritizing for Android phones under Termux
or proot, using `llama.cpp` server or another OpenAI-compatible local server.

### Tier 1: Best Practical Coding Choice

#### Qwen2.5-Coder-3B-Instruct-GGUF

Official source:

- <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF>

Why it fits:

- code-specialized instead of general chat-only
- small enough to be realistic on stronger phones when quantized
- a good fit for `code_generation`

Strengths:

- code synthesis
- file-local transformations
- structured patch-style output compared with general tiny instruct models

Weaknesses:

- weaker long-horizon reasoning than larger cloud models
- can still drift out of strict JSON when prompts get complex
- multi-file architectural refactors remain hit-or-miss on phone-class hardware


### Tier 1: Best Small Reasoning / Planning Pair

#### Phi-4-mini-instruct

Official source:

- <https://huggingface.co/microsoft/Phi-4-mini-instruct>

Why it fits:

- strong compact reasoning model for its size class
- better planner / debugger / critic candidate than a pure coder model

Strengths:

- planning
- diagnosis
- critique
- summary and restructuring tasks
- best fit for AURA's planner/debugger/critic path when you want stronger
  software-development reasoning than a phone-first generalist

Weaknesses:

- not code-specialized in the same way as Qwen Coder
- can produce respectable plans but weaker implementation details than coder-first models


### Tier 1: Best Android-Native Generalist

#### Gemma 3n

Official sources:

- <https://ai.google.dev/gemma>
- <https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/>

Why it fits:

- explicitly aimed at on-device usage
- better choice than older desktop-centric models if Android practicality matters most

Strengths:

- on-device friendliness
- low-latency general reasoning
- good `planning` / `analysis` fallback for constrained phones

Weaknesses:

- less code-specialized than Qwen2.5-Coder
- likely better as planner/analyst than as primary code generator


### Tier 2: Strong Reasoning Distill For Bigger Phones

#### DeepSeek-R1-Distill-Qwen-1.5B or 7B

Official source:

- <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>

Why it fits:

- useful when you want stronger chain-of-thought-style reasoning in a smallish package

Strengths:

- debugging
- reasoning-heavy critique
- failure analysis

Weaknesses:

- often verbose
- can be slower per useful token than more compact instruct models
- weaker than a code-specialized model for direct file generation


### Tier 3: Newest Open Coding Frontier, But Not A Real Phone Default

#### Qwen3-Coder-Next

Official source:

- <https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B-Instruct-GGUF>

Why it matters:

- this is one of the newest open coding families and worth watching

Why it is not the Android default:

- despite sparse activation, it is still an 80B-class artifact family and not a
  practical everyday Termux default for most phones


## Recommended AURA Deployment Strategy

### Baseline Android Stack

Use two local profiles:

1. `android_coder`
   Model: `Qwen2.5-Coder-3B-Instruct-GGUF`
   Role targets:
   - `code_generation`
   - `fast`

2. `android_planner`
   Model: `Phi-4-mini-instruct`
   Role targets:
   - `planning`
   - `analysis`
   - `critique`
   - `quality`

If you need the most Android-native fallback, `Gemma 3n` remains a good
alternative planner profile, but `Phi-4-mini-instruct` is the stronger default
for software-development reasoning.

### Why Two Models Instead Of One

This cleanly shores up AURA's current software-development gaps:

- coder model handles implementation and patch-style output better
- planner model handles diagnosis, planning, and critique better
- AURA stops overusing a single small model for every phase


## Runner Recommendation

### Preferred: `llama.cpp` Server

Official source:

- <https://github.com/ggml-org/llama.cpp>

Why:

- best fit for GGUF quantized models
- practical for Termux/proot
- simple OpenAI-compatible API path for AURA

### Secondary: Ollama-Compatible Endpoint

Official source:

- <https://ollama.com/library>

Why:

- simpler model UX if available in the environment
- good fallback where an Ollama-compatible local server exists


## New AURA Config Shape

Example:

```json
{
  "local_model_profiles": {
    "android_coder": {
      "provider": "openai_compatible",
      "base_url": "http://127.0.0.1:8080/v1",
      "model": "qwen2.5-coder-3b-instruct-q4",
      "temperature": 0.1,
      "max_tokens": 2048
    },
    "android_planner": {
      "provider": "openai_compatible",
      "base_url": "http://127.0.0.1:8081/v1",
      "model": "phi-4-mini-instruct-q4",
      "temperature": 0.2,
      "max_tokens": 1536
    }
  },
  "local_model_routing": {
    "planning": "android_planner",
    "analysis": "android_planner",
    "critique": "android_planner",
    "code_generation": "android_coder",
    "quality": "android_planner",
    "fast": "android_coder"
  }
}
```


## Rough Android Practicality Notes

These are implementation-oriented estimates, not guarantees:

- 1.5B to 4B models are the realistic daily-driver zone on phones
- 7B quantized models are plausible on higher-memory devices, but latency rises fast
- sparse MoE coder models are exciting, but storage and RAM still make them poor
  defaults for everyday phone use


## Implementation Status In This Repo

The runtime now supports:

- multiple named local profiles
- phase-aware local routing
- OpenAI-compatible local servers
- Ollama-compatible endpoints
- command-based local profiles for custom runners

Agents now route by role:

- planner -> `planning`
- debugger -> `analysis`
- critic -> `critique` / `analysis`
- coder -> `code_generation`
- tester -> `quality`


## Termux `llama.cpp` Launch Helper

This repo now includes:

- `scripts/run_android_local_models.sh`

It starts two separate `llama.cpp` OpenAI-compatible servers:

- coder model on `127.0.0.1:8080`
- planner model on `127.0.0.1:8081`

Example:

```bash
export AURA_ANDROID_CODER_MODEL="$HOME/models/qwen2.5-coder-3b-instruct-q4.gguf"
export AURA_ANDROID_PLANNER_MODEL="$HOME/models/phi-4-mini-instruct-q4.gguf"
bash scripts/run_android_local_models.sh
```

You can override:

- `LLAMA_SERVER_BIN`
- `AURA_ANDROID_HOST`
- `AURA_ANDROID_CODER_PORT`
- `AURA_ANDROID_PLANNER_PORT`
- `AURA_ANDROID_THREADS`
- `AURA_ANDROID_CODER_CTX`
- `AURA_ANDROID_PLANNER_CTX`

This two-port layout matches `aura.config.json` and
`aura.config.android.example.json`.


## Termux / proot Setup

This repo now also includes:

- `scripts/setup_llama_cpp_termux.sh`

That helper:

- installs the common Termux build dependencies
- clones `llama.cpp` if needed
- builds `llama-server`
- prints the `LLAMA_SERVER_BIN` export you can reuse with
  `scripts/run_android_local_models.sh`

Example:

```bash
bash scripts/setup_llama_cpp_termux.sh
export AURA_ANDROID_CODER_MODEL="$HOME/models/qwen2.5-coder-3b-instruct-q4.gguf"
export AURA_ANDROID_PLANNER_MODEL="$HOME/models/phi-4-mini-instruct-q4.gguf"
bash scripts/run_android_local_models.sh
```


## Recommended Next Steps

1. Stand up a local `llama.cpp` server in Termux or proot.
2. Start with `android_planner` + `android_coder`.
3. Verify planner JSON stability first.
4. Only then consider adding a third local reasoning model.
5. Add local embeddings later if semantic-memory quality becomes the next bottleneck.
