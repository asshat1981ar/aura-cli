# Advanced Semantic Context Manager (ASCM) PRD

## HR Eng

| ASCM PRD |  | **Summary**: Replaces the basic, recency-based context gathering with a sophisticated, semantic, and graph-aware context management system. This ensures the LLM always has the most relevant code and history in its prompt window, significantly improving autonomous development accuracy. |
| :---- | :---- | :---- |
| **Author**: Gemini CLI **Contributors**: [Names] **Intended audience**: Engineering, AI/LLM Teams | **Status**: Implemented **Created**: February 27, 2026 | **Self Link**: [Link] **Context**: Internal codebase inspection |

## Introduction

AURA's current context management is primitive. The `IngestAgent` simply lists files and retrieves the last 10 memory entries. This "recency-only" approach leads to "context blindness" in complex tasks, where the agent loses sight of relevant architectural patterns, distant but related files, or important past failures. The Advanced Semantic Context Manager (ASCM) will leverage AURA's `VectorStore` and `ContextGraph` to dynamically curate a "Context Bundle" for every agent phase, prioritizing information based on semantic relevance and relational importance.

## Problem Statement

**Current Process:** 
1. `IngestAgent` performs an `os.walk` to list files (capped at 150).
2. It retrieves the 10 most recent memory entries.
3. This context is passed statically to subsequent phases.

**Primary Users:** AURA agents (Planner, Coder, Critique) and developers managing AURA.

**Pain Points:**
- **Context Blindness:** Agents often lack information about files that are semantically related but not "recent" in the memory log.
- **Token Inefficiency:** The prompt window is often filled with irrelevant file lists while missing crucial code snippets.
- **Hallucinations:** Without seeing the actual content of related files, the LLM often guesses at APIs or internal structures.
- **Broken Implementation:** The current `agents/context_manager.py` is a broken directory containing filenames that are snippets of code, indicating a failed prior implementation attempt.

**Importance:** As tasks grow in complexity, AURA's success rate depends entirely on the quality of its context. ASCM is the foundation for high-level autonomy.

## Objective & Scope

**Objective:** To implement a robust, semantic context management system that replaces the current `IngestAgent` logic and the broken `context_manager.py` directory.

**Ideal Outcome:** Every AURA phase receives a perfectly curated context bundle that includes:
- The goal and critical constraints.
- Semantically relevant code snippets (via VectorStore).
- Relationally related file summaries (via ContextGraph).
- Intelligent token budget allocation.

### In-scope or Goals
- Delete the broken `agents/context_manager.py/` directory.
- Create a new `core/context_manager.py` module.
- Implement **Semantic Retrieval** integration with `VectorStore`.
- Implement **Relational Retrieval** integration with `ContextGraph`.
- Implement an **Information Ranker** that prioritizes context based on the current `goal_type`.
- Integrate ASCM into `LoopOrchestrator` and `IngestAgent`.

### Not-in-scope or Non-Goals
- Real-time streaming of context (this will remain request-response per phase).
- Training new embedding models (we will use existing OpenAI/local embeddings).
- Managing "Long-term Memory" outside of the existing `Brain` structure.

## Product Requirements

### Critical User Journeys (CUJs)
1. **The "Distantly Related File" Fix:**
   - AURA is fixing a bug in `core/orchestrator.py`.
   - ASCM identifies that `core/hybrid_loop.py` is semantically similar and frequently linked in the `ContextGraph`.
   - ASCM includes relevant snippets from `hybrid_loop.py` in the context, even if it hasn't been mentioned in the last 10 memories.
   - The `CoderAgent` correctly identifies the interaction between the two and applies a valid fix.

2. **The "Past Failure" Avoidance:**
   - AURA is tasked with a "refactor" goal.
   - ASCM queries `ContextGraph` for past "failed_on" relations for the involved files.
   - ASCM includes a "Warning: Past failures in this module were caused by X" hint in the context.
   - The `PlannerAgent` creates a plan that specifically avoids "X".

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | **Module Reconstruction** | As a developer, I want the broken `agents/context_manager.py/` replaced with a working module so the system is maintainable. |
| P0 | **Semantic Snippet Injection** | As an agent, I want to see code snippets from files related to my goal, not just their filenames. |
| P0 | **Token Budget Balancing** | As AURA, I want to intelligently drop low-value information (like long file lists) to make room for high-value code snippets. |
| P1 | **Graph-Based Context** | As an agent, I want to see files that are "downstream" of my changes according to the system's dependency graph. |
| P1 | **Goal-Aware Prioritization** | As AURA, I want "Bug Fix" goals to prioritize error logs and "Feature" goals to prioritize API contracts. |
| P2 | **Context Compression** | As AURA, I want the system to summarize large related files instead of just truncating them. |

## Assumptions

- `VectorStore` is populated with embeddings of the current codebase.
- `ContextGraph` has enough edges to provide meaningful relational data.
- The LLM's context window is at least 8k-16k tokens.

## Risks & Mitigations

- **Risk**: Semantic retrieval is slow. -> **Mitigation**: Use local L0/L1 caches for embeddings and pre-fetch context during the `Ingest` phase.
- **Risk**: Irrelevant context noise. -> **Mitigation**: Implement a strict "Relevance Score" threshold for snippet injection.
- **Risk**: Recursive Ingestion (ASCM calling itself). -> **Mitigation**: ASCM must be a pure utility called by agents, not an agent that executes cycles.

## Tradeoff

- **Option Chosen: ASCM Utility.** Instead of a full "Context Agent", we implement a high-performance utility library used by all agents.
- **Pros:** Lower latency, predictable behavior, easy to integrate into existing phases.
- **Cons:** Less "autonomy" in how it decides what is relevant compared to a dedicated LLM agent.

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State (Benchmark) | Future State (Target) | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| **Hallucination Rate** | High (est. 20%) | Low (<5%) | Fewer broken PRs. |
| **First-Pass Pass Rate** | ~40% | >65% | Faster task completion. |
| **Token Utilization** | Suboptimal (Static) | Optimal (Dynamic) | Better "intelligence" per token spent. |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| Gemini CLI | Engineering | Lead Implementer | |
| AURA Core | Engineering | Architectural Oversight | |
