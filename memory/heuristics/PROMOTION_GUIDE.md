# Heuristic Promotion Guide

This guide defines the lifecycle of a heuristic pattern in the AURA platform, from initial discovery to long-term memory.

## 1. Draft State
- **Discovery**: A lesson is identified during a sprint retrospective (e.g., `.aura_forge/retros/S000_retro.md`).
- **Capture**: The lesson is added to the `draft` section of `.aura_forge/indexes/heuristic_index.yaml`.
- **Criteria**: Any pattern that improves safety, reliability, or developer experience.

## 2. Validated State
- **Validation**: The draft pattern must be applied and proven successful across at least **3 different stories** or tasks.
- **Promotion**: Move the pattern from `draft` to `validated` in the `heuristic_index.yaml`.
- **Evidence**: Links to the stories where the pattern was applied should be included in the `origin` or a new `evidence` field.

## 3. Promoted (Long-Term Memory)
- **Persistence**: Once a pattern is validated, it is promoted to long-term memory in `memory/heuristics/*.yaml`.
- **Format**: Create a structured YAML file defining the pattern, rationale, and usage instructions.
- **Impact**: Promoted heuristics are injected into high-priority planning tasks to guide future agent behavior.

## 4. Archival
- **Staleness**: If a heuristic is superseded by a new platform capability or architectural change, it should be moved to an `archive/` sub-directory.
