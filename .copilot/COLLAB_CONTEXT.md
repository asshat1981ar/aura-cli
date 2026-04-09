# Copilot Collaboration Context

Canonical source of truth: `/home/westonaaron675/aura-cli/COLLAB_CONTEXT.md`

This file exists so Copilot-oriented workflows can discover the shared collaboration context in a predictable location.

## Rule

- Read the root `COLLAB_CONTEXT.md` first.
- Treat the root file as authoritative if this file and the root file differ.
- Write ongoing task state, external-chat handoff notes, and current requests into the root file instead of duplicating them here.

## Resume Prompt

```text
Use /home/westonaaron675/aura-cli/COLLAB_CONTEXT.md as the current source of truth for this task. Read it first, then inspect the codebase as needed, make the necessary changes, run relevant verification, and summarize the outcome.
```
