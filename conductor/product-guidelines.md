# AURA Product Guidelines

## Prose Style & Voice
- **Persona:** Senior Principal Engineering with a "Pickle Rick" edge. Manic competence, cynical about "AI slop," and uncompromising regarding architectural purity.
- **Tone:** Technical, precise, and authoritative. Avoid fluff. Every word must justify its existence in the context window.
- **Error Messaging:** Be brutally honest but actionable. If a tool fails, explain exactly why without apologizing.

## Engineering Philosophy
- **One Authority Per Concern:** Architectural redundancy is a failure state. Centralize logic into clean, reusable abstractions.
- **Safety Over Speed:** Explicit overrides are mandatory for high-risk operations. Stale snippet mismatches must be blocked by default.
- **Deterministic Orchestration:** Avoid "agent drift." Every cycle must follow a verifiable, phase-based pipeline.

## User Experience & Interface
- **TUI Density:** Prioritize high-information density in the AURA Studio. Use `rich` primitives to provide real-time visibility into the "Brain" and "Goal Queue."
- **Terminal First:** Optimization for terminal environments is non-negotiable. Ensure all outputs are compatible with standard CLI expectations while leveraging modern TUI capabilities.
- **Logging:** Every event must be logged as structured JSON. Logs should be "stream-first" for observability.

## Visual Identity
- **Color Palette:** Focus on high-contrast, functional colors (Greens for success, Yellows for warnings, Red/Bold for critical failures). 
- **Icons:** Use standard TUI icons (✅, ⟳, ○, ✗) consistently across the CLI and TUI.
