# ADR-011: 10-Phase Agent Pipeline

**Date:** 2026-04-10  
**Status:** Accepted  
**Deciders:** AURA Core Team  

## Context

AURA needed an architecture for autonomous software development that could:

1. Process natural language goals into working code
2. Maintain quality through verification and critique
3. Learn from execution results
4. Support iterative refinement
5. Provide observability into the development process
6. Allow for safe experimentation and rollback

We evaluated several approaches:
- **Single-pass generation** — Fast but low quality
- **Two-phase (plan + execute)** — Better but no verification
- **CI-like pipeline** — Sequential stages but rigid
- **Agent-based pipeline** — Flexible, multi-stage with feedback loops

## Decision

We implemented a **10-phase agent pipeline** with feedback loops.

## Pipeline Architecture

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  INGEST │──▶│  PLAN   │──▶│ CRITIQUE│──▶│SYNTHESIZE│──▶│   ACT   │
│         │   │         │   │         │   │          │   │         │
│ Gather  │   │ Generate│   │ Review  │   │ Bundle   │   │ Generate│
│ context │   │ steps   │   │ plan    │   │ tasks    │   │ code    │
└─────────┘   └─────────┘   └────┬────┘   └─────────┘   └────┬────┘
                                 │                          │
                                 └──────────┬───────────────┘
                                            ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ ARCHIVE │◀──│ EVOLVE  │◀──│  ADAPT  │◀──│ REFLECT │◀──│ VERIFY  │
│         │   │         │   │         │   │         │   │         │
│ Store   │   │ Improve │   │ Adjust  │   │ Analyze │   │  Test   │
│ results │   │ skills  │   │ strategy│   │ results │   │  code   │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
     ▲                                                        │
     └──────────────────── FEEDBACK LOOP ─────────────────────┘
```

## Phase Details

| Phase | Agent | Input | Output | Purpose |
|-------|-------|-------|--------|---------|
| **1. Ingest** | IngestAgent | Natural language goal | Context, memory hints | Gather context from memory, files, and previous runs |
| **2. Plan** | PlannerAdapter | Goal + context | Steps, structured output | Generate execution plan with milestones |
| **3. Critique** | CriticAdapter | Plan | Feedback, score | Review plan for feasibility and completeness |
| **4. Synthesize** | SynthesizerAgent | Plan + feedback | Task bundle | Combine plan and critique into executable tasks |
| **5. Act** | ActAdapter | Task bundle | Change set | Generate code changes |
| **6. Sandbox** | SandboxAdapter | Change set | Sandbox result | Execute changes in isolated environment |
| **7. Verify** | VerifierAgent | Changes + results | Verification result | Run tests and validation |
| **8. Reflect** | ReflectorAgent | Goal + results | Reflection, skill updates | Analyze what worked and what didn't |
| **9. Adapt** | (Internal) | Reflection | Strategy adjustments | Adjust approach based on reflection |
| **10. Evolve** | (Internal) | Experience | Skill improvements | Update skills and patterns |
| **11. Archive** | (Internal) | All artifacts | Stored results | Persist to memory tiers |

## Key Design Decisions

### 1. Feedback Loops

The pipeline includes multiple feedback paths:
- **Critique loop**: Plan → Critique → Synthesize
- **Verification loop**: Act → Sandbox → Verify → Reflect
- **Evolution loop**: Full cycle feeds back to improve skills

### 2. Agent Specialization

Each phase has a specialized agent with:
- Specific capabilities (e.g., `planning`, `code_generation`)
- Optimized model selection via `ModelRoutingConfig`
- Isolated concerns for testing and maintenance

### 3. Async Execution

```python
# Core orchestration supports async for I/O-bound operations
async def run_phase(self, phase: str, input_data: dict) -> dict:
    agent = self.registry.resolve_by_capability(phase)
    return await self._dispatch_task(agent, input_data)
```

### 4. Observability

Every phase emits structured telemetry:
```python
log_json(
    level="INFO",
    event="phase_start",
    goal=goal_id,
    phase=phase_name,
    details={"input_tokens": tokens}
)
```

## Consequences

### Positive

- High-quality output through multi-stage review
- Observable and debuggable pipeline
- Easy to add new phases or agents
- Feedback loops enable continuous improvement
- Model routing optimizes cost/quality tradeoffs
- Sandboxed execution prevents system damage

### Negative

- Higher latency than single-pass approaches
- More complex state management
- Requires careful error handling between phases
- Increased token usage for multi-stage processing

### Trade-offs

| Aspect | Single-Pass | 10-Phase Pipeline |
|--------|-------------|-------------------|
| Speed | Fast | Moderate |
| Quality | Variable | Consistently high |
| Cost | Lower | Higher |
| Debuggability | Hard | Easy |
| Safety | Risky | Sandboxed |

## Configuration

```json
{
  "pipeline": {
    "max_cycles": 5,
    "enable_critique": true,
    "enable_sandbox": true,
    "sandbox_timeout": 30,
    "verification_required": true
  }
}
```

## Implementation

The orchestrator is implemented in `core/orchestrator.py`:

```python
class LoopOrchestrator:
    """Main orchestrator for the 10-phase pipeline."""
    
    PHASES = [
        "ingest", "plan", "critique", "synthesize", 
        "act", "sandbox", "verify", "reflect", 
        "adapt", "evolve", "archive"
    ]
    
    async def run_goal(self, goal: str, context: dict = None) -> RunResult:
        state = PipelineState(goal=goal, context=context or {})
        
        for phase in self.PHASES:
            if self._should_skip(phase, state):
                continue
                
            result = await self._run_phase(phase, state)
            state = self._update_state(state, phase, result)
            
            if result.requires_feedback:
                state = await self._handle_feedback(phase, state)
        
        return RunResult.from_state(state)
```

## References

- [Orchestrator Implementation](https://github.com/asshat1981ar/aura-cli/tree/main/core/orchestrator.py)
- [Agent Registry](https://github.com/asshat1981ar/aura-cli/tree/main/core/mcp_agent_registry.py)
- [AGENTS.md](https://github.com/asshat1981ar/aura-cli/blob/main/AGENTS.md)
