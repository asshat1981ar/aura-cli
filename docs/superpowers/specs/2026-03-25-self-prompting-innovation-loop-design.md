# Self-Prompting Innovation Loop Design

## Summary

Add a bounded, queue-first self-prompting background loop to AURA that continuously analyzes the codebase, runs divergent subagent-style research passes, synthesizes implementation-ready improvement bundles, and enqueues only the highest-value bounded work for the existing orchestrator to execute later.

This design uses the current runtime path centered on `core/orchestrator.py` and `core/evolution_loop.py`. It does not introduce a second code-execution authority. The new loop acts as a research, proposal, and queue-generation plane only.

## Goals

- Add continuous self-prompting to AURA without allowing uncontrolled same-run mutation.
- Use bounded subagent-style divergent loops to analyze architecture, capabilities, MCP/tools, test debt, and repeated failures.
- Produce implementation-ready queued bundles instead of free-form ideas.
- Keep execution ownership with the normal orchestrator and goal queue.
- Make background innovation activity inspectable through status, memory, and operator surfaces.

## Non-Goals

- No direct background code mutation.
- No unbounded recursive agent spawning.
- No second top-level orchestrator separate from the current runtime.
- No queue flooding with large speculative roadmap items.
- No replacing the current `EvolutionLoop`; this extends and refactors it.

## Architecture

Add a new bounded `SelfPromptingInnovationLoop` attached through the orchestrator improvement-loop system.

Runtime shape:

- `observe`
- `trigger`
- `self-prompt DAG`
- `divergent analysis branches`
- `synthesize`
- `rank`
- `dedupe`
- `queue bundle`
- `record artifacts`

Recommended ownership:

- `core/orchestrator.py`
  - loop attachment
  - trigger invocation
  - queue handoff
  - runtime summaries and operator visibility
- `core/evolution_loop.py`
  - innovation-state assembly
  - branch execution
  - proposal synthesis
  - bundle ranking
  - dedupe logic

The normal orchestrator remains the only component allowed to execute queued work.

## DAG Model

The loop adapts the user-provided architect/coder/reviewer/debugger pattern into a queue-first DAG:

- `architect`
  - frames the improvement target
  - emits a structured plan for the current innovation run
- `divergent_researchers`
  - parallel bounded branches for:
    - architecture exploration
    - capability gap analysis
    - MCP/tooling opportunity analysis
    - test and verification debt analysis
    - repeated failure or hotspot analysis
- `reviewer`
  - filters weak, unsafe, or oversized proposals
- `debugger`
  - activates when the trigger source is repeated failure or verification regression
- `synthesizer`
  - merges branch outputs into normalized candidate bundles
- `ranker`
  - scores bundles by value, boundedness, risk, and verification cost
- `queue_writer`
  - writes only the top deduped bundles to the goal queue

## Queued Bundle Contract

Each queued artifact is a bundle, not a single loose goal.

Required fields:

- `bundle_id`
- `title`
- `category`
- `trigger_reason`
- `target_surfaces`
- `evidence`
- `proposal`
- `implementation_task`
- `test_plan`
- `verification_plan`
- `risk_notes`
- `estimated_scope`
- `recommended_priority`
- `status`

Recommended semantics:

- `category`
  - `capability`
  - `skill`
  - `mcp`
  - `verification`
  - `observability`
  - `developer_surface`
  - `architecture`
- `status`
  - `draft`
  - `queued`
  - `suppressed`
  - `roadmap_only`
- `estimated_scope`
  - `small`
  - `medium`
  - `large`

Only `small` or tightly-bounded `medium` bundles should be queued automatically. `large` items should degrade to roadmap artifacts.

## Trigger Model

The loop runs in the background, but only under explicit bounded conditions.

Supported trigger classes:

- cadence trigger
  - every `N` cycles
- failure trigger
  - repeated verification failures on similar surfaces
- capability trigger
  - recurring missing skills, tools, or MCP gaps
- hotspot trigger
  - repeated changes, churn, or debt signals in the same area
- operator trigger
  - explicit `evolve` or config-driven request

Recommended defaults:

- run every `10` or `20` cycles
- max `4-5` divergent branches per run
- max `3` refinement iterations per run
- queue at most `1-2` bundles per run

## Control Limits

To prevent queue spam and prompt drift:

- dedupe against recent queued and recently completed bundles
- enforce cooldown per target surface
- reject schema-invalid branch output
- discard failed branches instead of retrying indefinitely
- stop early when no branch meaningfully improves the candidate set

Suggested controls:

- `innovation_cadence_cycles`
- `innovation_max_branches`
- `innovation_max_iterations`
- `innovation_max_queued_bundles`
- `innovation_surface_cooldown_cycles`

## Divergent Branch Roles

The first implementation should support these bounded branch roles:

- `architecture_explorer`
  - structural analyzer
  - tech debt quantifier
  - hotspot extraction
- `capability_researcher`
  - capability analysis
  - skill gap reporting
  - MCP inventory summary
- `verification_reviewer`
  - repeated-failure patterns
  - test debt and cheapest proof path
- `innovation_synthesizer`
  - combines branch evidence into candidate bundles
- optional `debugger`
  - converts repeated trace/failure patterns into fix-oriented bundles

These are subagent-style roles, but the initial implementation can execute them as bounded role functions inside `EvolutionLoop` before later promotion to explicit external agent packets if needed.

## Ranking Model

Each candidate bundle should be scored using:

- expected value
- boundedness
- implementation surface size
- residual risk
- verification cost
- novelty relative to recent queue history

Priority order:

1. capability expansions that remove repeated operator friction
2. MCP or tool integrations with clear bounded payoffs
3. verification and observability improvements that reduce future rework
4. architecture cleanups with a small change surface
5. speculative or broad ideas last

## Observability

Every self-prompting run should persist artifacts into runtime memory and operator status.

Persist:

- trigger source
- branch summaries
- rejected proposals
- selected bundles
- dedupe decisions
- queue write results
- suppression or cooldown reasons
- residual risks

Expose in operator-facing status:

- last self-prompt run time
- trigger reason
- number of branches executed
- number of candidates proposed, rejected, and queued
- titles of top queued bundles
- cooldown or suppression reasons

## Safety

Safety constraints are strict:

- background innovation loop never writes code directly
- queued outputs must pass schema validation
- invalid branch outputs are dropped
- no recursive trigger from the loop’s own queued outputs
- no automatic queueing of oversized or ambiguous work
- direct execution remains an opt-in future mode, not the default

## Integration Plan

Recommended implementation sequence:

1. add a new bounded innovation-loop class or expand `EvolutionLoop`
2. define bundle schema and ranking helper
3. attach the loop through orchestrator improvement hooks
4. persist run artifacts into operator summary and memory
5. add CLI/status visibility for last innovation run and queued bundles
6. later add explicit opt-in foreground execution modes if needed

## Testing

Add focused tests for:

- trigger gating and cadence behavior
- divergent branch budgeting
- bundle schema generation
- ranking and dedupe behavior
- queue write suppression for large or duplicate bundles
- operator summary artifact exposure
- cooldown behavior
- failure-triggered debugger branch activation

## Future Extensions

Possible later additions:

- explicit foreground `evolve --self-prompt` mode
- richer external subagent dispatch packets
- multi-bundle tournament ranking
- adaptive cadence based on queue health
- direct execution mode behind an explicit operator flag
