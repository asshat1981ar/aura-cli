# Recursive Improvement Prototype Formalization

## Status

This document formalizes the intent behind the local recursive-improvement
prototype files without adopting their code directly into the product.

Reviewed prototype files:

- `agents/RecursiveImprovementAgent.py`
- `agents/recursive_improvement.py`
- `agents/recursive_improvement_agent.py`
- `core/aura_improvement.py`
- `core/improvement_plan.py`
- `core/policies/fitness_policy.py`
- `core/recursive_improvement_agent.py`

Current conclusion:

- do not commit the prototype implementations as-is
- preserve the useful concepts in a single future design
- keep runtime policy, orchestrator flow, and improvement-loop wiring aligned


## What Is Salvageable

The prototype cluster repeats the same themes across multiple files. The ideas
worth keeping are:

- system analysis before proposing changes
- explicit success metrics and KPIs
- risk and failure-mode assessment
- incremental rollout rather than monolithic mutation
- feedback loops that learn from outcomes
- post-implementation review
- a lightweight fitness score for meta-performance


## What Should Be Discarded

The current prototype files should not be treated as implementation-ready
because they introduce several problems:

- four overlapping `RecursiveImprovementAgent` classes with no clear owner
- placeholder methods with no integration into the actual AURA runtime
- a second `Policy` implementation in `core/policies/fitness_policy.py` that
  conflicts with `core/policy.py`
- hard-coded planning text and KPIs that are not grounded in the repo state
- no tests and no usage references in tracked code


## Canonical Future Shape

If this feature is implemented, it should be consolidated into three tracked
components only:

1. `core/recursive_improvement.py`
   A runtime service that evaluates improvement opportunities from recent cycle
   history, quality signals, and operator-configured constraints.

2. `core/fitness.py`
   A small helper module that computes a meta-performance score from inputs such
   as success rate, token usage, retry count, and complexity deltas.

3. `conductor/tracks/recursive_self_improvement_20260301/`
   The conductor track remains the planning authority for rollout and
   acceptance, rather than embedding product planning into runtime classes.


## Recommended Responsibilities

### Recursive Improvement Service

The future recursive-improvement service should:

- read recent cycle outcomes from canonical runtime history
- compute a bounded improvement candidate set
- emit structured proposals, not directly mutate files
- respect policy and stop conditions already enforced by the orchestrator
- write operator-visible summaries for review and audit

It should not:

- replace `LoopOrchestrator`
- redefine the core `Policy` abstraction
- invent its own execution loop outside the existing runtime


### Fitness Scoring

The `FitnessFunction` idea is useful, but only as a helper. A future fitness
module may score dimensions like:

- success rate
- retry count
- verification pass rate
- token efficiency
- complexity delta
- latency

This score should be consumed by improvement logic, not used to replace the
existing stop-policy evaluator in `core/policy.py`.


## Proposed Data Contract

Any future recursive-improvement proposal should use a structured contract like:

```python
{
    "proposal_id": "ri_20260303_001",
    "source_cycles": ["cycle_abc123", "cycle_def456"],
    "summary": "Reduce retry churn in act/verify handoff.",
    "fitness_snapshot": {
        "score": 0.61,
        "success_rate": 0.72,
        "retry_rate": 0.28,
        "complexity_delta": 0.03,
    },
    "hypotheses": [
        "retry churn is driven by unstable change-set shape",
    ],
    "recommended_actions": [
        "tighten synthesize schema for act phase",
        "add verification fixture for retry regressions",
    ],
    "risk_level": "medium",
    "requires_operator_review": true,
}
```


## Migration Plan From Prototypes

1. Keep the current prototype files out of tracked history.
2. If work resumes, start with tests and the data contract above.
3. Implement only one runtime service and one fitness helper.
4. Wire them through existing orchestrator/improvement-loop hooks.
5. Remove any attempt to create a second `Policy` class.


## Recommendation

Treat the current prototype cluster as ideation material, not source code.

If we continue this feature later, the next implementation step should be:

- add a tracked test-first `core/fitness.py`
- add a tracked test-first `core/recursive_improvement.py`
- connect them to the existing improvement-loop framework behind config

Until then, these prototype files should remain untracked or be moved to a
scratch branch.
