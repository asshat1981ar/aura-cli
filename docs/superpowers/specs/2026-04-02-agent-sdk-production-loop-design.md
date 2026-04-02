# Agent SDK Production-Grade Autonomous Loop — Design Spec

**Status:** Approved  
**Date:** 2026-04-02  
**Builds on:** `core/agent_sdk/` skeleton (8 modules, 45 tests, 1585 LOC)

## Overview

Evolve the Agent SDK meta-controller from a working skeleton into a production-grade autonomous development loop with adaptive model routing, predefined workflow templates, session persistence with cost tracking, and a feedback loop that makes the system learn from outcomes.

## 1. Adaptive Model Router

### Purpose

Select the cheapest Claude model tier that can reliably handle each goal type, learning from historical outcomes.

### Model Tiers

| Tier | Model | Cost/1M in | Cost/1M out | Use when |
|------|-------|-----------|-------------|----------|
| fast | claude-haiku-4-5 | $1.00 | $5.00 | Simple, well-understood goals |
| standard | claude-sonnet-4-6 | $3.00 | $15.00 | Default; most goals |
| powerful | claude-opus-4-6 | $5.00 | $25.00 | Complex planning, persistent failures |

### Selection Algorithm

```
1. Classify goal → goal_type (bug_fix, feature, refactor, security, docs, default)
2. Load stats from model_stats ledger for this goal_type
3. For each tier (cheapest first):
     if tier.success_rate >= 0.7 AND tier.consecutive_failures < 2:
       select this tier
4. If no tier qualifies, use "powerful"
```

### Escalation & De-escalation

- **Escalate:** 2 consecutive failures at current tier → bump up one tier for this session
- **De-escalate:** 5 consecutive successes at a tier → try one tier lower next time (for this goal_type)
- **Session-scoped escalation:** Once escalated mid-session, stay escalated for remainder of session
- **Cross-session learning:** Stats persist across sessions via ledger file

### Persistence

File: `memory/agent_sdk_model_stats.json`

```json
{
  "bug_fix": {
    "fast": {"attempts": 12, "successes": 7, "consecutive_failures": 0, "consecutive_successes": 3, "ema_score": 0.65},
    "standard": {"attempts": 30, "successes": 27, "consecutive_failures": 0, "consecutive_successes": 8, "ema_score": 0.91},
    "powerful": {"attempts": 5, "successes": 5, "consecutive_failures": 0, "consecutive_successes": 5, "ema_score": 1.0}
  }
}
```

EMA update: `new_score = alpha * outcome + (1 - alpha) * old_score` where `alpha = 0.2`, `outcome = 1.0 (success) or 0.0 (failure)`.

### Module

File: `core/agent_sdk/model_router.py`

```python
class AdaptiveModelRouter:
    def __init__(self, stats_path: Path)
    def select_model(self, goal_type: str) -> str  # returns model ID
    def record_outcome(self, goal_type: str, model: str, success: bool) -> None
    def escalate(self, goal_type: str, current_model: str) -> str  # returns next tier
    def get_stats(self) -> Dict  # for CLI display
```

## 2. Workflow Templates

### Purpose

Predefined phase sequences for common goal types. Each workflow defines which tools to call, in what order, how to verify, and how to retry on failure.

### Workflow Dataclass

```python
@dataclass
class WorkflowPhase:
    tool_name: str           # e.g. "analyze_goal", "create_plan"
    required: bool = True    # skip if False and previous phase failed
    retry_on_fail: int = 0   # max retries for this phase
    escalate_on_fail: bool = False  # trigger model escalation on failure

@dataclass  
class WorkflowTemplate:
    name: str                        # e.g. "bug_fix"
    goal_types: List[str]            # which goal_types this handles
    phases: List[WorkflowPhase]      # ordered phase sequence
    max_retries_total: int = 3       # safety cap across all phases
    verification_mode: str = "post"  # "post" | "pre_and_post" | "none"
```

### Bug-Fix Workflow

```
analyze_goal → search_memory → dispatch_skills(bug_fix) → create_plan
→ generate_code (retry=3, escalate_on_fail) → run_sandbox (retry=2)
→ apply_changes → verify_changes → reflect_on_outcome → store_memory
```

- verification_mode: `"post"`
- On verify fail: re-analyze error, retry generate_code with error context
- On 2 sandbox failures: escalate model tier

### Feature Workflow

```
analyze_goal → search_memory → dispatch_skills(feature) → create_plan
→ critique_plan → generate_code (retry=2, escalate_on_fail)
→ run_sandbox (retry=2) → apply_changes → verify_changes
→ reflect_on_outcome → store_memory
```

- verification_mode: `"post"`
- Adds critique_plan for architectural review
- On verify fail: re-plan from critique feedback (max 2 re-plans)

### Refactor Workflow

```
analyze_goal → search_memory → dispatch_skills(refactor)
→ verify_changes(baseline) → create_plan → generate_code (retry=2)
→ apply_changes → verify_changes(regression) → reflect_on_outcome → store_memory
```

- verification_mode: `"pre_and_post"`
- Baseline verification BEFORE changes to detect regressions
- On regression: revert and retry with smaller scope

### Fallback

Goals that don't match any workflow use the feature workflow (most general).

### Module

File: `core/agent_sdk/workflow_templates.py`

```python
@dataclass
class PhaseResult:
    tool_name: str
    success: bool
    output: Dict[str, Any]      # raw tool handler return value
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    model_used: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0

@dataclass
class WorkflowResult:
    success: bool
    phases_completed: int
    phase_results: List[PhaseResult]
    total_cost_usd: float = 0.0
    model_escalations: int = 0
    error_summary: Optional[str] = None

class FailureAction(enum.Enum):
    RETRY_PHASE = "retry_phase"          # retry same phase
    ESCALATE_AND_RETRY = "escalate"      # bump model tier, retry
    REPLAN = "replan"                    # go back to create_plan
    ABORT = "abort"                      # give up

class WorkflowExecutor:
    def __init__(self, templates: Dict[str, WorkflowTemplate], 
                 tool_handlers: Dict[str, Callable[[Dict], Dict]])
        """
        tool_handlers: maps tool name → bound handler function from tool_registry.
        Built by: {t.name: t.handler for t in create_aura_tools(...)}
        """
    def select_workflow(self, goal_type: str) -> WorkflowTemplate
    def execute(self, workflow: WorkflowTemplate, goal: str, context: Dict) -> WorkflowResult
    def _run_phase(self, phase: WorkflowPhase, context: Dict) -> PhaseResult
    def _handle_failure(self, phase, result, workflow, context) -> FailureAction
```

## 3. Session Persistence & Cost Tracking

### Purpose

Persist session state for resumption. Track cost per session, per goal type, per model tier for trend analysis and router feedback.

### Database

File: `memory/agent_sdk_sessions.db` (SQLite, WAL mode, create-tables-if-not-exists on init, graceful fallback on corruption)

**sessions table:**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | TEXT UNIQUE | Agent SDK session ID |
| goal | TEXT | Original goal text |
| goal_type | TEXT | Classified goal type |
| workflow | TEXT | Workflow template name |
| model_tier | TEXT | Starting model tier |
| status | TEXT | active, completed, failed, paused |
| total_cost_usd | REAL | Running total |
| total_input_tokens | INTEGER | Running total |
| total_output_tokens | INTEGER | Running total |
| phases_completed | INTEGER | Count |
| resumed_count | INTEGER | Times resumed |
| error_summary | TEXT | Last error if failed |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |

**cycle_events table:**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_pk | INTEGER FK | References sessions.id |
| phase | TEXT | Workflow phase name |
| tool_name | TEXT | Tool that was called |
| model_used | TEXT | Model ID for this call |
| input_tokens | INTEGER | Tokens in |
| output_tokens | INTEGER | Tokens out |
| cost_usd | REAL | Computed cost |
| elapsed_ms | INTEGER | Wall clock time |
| success | BOOLEAN | Did this phase succeed |
| error_msg | TEXT | Error details if failed |
| created_at | TEXT | ISO timestamp |

### Cost Calculation

```python
COST_PER_1M = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
}

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1M[model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
```

### Session Manager

File: `core/agent_sdk/session_persistence.py` (named to avoid collision with `core/sadd/session_store.py`)

```python
class SessionStore:
    def __init__(self, db_path: Path)
    def create_session(self, session_id, goal, goal_type, workflow, model_tier) -> int
    def record_event(self, session_pk, phase, tool_name, model, tokens_in, tokens_out, success, error) -> None
    def update_status(self, session_pk, status, error_summary=None) -> None
    def get_session(self, session_id) -> Optional[Dict]
    def list_sessions(self, status=None, limit=20) -> List[Dict]
    def get_cost_summary(self, days=7) -> Dict  # per goal_type, per model
    def get_resumable(self) -> List[Dict]  # paused sessions
```

### CLI Commands

| Command | Action |
|---------|--------|
| `agent run --goal "..."` | Create session, execute, persist |
| `agent resume --session-id <id>` | Load session, resume via SDK |
| `agent status` | List recent sessions with status/cost |
| `agent cost --last 7d` | Cost breakdown by goal type and model |

### Hook Integration

Cost tracking uses a **separate callback on the Agent SDK message stream**, not the PostToolUse hook. The Agent SDK emits `AssistantMessage` objects with a `usage` dict (`input_tokens`, `output_tokens`) on each turn. The controller's `_execute_with_sdk()` method reads `message.usage` from each `AssistantMessage` and calls `session_store.record_event()`. This is more reliable than trying to extract tokens from hook callbacks, which don't receive usage data.

The existing `PostToolUse` hook continues to record tool-level metrics (name, success, timing) to `MetricsCollector` as before. The `Stop` hook calls `session_store.update_status()` to finalize the session.

## 4. Feedback Loop & Learning

### Purpose

Outcomes feed back to improve future decisions across three systems: model router, skill weights, and brain memory.

### Feedback Chain

```
Goal completes (success or failure)
  │
  ├─→ AdaptiveModelRouter.record_outcome(goal_type, model, success)
  │     Updates EMA score, consecutive counters
  │
  ├─→ SkillWeightUpdater.update(skills_used, success)
  │     Adjusts weights in memory/skill_weights.json
  │     +0.1 on success (cap 1.0), -0.05 on failure (floor 0.1)
  │
  └─→ Brain.remember({goal, outcome, model, cost, learnings})
        Stores for future search_memory retrieval
```

### Skill Weight Updater

File: `core/agent_sdk/feedback.py`

```python
class SkillWeightUpdater:
    def __init__(self, weights_path: Path)
    def update(self, skills_used: List[str], success: bool) -> None
    def get_weights(self) -> Dict[str, float]

class FeedbackCollector:
    def __init__(self, model_router, skill_updater, brain, session_store)
        """
        session_store is used by get_failure_patterns() to query recent
        session history. on_goal_complete() takes session_pk only to
        pass it to session_store.update_status() — the collector does
        NOT query the store using the pk; all other data comes from args.
        """
    def on_goal_complete(self, session_pk, goal, goal_type, model, skills_used, 
                         success, verification_result, cost) -> Dict
        """Dispatches to all three feedback systems:
        1. model_router.record_outcome(goal_type, model, success)
        2. skill_updater.update(skills_used, success)
        3. brain.remember({goal, outcome, model, cost, learnings})
        Returns: summary dict of what was updated.
        """
    def get_failure_patterns(self, goal_type, limit=3) -> List[str]
        """Queries session_store.list_sessions(goal_type) for recent failures,
        extracts error_summary patterns. Returns empty list if < 3 failures."""
```

### Failure Pattern Detection

- Query session_store for last N sessions of this goal_type
- If 3+ consecutive failures: extract common error patterns
- Inject `failure_pattern_warning` into system prompt via context_builder
- Warning format: "Recent {goal_type} goals have failed due to: {patterns}. Consider: {suggestions}"

### Context Builder Enhancement

`ContextBuilder.build()` does NOT gain new constructor dependencies. Instead, the controller injects feedback data into the context dict *after* `build()` returns (see Section 5, step 1b). This keeps ContextBuilder free of circular dependencies.

The controller adds two fields to the context dict:
- `failure_patterns`: List[str] from `FeedbackCollector.get_failure_patterns()`
- `skill_weights`: Dict[str, float] — skills ordered by weight, high-weight first

`ContextBuilder.build_system_prompt()` is updated to render these fields when present in the context dict (they're optional — absent means no feedback data available yet).

## 5. Controller Enhancements

### Constructor Changes

`AuraController.__init__` gains optional parameters for the new subsystems. All are optional for backward compatibility — when `None`, the controller falls back to current behavior (no routing, no sessions, no feedback).

```python
class AuraController:
    def __init__(
        self,
        config: AgentSDKConfig,
        project_root: Path,
        brain: Any = None,
        model_adapter: Any = None,
        goal_queue: Any = None,
        goal_archive: Any = None,
        # NEW optional dependencies:
        model_router: Optional[AdaptiveModelRouter] = None,
        workflow_executor: Optional[WorkflowExecutor] = None,
        session_store: Optional[SessionStore] = None,
        feedback: Optional[FeedbackCollector] = None,
    )
```

### Signature Changes

`_build_options(self, goal, model=None, resume=None)` — `model` and `resume` become optional kwargs (default `None`). When `model` is None, uses `self.config.model` (existing behavior). When `resume` is not None, passes it as `resume=session_id` to `ClaudeAgentOptions`.

`run(self, goal, resume_session_id=None)` — `resume_session_id` is an optional kwarg. Existing callers with `run(goal)` are unaffected.

`run_with_client(self, goal, resume_session_id=None)` — same change. Both methods support resume.

### Updated Run Flow

```python
async def run(self, goal: str, resume_session_id: str = None) -> Dict:
    # 1. Build context (now includes failure patterns + skill weights)
    context = self.context_builder.build(goal)
    
    # 1b. Enrich context with feedback data (if feedback collector available)
    if self.feedback:
        context["failure_patterns"] = self.feedback.get_failure_patterns(context["goal_type"])
        context["skill_weights"] = self.feedback.skill_updater.get_weights()
    
    # 2. Select model via adaptive router (or use config default)
    if self.model_router:
        model = self.model_router.select_model(context["goal_type"])
    else:
        model = self.config.model
    
    # 3. Select workflow template (or None for freeform)
    workflow = None
    if self.workflow_executor:
        workflow = self.workflow_executor.select_workflow(context["goal_type"])
    
    # 4. Create or resume session
    session_pk = None
    if self.session_store:
        if resume_session_id:
            session = self.session_store.get_session(resume_session_id)
            session_pk = session["id"] if session else None
        else:
            session_pk = self.session_store.create_session(
                session_id="pending",  # updated after SDK init
                goal=goal, goal_type=context["goal_type"],
                workflow=workflow.name if workflow else "freeform",
                model_tier=model,
            )
    
    # 5. Build options (backward-compatible signature)
    options = self._build_options(goal, model=model, resume=resume_session_id)
    
    # 6. Execute via Agent SDK
    result = await self._execute_with_sdk(goal, options, workflow, context)
    
    # 7. Record outcome and trigger feedback
    success = result.get("success", False)
    if self.feedback and session_pk:
        self.feedback.on_goal_complete(
            session_pk=session_pk, goal=goal,
            goal_type=context["goal_type"], model=model,
            skills_used=context.get("recommended_skills", []),
            success=success,
            verification_result=result.get("verification", {}),
            cost=result.get("total_cost_usd", 0.0),
        )
    
    # 8. Update session status
    if self.session_store and session_pk:
        self.session_store.update_status(
            session_pk, 
            "completed" if success else "failed",
            error_summary=result.get("error"),
        )
    
    return result
```

### Budget Enforcement

The `PostToolUse` hook tracks running cost. When `total_cost_usd` exceeds `config.max_budget_usd`, the hook injects a warning into the next tool call context. The controller checks cost after each phase and pauses the session (status → `paused`) if budget is exhausted, allowing the user to resume with a higher budget.

### System Prompt Enhancement

The system prompt now includes:
- Workflow template (which phases to execute and in what order)
- Model tier info ("You are running on Sonnet. If you encounter complexity beyond your capability, say so.")
- Failure pattern warnings (if any)
- Skill weights (prioritized skill list)
- Cost awareness ("Current session cost: $X.XX. Budget remaining: $Y.YY")

## 6. Config Additions

`AgentSDKConfig` gains these new fields (all with sensible defaults):

```python
# Model router config
model_stats_path: Path = Path("memory/agent_sdk_model_stats.json")
escalation_threshold: int = 2        # consecutive failures before escalation
de_escalation_threshold: int = 5     # consecutive successes before de-escalation
min_success_rate: float = 0.7        # minimum EMA score to select a tier
ema_alpha: float = 0.2               # EMA learning rate

# Session persistence
session_db_path: Path = Path("memory/agent_sdk_sessions.db")

# Skill weight updater
skill_weights_path: Path = Path("memory/skill_weights.json")
skill_weight_success_delta: float = 0.1
skill_weight_failure_delta: float = -0.05
skill_weight_cap: float = 1.0
skill_weight_floor: float = 0.1
```

`from_aura_config()` reads these from `aura_config.get("agent_sdk", {})` with the above defaults.

## 7. Error Resilience

- **Model stats file:** `AdaptiveModelRouter.__init__` handles missing file (create empty stats → default to "standard" tier) and corrupt JSON (log warning, fall back to empty stats). File writes use atomic temp-file + rename.
- **Session DB:** `SessionPersistence.__init__` runs `CREATE TABLE IF NOT EXISTS` on init. WAL mode for concurrent reads. On locked-DB error, retry once after 100ms, then log and continue without persistence.
- **De-escalation vs selection threshold:** De-escalation (5 consecutive successes) only *suggests* trying a lower tier. The selection algorithm's `min_success_rate >= 0.7` threshold still applies — if the lower tier has poor stats, it won't be selected regardless of de-escalation signal.

## File Structure (New + Modified)

```
core/agent_sdk/
├── model_router.py          # NEW: Adaptive model tier selection
├── workflow_templates.py     # NEW: Workflow template definitions + executor
├── session_persistence.py    # NEW: SQLite session persistence + cost tracking
├── feedback.py               # NEW: Feedback collector + skill weight updater
├── controller.py             # MODIFY: Wire in router, workflows, sessions, feedback
├── context_builder.py        # MODIFY: Render failure_patterns + skill_weights in prompt
├── hooks.py                  # MODIFY: Stop hook finalizes session status
├── config.py                 # MODIFY: Add router/session/feedback config fields
├── cli_integration.py        # MODIFY: Add resume/status/cost commands
└── (existing unchanged)

tests/
├── test_agent_sdk_model_router.py
├── test_agent_sdk_workflow_templates.py
├── test_agent_sdk_session_persistence.py
├── test_agent_sdk_feedback.py
└── test_agent_sdk_controller_v2.py  # Enhanced controller tests
```

## Testing Strategy

- **Unit tests:** Each new module tested in isolation with mocks
- **Integration tests:** Full feedback chain (goal → router → workflow → session → feedback) with mock SDK
- **No live SDK required:** All tests mock the `claude_agent_sdk` import; the `_check_sdk()` pattern is already established
