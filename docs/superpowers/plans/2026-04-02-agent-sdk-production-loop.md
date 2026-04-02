# Agent SDK Production-Grade Autonomous Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the Agent SDK meta-controller into a production-grade autonomous loop with adaptive model routing, workflow templates, session persistence with cost tracking, and a feedback loop that learns from outcomes.

**Architecture:** Four new modules (model_router, workflow_templates, session_persistence, feedback) plus modifications to five existing modules (config, context_builder, controller, hooks, cli_integration). Each new module is independently testable. The controller wires them together via optional constructor params — backward compatible with existing code and tests.

**Tech Stack:** Python 3.10+, SQLite (WAL mode), JSON file persistence, existing `core/agent_sdk/` skeleton (8 modules, 45 passing tests).

**Spec:** `docs/superpowers/specs/2026-04-02-agent-sdk-production-loop-design.md`

---

## File Structure

**New files:**
| File | Responsibility |
|------|---------------|
| `core/agent_sdk/model_router.py` | Adaptive model tier selection with EMA scoring |
| `core/agent_sdk/workflow_templates.py` | Workflow dataclasses + executor with retry/escalation |
| `core/agent_sdk/session_persistence.py` | SQLite session store + cost tracking |
| `core/agent_sdk/feedback.py` | Skill weight updater + feedback collector |

**Modified files:**
| File | Change |
|------|--------|
| `core/agent_sdk/config.py` | Add router/session/feedback config fields |
| `core/agent_sdk/context_builder.py` | Render failure_patterns + skill_weights in prompt |
| `core/agent_sdk/controller.py` | Wire in router, workflows, sessions, feedback |
| `core/agent_sdk/hooks.py` | Stop hook finalizes session status |
| `core/agent_sdk/cli_integration.py` | Add resume/status/cost handlers + subsystem init |

**Test files:**
| File | What it tests |
|------|--------------|
| `tests/test_agent_sdk_model_router.py` | Tier selection, EMA updates, escalation/de-escalation |
| `tests/test_agent_sdk_workflow_templates.py` | Template selection, phase execution, failure handling |
| `tests/test_agent_sdk_session_persistence.py` | SQLite CRUD, cost computation, resumable queries |
| `tests/test_agent_sdk_feedback.py` | Skill weight updates, feedback dispatch, failure patterns |
| `tests/test_agent_sdk_controller_v2.py` | Enhanced controller with all subsystems wired |

---

## Task 1: Extend Config with New Fields

**Files:**
- Modify: `core/agent_sdk/config.py:27-58`
- Test: `tests/test_agent_sdk_config.py` (add new tests)

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_agent_sdk_config.py

class TestAgentSDKConfigV2(unittest.TestCase):
    """Test new config fields for production loop."""

    def test_default_config_has_router_fields(self):
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        self.assertEqual(config.escalation_threshold, 2)
        self.assertEqual(config.de_escalation_threshold, 5)
        self.assertAlmostEqual(config.min_success_rate, 0.7)
        self.assertAlmostEqual(config.ema_alpha, 0.2)

    def test_default_config_has_session_fields(self):
        from core.agent_sdk.config import AgentSDKConfig
        from pathlib import Path
        config = AgentSDKConfig()
        self.assertEqual(config.session_db_path, Path("memory/agent_sdk_sessions.db"))
        self.assertEqual(config.model_stats_path, Path("memory/agent_sdk_model_stats.json"))

    def test_default_config_has_skill_weight_fields(self):
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        self.assertAlmostEqual(config.skill_weight_success_delta, 0.1)
        self.assertAlmostEqual(config.skill_weight_failure_delta, -0.05)
        self.assertAlmostEqual(config.skill_weight_cap, 1.0)
        self.assertAlmostEqual(config.skill_weight_floor, 0.1)

    def test_from_aura_config_reads_new_fields(self):
        from core.agent_sdk.config import AgentSDKConfig
        aura_config = {
            "agent_sdk": {
                "escalation_threshold": 3,
                "ema_alpha": 0.3,
            }
        }
        config = AgentSDKConfig.from_aura_config(aura_config)
        self.assertEqual(config.escalation_threshold, 3)
        self.assertAlmostEqual(config.ema_alpha, 0.3)
        # Unchanged defaults
        self.assertEqual(config.de_escalation_threshold, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_config.py::TestAgentSDKConfigV2 -v`
Expected: FAIL — `AttributeError: AgentSDKConfig has no attribute 'escalation_threshold'`

- [ ] **Step 3: Add new fields to AgentSDKConfig**

In `core/agent_sdk/config.py`, add after line 40 (`enable_hooks: bool = True`):

```python
    # Model router config
    model_stats_path: Path = field(default_factory=lambda: Path("memory/agent_sdk_model_stats.json"))
    escalation_threshold: int = 2
    de_escalation_threshold: int = 5
    min_success_rate: float = 0.7
    ema_alpha: float = 0.2
    # Session persistence
    session_db_path: Path = field(default_factory=lambda: Path("memory/agent_sdk_sessions.db"))
    # Skill weight updater
    skill_weights_path: Path = field(default_factory=lambda: Path("memory/skill_weights.json"))
    skill_weight_success_delta: float = 0.1
    skill_weight_failure_delta: float = -0.05
    skill_weight_cap: float = 1.0
    skill_weight_floor: float = 0.1
```

Add `from pathlib import Path` to imports at top of file.

Update `from_aura_config()` to read the new fields — add after the existing `enable_hooks` line in the `cls(...)` call:

```python
            model_stats_path=Path(sdk_section.get("model_stats_path", "memory/agent_sdk_model_stats.json")),
            escalation_threshold=sdk_section.get("escalation_threshold", 2),
            de_escalation_threshold=sdk_section.get("de_escalation_threshold", 5),
            min_success_rate=sdk_section.get("min_success_rate", 0.7),
            ema_alpha=sdk_section.get("ema_alpha", 0.2),
            session_db_path=Path(sdk_section.get("session_db_path", "memory/agent_sdk_sessions.db")),
            skill_weights_path=Path(sdk_section.get("skill_weights_path", "memory/skill_weights.json")),
            skill_weight_success_delta=sdk_section.get("skill_weight_success_delta", 0.1),
            skill_weight_failure_delta=sdk_section.get("skill_weight_failure_delta", -0.05),
            skill_weight_cap=sdk_section.get("skill_weight_cap", 1.0),
            skill_weight_floor=sdk_section.get("skill_weight_floor", 0.1),
```

- [ ] **Step 4: Run ALL config tests**

Run: `python3 -m pytest tests/test_agent_sdk_config.py -v`
Expected: All 9 tests PASS (5 existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/config.py tests/test_agent_sdk_config.py
git commit -m "feat: extend AgentSDKConfig with router, session, and feedback fields"
```

---

## Task 2: Adaptive Model Router

**Files:**
- Create: `core/agent_sdk/model_router.py`
- Test: `tests/test_agent_sdk_model_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_sdk_model_router.py
"""Tests for adaptive model router."""
import json
import os
import tempfile
import unittest
from pathlib import Path


class TestAdaptiveModelRouter(unittest.TestCase):
    """Test model tier selection and learning."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.stats_path = Path(self.tmpdir) / "model_stats.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_selection_is_standard(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_selects_cheapest_viable_tier(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        # Pre-seed stats: fast tier has good success rate
        stats = {
            "bug_fix": {
                "fast": {"attempts": 10, "successes": 9, "consecutive_failures": 0,
                         "consecutive_successes": 5, "ema_score": 0.9},
            }
        }
        self.stats_path.write_text(json.dumps(stats))
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-haiku-4-5")

    def test_skips_tier_with_low_ema(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        stats = {
            "bug_fix": {
                "fast": {"attempts": 10, "successes": 3, "consecutive_failures": 1,
                         "consecutive_successes": 0, "ema_score": 0.4},
                "standard": {"attempts": 20, "successes": 18, "consecutive_failures": 0,
                             "consecutive_successes": 6, "ema_score": 0.88},
            }
        }
        self.stats_path.write_text(json.dumps(stats))
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_record_outcome_updates_ema(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path, ema_alpha=0.2)
        router.record_outcome("bug_fix", "claude-sonnet-4-6", success=True)
        router.record_outcome("bug_fix", "claude-sonnet-4-6", success=True)
        stats = router.get_stats()
        tier_stats = stats["bug_fix"]["standard"]
        self.assertEqual(tier_stats["attempts"], 2)
        self.assertEqual(tier_stats["successes"], 2)
        self.assertGreater(tier_stats["ema_score"], 0.5)

    def test_record_outcome_persists_to_file(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        router.record_outcome("feature", "claude-sonnet-4-6", success=True)
        # Re-load from file
        router2 = AdaptiveModelRouter(stats_path=self.stats_path)
        stats = router2.get_stats()
        self.assertIn("feature", stats)

    def test_escalate_returns_next_tier(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        next_model = router.escalate("bug_fix", "claude-haiku-4-5")
        self.assertEqual(next_model, "claude-sonnet-4-6")
        next_model = router.escalate("bug_fix", "claude-sonnet-4-6")
        self.assertEqual(next_model, "claude-opus-4-6")
        # Already at top — stays there
        next_model = router.escalate("bug_fix", "claude-opus-4-6")
        self.assertEqual(next_model, "claude-opus-4-6")

    def test_consecutive_failures_trigger_skip(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        stats = {
            "bug_fix": {
                "fast": {"attempts": 5, "successes": 4, "consecutive_failures": 2,
                         "consecutive_successes": 0, "ema_score": 0.75},
            }
        }
        self.stats_path.write_text(json.dumps(stats))
        router = AdaptiveModelRouter(stats_path=self.stats_path, escalation_threshold=2)
        model = router.select_model("bug_fix")
        # Should skip fast despite good EMA because consecutive_failures >= threshold
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_handles_missing_stats_file(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        missing_path = Path(self.tmpdir) / "nonexistent.json"
        router = AdaptiveModelRouter(stats_path=missing_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_handles_corrupt_stats_file(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        self.stats_path.write_text("not valid json {{{")
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_model_router.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/model_router.py
"""Adaptive model tier selection with EMA-based learning.

Selects the cheapest model tier that can reliably handle each goal type,
learning from historical outcomes. Persists stats to JSON file.
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tier ordering: cheapest first
MODEL_TIERS: List[Dict[str, str]] = [
    {"tier": "fast", "model": "claude-haiku-4-5"},
    {"tier": "standard", "model": "claude-sonnet-4-6"},
    {"tier": "powerful", "model": "claude-opus-4-6"},
]

MODEL_TO_TIER: Dict[str, str] = {t["model"]: t["tier"] for t in MODEL_TIERS}
TIER_TO_MODEL: Dict[str, str] = {t["tier"]: t["model"] for t in MODEL_TIERS}
TIER_ORDER: List[str] = [t["tier"] for t in MODEL_TIERS]

_EMPTY_TIER_STATS = {
    "attempts": 0, "successes": 0,
    "consecutive_failures": 0, "consecutive_successes": 0,
    "ema_score": 0.5,
}


class AdaptiveModelRouter:
    """Select model tier based on historical goal-type performance."""

    def __init__(
        self,
        stats_path: Path,
        ema_alpha: float = 0.2,
        min_success_rate: float = 0.7,
        escalation_threshold: int = 2,
        de_escalation_threshold: int = 5,
    ) -> None:
        self._stats_path = stats_path
        self._alpha = ema_alpha
        self._min_rate = min_success_rate
        self._esc_threshold = escalation_threshold
        self._deesc_threshold = de_escalation_threshold
        self._stats: Dict[str, Dict[str, Dict[str, Any]]] = self._load()

    def _load(self) -> Dict:
        """Load stats from file. Returns empty dict on missing/corrupt file."""
        if not self._stats_path.exists():
            return {}
        try:
            return json.loads(self._stats_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt model stats at %s: %s", self._stats_path, exc)
            return {}

    def _save(self) -> None:
        """Atomically persist stats to file."""
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(self._stats_path.parent), suffix=".tmp"
            )
            with open(fd, "w") as f:
                json.dump(self._stats, f, indent=2)
            Path(tmp).replace(self._stats_path)
        except OSError as exc:
            logger.warning("Failed to save model stats: %s", exc)
            if tmp and Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)

    def _get_tier_stats(self, goal_type: str, tier: str) -> Dict[str, Any]:
        """Get stats for a goal_type + tier, creating defaults if absent."""
        return self._stats.get(goal_type, {}).get(tier, dict(_EMPTY_TIER_STATS))

    def select_model(self, goal_type: str) -> str:
        """Select the cheapest viable model tier for a goal type.

        Checks tiers cheapest-first. A tier qualifies if:
        - EMA score >= min_success_rate
        - consecutive_failures < escalation_threshold

        De-escalation: if a higher tier has consecutive_successes >= threshold,
        we also check the tier below it (already covered by cheapest-first order,
        but de-escalation resets the consecutive_failures counter for the lower
        tier to give it another chance).
        """
        for i, entry in enumerate(MODEL_TIERS):
            tier = entry["tier"]
            stats = self._get_tier_stats(goal_type, tier)
            ema = stats.get("ema_score", 0.5)
            consec_fail = stats.get("consecutive_failures", 0)

            # Check de-escalation: if a HIGHER tier has enough successes,
            # reset this tier's consecutive failure count to give it a chance
            if i < len(MODEL_TIERS) - 1:
                higher_tier = MODEL_TIERS[i + 1]["tier"]
                higher_stats = self._get_tier_stats(goal_type, higher_tier)
                if higher_stats.get("consecutive_successes", 0) >= self._deesc_threshold:
                    consec_fail = 0  # give lower tier another shot

            if ema >= self._min_rate and consec_fail < self._esc_threshold:
                return entry["model"]
        # Fallback: most powerful
        return TIER_TO_MODEL["powerful"]

    def record_outcome(
        self, goal_type: str, model: str, success: bool
    ) -> None:
        """Record a goal outcome and update EMA + counters."""
        tier = MODEL_TO_TIER.get(model, "standard")
        if goal_type not in self._stats:
            self._stats[goal_type] = {}
        if tier not in self._stats[goal_type]:
            self._stats[goal_type][tier] = dict(_EMPTY_TIER_STATS)

        s = self._stats[goal_type][tier]
        s["attempts"] += 1
        outcome = 1.0 if success else 0.0
        s["ema_score"] = self._alpha * outcome + (1 - self._alpha) * s["ema_score"]

        if success:
            s["successes"] += 1
            s["consecutive_successes"] += 1
            s["consecutive_failures"] = 0
        else:
            s["consecutive_failures"] += 1
            s["consecutive_successes"] = 0

        self._save()

    def escalate(self, goal_type: str, current_model: str) -> str:
        """Return the next tier up from current_model."""
        current_tier = MODEL_TO_TIER.get(current_model, "standard")
        idx = TIER_ORDER.index(current_tier) if current_tier in TIER_ORDER else 1
        next_idx = min(idx + 1, len(TIER_ORDER) - 1)
        return TIER_TO_MODEL[TIER_ORDER[next_idx]]

    def get_stats(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return the full stats dict for display."""
        return dict(self._stats)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_sdk_model_router.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/model_router.py tests/test_agent_sdk_model_router.py
git commit -m "feat: add adaptive model router with EMA-based learning"
```

---

## Task 3: Workflow Templates

**Files:**
- Create: `core/agent_sdk/workflow_templates.py`
- Test: `tests/test_agent_sdk_workflow_templates.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_sdk_workflow_templates.py
"""Tests for workflow templates and executor."""
import enum
import unittest
from unittest.mock import MagicMock


class TestWorkflowDataclasses(unittest.TestCase):
    """Test workflow phase and template dataclasses."""

    def test_workflow_phase_defaults(self):
        from core.agent_sdk.workflow_templates import WorkflowPhase
        phase = WorkflowPhase(tool_name="analyze_goal")
        self.assertTrue(phase.required)
        self.assertEqual(phase.retry_on_fail, 0)
        self.assertFalse(phase.escalate_on_fail)

    def test_workflow_template_has_phases(self):
        from core.agent_sdk.workflow_templates import WorkflowTemplate, WorkflowPhase
        wf = WorkflowTemplate(
            name="test",
            goal_types=["bug_fix"],
            phases=[WorkflowPhase(tool_name="analyze_goal")],
        )
        self.assertEqual(len(wf.phases), 1)
        self.assertEqual(wf.verification_mode, "post")

    def test_phase_result_fields(self):
        from core.agent_sdk.workflow_templates import PhaseResult
        pr = PhaseResult(tool_name="create_plan", success=True, output={"steps": []})
        self.assertTrue(pr.success)
        self.assertIsNone(pr.error)

    def test_workflow_result_fields(self):
        from core.agent_sdk.workflow_templates import WorkflowResult
        wr = WorkflowResult(success=True, phases_completed=3, phase_results=[])
        self.assertAlmostEqual(wr.total_cost_usd, 0.0)
        self.assertEqual(wr.model_escalations, 0)

    def test_failure_action_enum(self):
        from core.agent_sdk.workflow_templates import FailureAction
        self.assertEqual(FailureAction.RETRY_PHASE.value, "retry_phase")
        self.assertEqual(FailureAction.ABORT.value, "abort")


class TestBuiltinTemplates(unittest.TestCase):
    """Test the three built-in workflow templates."""

    def test_get_builtin_templates_returns_three(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        templates = get_builtin_templates()
        self.assertEqual(len(templates), 3)
        self.assertIn("bug_fix", templates)
        self.assertIn("feature", templates)
        self.assertIn("refactor", templates)

    def test_bug_fix_has_correct_phases(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        wf = get_builtin_templates()["bug_fix"]
        names = [p.tool_name for p in wf.phases]
        self.assertEqual(names[0], "analyze_goal")
        self.assertIn("generate_code", names)
        self.assertIn("verify_changes", names)
        self.assertIn("reflect_on_outcome", names)

    def test_feature_includes_critique(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        wf = get_builtin_templates()["feature"]
        names = [p.tool_name for p in wf.phases]
        self.assertIn("critique_plan", names)

    def test_refactor_has_pre_and_post_verification(self):
        from core.agent_sdk.workflow_templates import get_builtin_templates
        wf = get_builtin_templates()["refactor"]
        self.assertEqual(wf.verification_mode, "pre_and_post")


class TestWorkflowExecutor(unittest.TestCase):
    """Test workflow execution engine."""

    def test_select_workflow_by_goal_type(self):
        from core.agent_sdk.workflow_templates import WorkflowExecutor, get_builtin_templates
        executor = WorkflowExecutor(templates=get_builtin_templates(), tool_handlers={})
        wf = executor.select_workflow("bug_fix")
        self.assertEqual(wf.name, "bug_fix")

    def test_select_workflow_falls_back_to_feature(self):
        from core.agent_sdk.workflow_templates import WorkflowExecutor, get_builtin_templates
        executor = WorkflowExecutor(templates=get_builtin_templates(), tool_handlers={})
        wf = executor.select_workflow("unknown_type")
        self.assertEqual(wf.name, "feature")

    def test_execute_runs_all_phases(self):
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase, get_builtin_templates,
        )
        mock_handler = MagicMock(return_value={"success": True})
        handlers = {"analyze_goal": mock_handler, "store_memory": mock_handler}
        simple_wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="store_memory"),
            ],
        )
        executor = WorkflowExecutor(templates={"test": simple_wf}, tool_handlers=handlers)
        result = executor.execute(simple_wf, goal="test goal", context={})
        self.assertTrue(result.success)
        self.assertEqual(result.phases_completed, 2)

    def test_execute_handles_phase_failure(self):
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase,
        )
        fail_handler = MagicMock(return_value={"error": "broken"})
        handlers = {"bad_tool": fail_handler}
        wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[WorkflowPhase(tool_name="bad_tool")],
            max_retries_total=0,
        )
        executor = WorkflowExecutor(templates={"test": wf}, tool_handlers=handlers)
        result = executor.execute(wf, goal="test", context={})
        self.assertFalse(result.success)

    def test_execute_retries_phase(self):
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase,
        )
        call_count = {"n": 0}
        def flaky_handler(args):
            call_count["n"] += 1
            if call_count["n"] < 2:
                return {"error": "transient"}
            return {"result": "ok"}

        handlers = {"flaky": flaky_handler}
        wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[WorkflowPhase(tool_name="flaky", retry_on_fail=2)],
        )
        executor = WorkflowExecutor(templates={"test": wf}, tool_handlers=handlers)
        result = executor.execute(wf, goal="test", context={})
        self.assertTrue(result.success)
        self.assertEqual(call_count["n"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_workflow_templates.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/workflow_templates.py
"""Workflow templates and executor for the Agent SDK meta-controller.

Defines predefined phase sequences for common goal types (bug_fix, feature,
refactor) with retry policies and escalation rules.
"""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowPhase:
    """A single phase in a workflow template."""

    tool_name: str
    required: bool = True
    retry_on_fail: int = 0
    escalate_on_fail: bool = False


@dataclass
class WorkflowTemplate:
    """Ordered phase sequence for a goal type."""

    name: str
    goal_types: List[str]
    phases: List[WorkflowPhase]
    max_retries_total: int = 3
    verification_mode: str = "post"  # "post" | "pre_and_post" | "none"


@dataclass
class PhaseResult:
    """Result of executing a single workflow phase."""

    tool_name: str
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    model_used: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class WorkflowResult:
    """Result of executing a complete workflow."""

    success: bool
    phases_completed: int
    phase_results: List[PhaseResult]
    total_cost_usd: float = 0.0
    model_escalations: int = 0
    error_summary: Optional[str] = None


class FailureAction(enum.Enum):
    """What to do when a phase fails."""

    RETRY_PHASE = "retry_phase"
    ESCALATE_AND_RETRY = "escalate"
    REPLAN = "replan"
    ABORT = "abort"


def get_builtin_templates() -> Dict[str, WorkflowTemplate]:
    """Return the three built-in workflow templates."""
    return {
        "bug_fix": WorkflowTemplate(
            name="bug_fix",
            goal_types=["bug_fix"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="search_memory", required=False),
                WorkflowPhase(tool_name="dispatch_skills", required=False),
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="generate_code", retry_on_fail=3, escalate_on_fail=True),
                WorkflowPhase(tool_name="run_sandbox", retry_on_fail=2),
                WorkflowPhase(tool_name="apply_changes"),
                WorkflowPhase(tool_name="verify_changes"),
                WorkflowPhase(tool_name="reflect_on_outcome", required=False),
                WorkflowPhase(tool_name="store_memory", required=False),
            ],
            verification_mode="post",
        ),
        "feature": WorkflowTemplate(
            name="feature",
            goal_types=["feature", "default"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="search_memory", required=False),
                WorkflowPhase(tool_name="dispatch_skills", required=False),
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="critique_plan"),
                WorkflowPhase(tool_name="generate_code", retry_on_fail=2, escalate_on_fail=True),
                WorkflowPhase(tool_name="run_sandbox", retry_on_fail=2),
                WorkflowPhase(tool_name="apply_changes"),
                WorkflowPhase(tool_name="verify_changes"),
                WorkflowPhase(tool_name="reflect_on_outcome", required=False),
                WorkflowPhase(tool_name="store_memory", required=False),
            ],
            verification_mode="post",
        ),
        "refactor": WorkflowTemplate(
            name="refactor",
            goal_types=["refactor"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="search_memory", required=False),
                WorkflowPhase(tool_name="dispatch_skills", required=False),
                WorkflowPhase(tool_name="verify_changes"),  # baseline
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="generate_code", retry_on_fail=2),
                WorkflowPhase(tool_name="apply_changes"),
                WorkflowPhase(tool_name="verify_changes"),  # regression check
                WorkflowPhase(tool_name="reflect_on_outcome", required=False),
                WorkflowPhase(tool_name="store_memory", required=False),
            ],
            verification_mode="pre_and_post",
        ),
    }


class WorkflowExecutor:
    """Execute workflow templates by dispatching tool handlers in order."""

    def __init__(
        self,
        templates: Dict[str, WorkflowTemplate],
        tool_handlers: Dict[str, Callable[[Dict], Dict]],
    ) -> None:
        self._templates = templates
        self._handlers = tool_handlers

    def select_workflow(self, goal_type: str) -> WorkflowTemplate:
        """Select workflow by goal type, falling back to feature."""
        for wf in self._templates.values():
            if goal_type in wf.goal_types:
                return wf
        return self._templates.get("feature", list(self._templates.values())[0])

    def execute(
        self,
        workflow: WorkflowTemplate,
        goal: str,
        context: Dict[str, Any],
    ) -> WorkflowResult:
        """Execute all phases in order with retry support."""
        results: List[PhaseResult] = []
        total_retries = 0

        for phase in workflow.phases:
            pr = self._run_phase(phase, goal, context)
            results.append(pr)

            if pr.success:
                continue

            # Phase failed — try retries
            if not phase.required:
                continue  # skip optional failures

            retried = False
            for attempt in range(phase.retry_on_fail):
                if total_retries >= workflow.max_retries_total:
                    break
                total_retries += 1
                pr = self._run_phase(phase, goal, context)
                results.append(pr)
                if pr.success:
                    retried = True
                    break

            if not pr.success and phase.required:
                return WorkflowResult(
                    success=False,
                    phases_completed=len([r for r in results if r.success]),
                    phase_results=results,
                    error_summary=pr.error or f"Phase {phase.tool_name} failed",
                )

        return WorkflowResult(
            success=True,
            phases_completed=len([r for r in results if r.success]),
            phase_results=results,
        )

    def _run_phase(
        self, phase: WorkflowPhase, goal: str, context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute a single phase via its tool handler."""
        handler = self._handlers.get(phase.tool_name)
        if handler is None:
            return PhaseResult(
                tool_name=phase.tool_name,
                success=not phase.required,
                output={},
                error=f"No handler for {phase.tool_name}",
            )

        start = time.monotonic()
        try:
            output = handler({"goal": goal, **context})
            elapsed = (time.monotonic() - start) * 1000
            has_error = "error" in output and output["error"]
            return PhaseResult(
                tool_name=phase.tool_name,
                success=not has_error,
                output=output,
                error=output.get("error"),
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return PhaseResult(
                tool_name=phase.tool_name,
                success=False,
                output={},
                error=str(exc),
                elapsed_ms=elapsed,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_sdk_workflow_templates.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/workflow_templates.py tests/test_agent_sdk_workflow_templates.py
git commit -m "feat: add workflow templates with bug-fix, feature, and refactor sequences"
```

---

## Task 4: Session Persistence & Cost Tracking

**Files:**
- Create: `core/agent_sdk/session_persistence.py`
- Test: `tests/test_agent_sdk_session_persistence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_sdk_session_persistence.py
"""Tests for session persistence and cost tracking."""
import tempfile
import unittest
from pathlib import Path


class TestSessionStore(unittest.TestCase):
    """Test SQLite session persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_sessions.db"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_session(self):
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("sdk-123", "Fix bug", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        self.assertIsInstance(pk, int)
        self.assertGreater(pk, 0)

    def test_get_session(self):
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=self.db_path)
        store.create_session("sdk-456", "Add feature", "feature", "feature", "claude-sonnet-4-6")
        session = store.get_session("sdk-456")
        self.assertIsNotNone(session)
        self.assertEqual(session["goal"], "Add feature")
        self.assertEqual(session["status"], "active")

    def test_update_status(self):
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("sdk-789", "Refactor", "refactor", "refactor", "claude-sonnet-4-6")
        store.update_status(pk, "completed")
        session = store.get_session("sdk-789")
        self.assertEqual(session["status"], "completed")

    def test_record_event(self):
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("sdk-e1", "Test", "default", "feature", "claude-sonnet-4-6")
        store.record_event(pk, "analyze_goal", "analyze_goal", "claude-sonnet-4-6", 100, 50, True, None)
        session = store.get_session("sdk-e1")
        self.assertGreater(session["total_input_tokens"], 0)

    def test_list_sessions(self):
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=self.db_path)
        store.create_session("s1", "Goal 1", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        store.create_session("s2", "Goal 2", "feature", "feature", "claude-sonnet-4-6")
        sessions = store.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_list_sessions_by_status(self):
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=self.db_path)
        pk1 = store.create_session("s1", "G1", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        store.create_session("s2", "G2", "feature", "feature", "claude-sonnet-4-6")
        store.update_status(pk1, "completed")
        active = store.list_sessions(status="active")
        self.assertEqual(len(active), 1)

    def test_get_resumable(self):
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("s1", "Paused goal", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        store.update_status(pk, "paused")
        resumable = store.get_resumable()
        self.assertEqual(len(resumable), 1)
        self.assertEqual(resumable[0]["session_id"], "s1")


class TestCostComputation(unittest.TestCase):
    """Test cost calculation utility."""

    def test_compute_cost_sonnet(self):
        from core.agent_sdk.session_persistence import compute_cost
        cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 18.0)  # 3 + 15

    def test_compute_cost_haiku(self):
        from core.agent_sdk.session_persistence import compute_cost
        cost = compute_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 6.0)  # 1 + 5

    def test_compute_cost_small_usage(self):
        from core.agent_sdk.session_persistence import compute_cost
        cost = compute_cost("claude-sonnet-4-6", 1000, 500)
        expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
        self.assertAlmostEqual(cost, expected)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_session_persistence.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/session_persistence.py
"""SQLite session persistence and cost tracking.

Stores session state for resumption and tracks cost per session,
per goal type, per model tier.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COST_PER_1M = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a model call."""
    rates = COST_PER_1M.get(model, COST_PER_1M["claude-sonnet-4-6"])
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE,
    goal TEXT NOT NULL,
    goal_type TEXT NOT NULL,
    workflow TEXT NOT NULL,
    model_tier TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    phases_completed INTEGER NOT NULL DEFAULT 0,
    resumed_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_pk INTEGER NOT NULL REFERENCES sessions(id),
    phase TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    model_used TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    elapsed_ms INTEGER NOT NULL DEFAULT 0,
    success BOOLEAN NOT NULL DEFAULT 1,
    error_msg TEXT,
    created_at TEXT NOT NULL
);
"""


class SessionStore:
    """SQLite-backed session persistence with cost tracking."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def create_session(
        self, session_id: str, goal: str, goal_type: str,
        workflow: str, model_tier: str,
    ) -> int:
        """Create a new session. Returns the primary key."""
        now = datetime.utcnow().isoformat()
        cur = self._conn.execute(
            """INSERT INTO sessions (session_id, goal, goal_type, workflow,
               model_tier, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
            (session_id, goal, goal_type, workflow, model_tier, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_event(
        self, session_pk: int, phase: str, tool_name: str,
        model: str, tokens_in: int, tokens_out: int,
        success: bool, error: Optional[str],
    ) -> None:
        """Record a cycle event and update session totals."""
        cost = compute_cost(model, tokens_in, tokens_out)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO cycle_events (session_pk, phase, tool_name, model_used,
               input_tokens, output_tokens, cost_usd, success, error_msg, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_pk, phase, tool_name, model, tokens_in, tokens_out,
             cost, success, error, now),
        )
        self._conn.execute(
            """UPDATE sessions SET
               total_cost_usd = total_cost_usd + ?,
               total_input_tokens = total_input_tokens + ?,
               total_output_tokens = total_output_tokens + ?,
               phases_completed = phases_completed + ?,
               updated_at = ?
               WHERE id = ?""",
            (cost, tokens_in, tokens_out, 1 if success else 0, now, session_pk),
        )
        self._conn.commit()

    def update_status(
        self, session_pk: int, status: str, error_summary: Optional[str] = None,
    ) -> None:
        """Update session status."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """UPDATE sessions SET status = ?, error_summary = ?, updated_at = ?
               WHERE id = ?""",
            (status, error_summary, now, session_pk),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by SDK session ID."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self, status: Optional[str] = None, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """Cost breakdown by goal_type and model for recent sessions."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """SELECT goal_type, model_tier, COUNT(*) as count,
               SUM(total_cost_usd) as total_cost
               FROM sessions WHERE created_at >= ?
               GROUP BY goal_type, model_tier""",
            (cutoff,),
        ).fetchall()
        return {"period_days": days, "breakdown": [dict(r) for r in rows]}

    def get_resumable(self) -> List[Dict[str, Any]]:
        """Get paused sessions that can be resumed."""
        return self.list_sessions(status="paused")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_sdk_session_persistence.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/session_persistence.py tests/test_agent_sdk_session_persistence.py
git commit -m "feat: add session persistence with SQLite and cost tracking"
```

---

## Task 5: Feedback Collector & Skill Weight Updater

**Files:**
- Create: `core/agent_sdk/feedback.py`
- Test: `tests/test_agent_sdk_feedback.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_sdk_feedback.py
"""Tests for feedback collector and skill weight updater."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestSkillWeightUpdater(unittest.TestCase):
    """Test skill weight adjustment."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.weights_path = Path(self.tmpdir) / "skill_weights.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_update_increases_on_success(self):
        from core.agent_sdk.feedback import SkillWeightUpdater
        self.weights_path.write_text(json.dumps({"linter": 0.5}))
        updater = SkillWeightUpdater(weights_path=self.weights_path)
        updater.update(["linter"], success=True)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 0.6)

    def test_update_decreases_on_failure(self):
        from core.agent_sdk.feedback import SkillWeightUpdater
        self.weights_path.write_text(json.dumps({"linter": 0.5}))
        updater = SkillWeightUpdater(weights_path=self.weights_path)
        updater.update(["linter"], success=False)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 0.45)

    def test_weight_capped_at_max(self):
        from core.agent_sdk.feedback import SkillWeightUpdater
        self.weights_path.write_text(json.dumps({"linter": 0.95}))
        updater = SkillWeightUpdater(weights_path=self.weights_path, cap=1.0, success_delta=0.1)
        updater.update(["linter"], success=True)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 1.0)

    def test_weight_floored_at_min(self):
        from core.agent_sdk.feedback import SkillWeightUpdater
        self.weights_path.write_text(json.dumps({"linter": 0.12}))
        updater = SkillWeightUpdater(weights_path=self.weights_path, floor=0.1, failure_delta=-0.05)
        updater.update(["linter"], success=False)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 0.1)

    def test_new_skill_starts_at_default(self):
        from core.agent_sdk.feedback import SkillWeightUpdater
        updater = SkillWeightUpdater(weights_path=self.weights_path)
        updater.update(["brand_new_skill"], success=True)
        weights = updater.get_weights()
        self.assertIn("brand_new_skill", weights)

    def test_handles_missing_file(self):
        from core.agent_sdk.feedback import SkillWeightUpdater
        updater = SkillWeightUpdater(weights_path=self.weights_path)
        weights = updater.get_weights()
        self.assertIsInstance(weights, dict)


class TestFeedbackCollector(unittest.TestCase):
    """Test feedback dispatch to all three systems."""

    def test_on_goal_complete_dispatches_all(self):
        from core.agent_sdk.feedback import FeedbackCollector
        mock_router = MagicMock()
        mock_updater = MagicMock()
        mock_updater.get_weights.return_value = {}
        mock_brain = MagicMock()
        mock_store = MagicMock()
        collector = FeedbackCollector(
            model_router=mock_router,
            skill_updater=mock_updater,
            brain=mock_brain,
            session_store=mock_store,
        )
        collector.on_goal_complete(
            session_pk=1, goal="Fix bug", goal_type="bug_fix",
            model="claude-sonnet-4-6", skills_used=["linter"],
            success=True, verification_result={"passed": True}, cost=0.5,
        )
        mock_router.record_outcome.assert_called_once_with("bug_fix", "claude-sonnet-4-6", True)
        mock_updater.update.assert_called_once_with(["linter"], True)
        mock_brain.remember.assert_called_once()

    def test_get_failure_patterns_empty_when_few_failures(self):
        from core.agent_sdk.feedback import FeedbackCollector
        mock_store = MagicMock()
        mock_store.list_sessions.return_value = [
            {"status": "failed", "error_summary": "test error"}
        ]
        collector = FeedbackCollector(
            model_router=MagicMock(), skill_updater=MagicMock(),
            brain=None, session_store=mock_store,
        )
        patterns = collector.get_failure_patterns("bug_fix")
        # Less than 3 failures, should return empty
        self.assertEqual(patterns, [])

    def test_get_failure_patterns_returns_errors(self):
        from core.agent_sdk.feedback import FeedbackCollector
        mock_store = MagicMock()
        mock_store.list_sessions.return_value = [
            {"status": "failed", "goal_type": "bug_fix", "error_summary": "ImportError: no module X"},
            {"status": "failed", "goal_type": "bug_fix", "error_summary": "ImportError: no module Y"},
            {"status": "failed", "goal_type": "bug_fix", "error_summary": "SyntaxError: bad token"},
        ]
        collector = FeedbackCollector(
            model_router=MagicMock(), skill_updater=MagicMock(),
            brain=None, session_store=mock_store,
        )
        patterns = collector.get_failure_patterns("bug_fix")
        self.assertEqual(len(patterns), 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_feedback.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/feedback.py
"""Feedback collector and skill weight updater.

Dispatches goal outcomes to three systems: model router (EMA updates),
skill weights (JSON file), and brain memory (semantic storage).
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHT = 0.5


class SkillWeightUpdater:
    """Adjust skill weights based on goal outcomes."""

    def __init__(
        self,
        weights_path: Path,
        success_delta: float = 0.1,
        failure_delta: float = -0.05,
        cap: float = 1.0,
        floor: float = 0.1,
    ) -> None:
        self._path = weights_path
        self._success_delta = success_delta
        self._failure_delta = failure_delta
        self._cap = cap
        self._floor = floor
        self._weights = self._load()

    def _load(self) -> Dict[str, float]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            with open(fd, "w") as f:
                json.dump(self._weights, f, indent=2)
            Path(tmp).replace(self._path)
        except OSError as exc:
            logger.warning("Failed to save skill weights: %s", exc)
            if tmp and Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)

    def update(self, skills_used: List[str], success: bool) -> None:
        """Adjust weights for used skills based on outcome."""
        delta = self._success_delta if success else self._failure_delta
        for skill in skills_used:
            current = self._weights.get(skill, _DEFAULT_WEIGHT)
            new_weight = max(self._floor, min(self._cap, current + delta))
            self._weights[skill] = round(new_weight, 4)
        self._save()

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)


class FeedbackCollector:
    """Dispatch goal outcomes to model router, skill weights, and brain."""

    def __init__(
        self,
        model_router: Any,
        skill_updater: SkillWeightUpdater,
        brain: Any = None,
        session_store: Any = None,
    ) -> None:
        self.model_router = model_router
        self.skill_updater = skill_updater
        self._brain = brain
        self._session_store = session_store

    def on_goal_complete(
        self,
        session_pk: int,
        goal: str,
        goal_type: str,
        model: str,
        skills_used: List[str],
        success: bool,
        verification_result: Dict[str, Any],
        cost: float,
    ) -> Dict[str, Any]:
        """Dispatch outcome to all three feedback systems."""
        # 1. Model router
        self.model_router.record_outcome(goal_type, model, success)

        # 2. Skill weights
        self.skill_updater.update(skills_used, success)

        # 3. Brain memory
        if self._brain is not None:
            try:
                self._brain.remember({
                    "type": "goal_outcome",
                    "goal": goal,
                    "goal_type": goal_type,
                    "model": model,
                    "success": success,
                    "cost_usd": cost,
                    "verification": verification_result,
                })
            except Exception as exc:
                logger.warning("Failed to store outcome in brain: %s", exc)

        return {
            "model_updated": True,
            "skills_updated": skills_used,
            "brain_stored": self._brain is not None,
        }

    def get_failure_patterns(self, goal_type: str, limit: int = 3) -> List[str]:
        """Extract error patterns from recent failed sessions."""
        if self._session_store is None:
            return []

        try:
            recent = self._session_store.list_sessions(status="failed", limit=10)
        except Exception:
            return []

        # Filter to this goal_type if possible
        failures = [
            s for s in recent
            if s.get("goal_type") == goal_type and s.get("error_summary")
        ]
        if not failures:
            failures = [s for s in recent if s.get("error_summary")]

        if len(failures) < limit:
            return []

        return [f["error_summary"] for f in failures[:limit]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_sdk_feedback.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/feedback.py tests/test_agent_sdk_feedback.py
git commit -m "feat: add feedback collector with skill weight updates and failure patterns"
```

---

## Task 6: Enhance Context Builder

**Files:**
- Modify: `core/agent_sdk/context_builder.py:150-170`
- Test: `tests/test_agent_sdk_context_builder.py` (add new tests)

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_agent_sdk_context_builder.py

class TestContextBuilderV2(unittest.TestCase):
    """Test enhanced context builder with feedback data."""

    def test_system_prompt_includes_failure_patterns(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = {
            "failure_patterns": ["ImportError: no module X", "SyntaxError: bad token"],
            "recommended_skills": ["linter"],
        }
        prompt = builder.build_system_prompt(goal="Fix bug", goal_type="bug_fix", context=ctx)
        self.assertIn("ImportError", prompt)
        self.assertIn("Failure Patterns", prompt)

    def test_system_prompt_includes_skill_weights(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = {
            "skill_weights": {"linter": 0.9, "type_checker": 0.3},
            "recommended_skills": ["linter"],
        }
        prompt = builder.build_system_prompt(goal="Fix bug", goal_type="bug_fix", context=ctx)
        self.assertIn("Skill Weights", prompt)
        self.assertIn("linter", prompt)

    def test_system_prompt_without_feedback_data(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = {"recommended_skills": ["linter"]}
        prompt = builder.build_system_prompt(goal="Fix bug", goal_type="bug_fix", context=ctx)
        self.assertNotIn("Failure Patterns", prompt)
        self.assertNotIn("Skill Weights", prompt)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py::TestContextBuilderV2 -v`
Expected: FAIL — assertions fail (new sections not rendered yet)

- [ ] **Step 3: Update build_system_prompt in context_builder.py**

In `core/agent_sdk/context_builder.py`, modify `build_system_prompt()` (lines 150-170). Add after the `recommended_skills` block (after line 163):

```python
            if context.get("failure_patterns"):
                patterns = "\n".join(f"- {p}" for p in context["failure_patterns"])
                parts.append(f"### Failure Patterns\nRecent failures for this goal type:\n{patterns}")
            if context.get("skill_weights"):
                sorted_skills = sorted(
                    context["skill_weights"].items(), key=lambda x: x[1], reverse=True
                )
                weights_str = ", ".join(f"{name} ({w:.1f})" for name, w in sorted_skills)
                parts.append(f"### Skill Weights\n{weights_str}")
```

- [ ] **Step 4: Run ALL context builder tests**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py -v`
Expected: All 10 tests PASS (7 existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/context_builder.py tests/test_agent_sdk_context_builder.py
git commit -m "feat: render failure patterns and skill weights in system prompt"
```

---

## Task 7: Enhance Controller

**Files:**
- Modify: `core/agent_sdk/controller.py`
- Test: `tests/test_agent_sdk_controller_v2.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_sdk_controller_v2.py
"""Tests for enhanced controller with production subsystems."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestControllerV2Init(unittest.TestCase):
    """Test enhanced controller construction."""

    def test_accepts_new_optional_deps(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(
            config=config,
            project_root=Path("/tmp/test"),
            model_router=MagicMock(),
            workflow_executor=MagicMock(),
            session_store=MagicMock(),
            feedback=MagicMock(),
        )
        self.assertIsNotNone(controller.model_router)
        self.assertIsNotNone(controller.workflow_executor)
        self.assertIsNotNone(controller.session_store)
        self.assertIsNotNone(controller.feedback)

    def test_backward_compatible_without_new_deps(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        self.assertIsNone(controller.model_router)
        self.assertIsNone(controller.session_store)


class TestControllerV2Options(unittest.TestCase):
    """Test enhanced _build_options."""

    def test_build_options_with_model_override(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        opts = controller._build_options("Fix bug", model="claude-opus-4-6")
        self.assertEqual(opts.model, "claude-opus-4-6")

    def test_build_options_without_model_uses_config(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig(model="claude-sonnet-4-6")
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        opts = controller._build_options("Fix bug")
        self.assertEqual(opts.model, "claude-sonnet-4-6")

    def test_context_builder_stored_as_attribute(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        self.assertIsNotNone(controller.context_builder)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_controller_v2.py -v`
Expected: FAIL — `AttributeError` for new fields

- [ ] **Step 3: Update controller.py**

Modify `core/agent_sdk/controller.py`:

1. Update `__init__` (line 40-56) to accept new optional deps and create context_builder:

```python
    def __init__(
        self,
        config: Any,
        project_root: Path,
        brain: Any = None,
        model_adapter: Any = None,
        goal_queue: Any = None,
        goal_archive: Any = None,
        model_router: Any = None,
        workflow_executor: Any = None,
        session_store: Any = None,
        feedback: Any = None,
    ) -> None:
        from core.agent_sdk.config import AgentSDKConfig
        from core.agent_sdk.context_builder import ContextBuilder

        self.config: AgentSDKConfig = config
        self.project_root = project_root
        self._brain = brain
        self._model_adapter = model_adapter
        self._goal_queue = goal_queue
        self._goal_archive = goal_archive
        self.model_router = model_router
        self.workflow_executor = workflow_executor
        self.session_store = session_store
        self.feedback = feedback
        self.context_builder = ContextBuilder(
            project_root=project_root, brain=brain,
        )
```

2. Update `_build_options` (line 143) to accept optional `model` and `resume`:

```python
    def _build_options(self, goal: str, model: str = None, resume: str = None) -> Any:
        """Build ClaudeAgentOptions for a session."""
        system_prompt = self._build_prompt(goal)
        mcp_server = self._build_mcp_server()
        subagents = self._build_subagent_defs()
        hooks = self._build_hooks()
        effective_model = model or self.config.model

        if not _check_sdk():
            class _MockOptions:
                pass
            opts = _MockOptions()
            opts.model = effective_model
            opts.max_turns = self.config.max_turns
            opts.system_prompt = system_prompt
            opts.agents = subagents
            return opts

        from claude_agent_sdk import ClaudeAgentOptions

        options_kwargs = dict(
            cwd=str(self.project_root),
            model=effective_model,
            max_turns=self.config.max_turns,
            max_budget_usd=self.config.max_budget_usd,
            permission_mode=self.config.permission_mode,
            allowed_tools=list(self.config.allowed_tools),
            system_prompt=system_prompt,
            mcp_servers={"aura": mcp_server},
            agents=subagents,
            hooks=hooks,
            thinking=self.config.thinking_config,
        )
        if resume:
            options_kwargs["resume"] = resume
        return ClaudeAgentOptions(**options_kwargs)
```

3. Replace `run()` (starting at line 180) with the full production flow:

```python
    async def run(self, goal: str, resume_session_id: str = None) -> Dict[str, Any]:
        """Execute a goal with adaptive routing, workflow selection, and feedback."""
        if not _check_sdk():
            raise RuntimeError(
                "claude-agent-sdk not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        from claude_agent_sdk import query, ResultMessage, SystemMessage, AssistantMessage
        from core.agent_sdk.hooks import get_session_metrics

        # 1. Build context
        context = self.context_builder.build(goal=goal)

        # 1b. Enrich with feedback data
        if self.feedback:
            context["failure_patterns"] = self.feedback.get_failure_patterns(context["goal_type"])
            context["skill_weights"] = self.feedback.skill_updater.get_weights()

        # 2. Select model via adaptive router
        model = self.model_router.select_model(context["goal_type"]) if self.model_router else self.config.model

        # 3. Select workflow template
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
                    session_id="pending",
                    goal=goal, goal_type=context["goal_type"],
                    workflow=workflow.name if workflow else "freeform",
                    model_tier=model,
                )

        # 5. Build options
        options = self._build_options(goal, model=model, resume=resume_session_id)

        # 6. Execute via Agent SDK
        session_id = None
        result_text = ""
        total_cost = 0.0

        prompt = (
            f"Execute this development goal:\n\n{goal}\n\n"
            "Start by calling analyze_goal to understand the context, "
            "then proceed with the appropriate workflow."
        )
        if workflow:
            phase_names = " → ".join(p.tool_name for p in workflow.phases)
            prompt += f"\n\nRecommended workflow ({workflow.name}): {phase_names}"

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
            elif isinstance(message, AssistantMessage) and message.usage:
                tokens_in = message.usage.get("input_tokens", 0)
                tokens_out = message.usage.get("output_tokens", 0)
                if self.session_store and session_pk:
                    from core.agent_sdk.session_persistence import compute_cost
                    cost = compute_cost(model, tokens_in, tokens_out)
                    total_cost += cost
                    self.session_store.record_event(
                        session_pk, "sdk_turn", "agent_sdk", model,
                        tokens_in, tokens_out, True, None,
                    )
            elif isinstance(message, ResultMessage):
                result_text = message.result

        metrics = get_session_metrics().get_summary()

        # 7. Record outcome and trigger feedback
        success = bool(result_text and "error" not in result_text.lower()[:100])
        if self.feedback and session_pk:
            self.feedback.on_goal_complete(
                session_pk=session_pk, goal=goal,
                goal_type=context["goal_type"], model=model,
                skills_used=context.get("recommended_skills", []),
                success=success,
                verification_result={},
                cost=total_cost,
            )

        # 8. Update session status
        if self.session_store and session_pk:
            self.session_store.update_status(
                session_pk,
                "completed" if success else "failed",
                error_summary=None if success else "Goal execution failed",
            )

        return {
            "result": result_text,
            "session_id": session_id,
            "metrics": metrics,
            "total_cost_usd": total_cost,
            "success": success,
        }
```

4. Update `run_with_client()` similarly — add `resume_session_id=None` param:

```python
    async def run_with_client(self, goal: str, resume_session_id: str = None) -> Dict[str, Any]:
```

- [ ] **Step 4: Run ALL controller tests**

Run: `python3 -m pytest tests/test_agent_sdk_controller.py tests/test_agent_sdk_controller_v2.py -v`
Expected: All 10 tests PASS (6 existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/controller.py tests/test_agent_sdk_controller_v2.py
git commit -m "feat: enhance controller with router, workflows, sessions, and feedback"
```

---

## Task 8: Enhance CLI Integration

**Files:**
- Modify: `core/agent_sdk/cli_integration.py`
- Modify: `core/agent_sdk/__init__.py`
- Test: `tests/test_agent_sdk_cli_integration.py` (add new tests)

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_agent_sdk_cli_integration.py

class TestCLIIntegrationV2(unittest.TestCase):
    """Test enhanced CLI with subsystem initialization."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_controller_initializes_subsystems(self):
        from core.agent_sdk.cli_integration import build_controller_from_args
        args = MagicMock()
        args.model = None
        args.max_turns = None
        args.max_budget = None
        args.permission_mode = None
        args.project_root = self.tmpdir
        # Patch config to use temp paths so no side effects in CWD
        with patch("core.agent_sdk.cli_integration.AgentSDKConfig") as MockConfig:
            from core.agent_sdk.config import AgentSDKConfig
            cfg = AgentSDKConfig(
                model_stats_path=Path(self.tmpdir) / "stats.json",
                session_db_path=Path(self.tmpdir) / "sessions.db",
                skill_weights_path=Path(self.tmpdir) / "weights.json",
            )
            MockConfig.return_value = cfg
            MockConfig.from_aura_config.return_value = cfg
            controller = build_controller_from_args(args)
        self.assertIsNotNone(controller.model_router)
        self.assertIsNotNone(controller.session_store)

    def test_format_result_shows_cost(self):
        from core.agent_sdk.cli_integration import format_result
        result = {
            "result": "Done.",
            "session_id": "abc-123",
            "total_cost_usd": 0.37,
            "metrics": {"total_calls": 5, "success_rate": 1.0},
        }
        output = format_result(result)
        self.assertIn("$0.37", output)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py::TestCLIIntegrationV2 -v`
Expected: FAIL

- [ ] **Step 3: Update cli_integration.py**

Update `build_controller_from_args()` to initialize subsystems, and `format_result()` to show cost:

In `build_controller_from_args()`, after the `model_adapter` try/except block (around line 56), add:

```python
    # Initialize production subsystems
    from core.agent_sdk.model_router import AdaptiveModelRouter
    from core.agent_sdk.session_persistence import SessionStore
    from core.agent_sdk.feedback import SkillWeightUpdater, FeedbackCollector

    model_router = AdaptiveModelRouter(
        stats_path=config.model_stats_path,
        ema_alpha=config.ema_alpha,
        min_success_rate=config.min_success_rate,
        escalation_threshold=config.escalation_threshold,
        de_escalation_threshold=config.de_escalation_threshold,
    )
    session_store = SessionStore(db_path=config.session_db_path)
    skill_updater = SkillWeightUpdater(
        weights_path=config.skill_weights_path,
        success_delta=config.skill_weight_success_delta,
        failure_delta=config.skill_weight_failure_delta,
        cap=config.skill_weight_cap,
        floor=config.skill_weight_floor,
    )
    feedback = FeedbackCollector(
        model_router=model_router,
        skill_updater=skill_updater,
        brain=brain,
        session_store=session_store,
    )
```

And update the `return AuraController(...)` call to pass the new deps:

```python
    return AuraController(
        config=config,
        project_root=project_root,
        brain=brain,
        model_adapter=model_adapter,
        model_router=model_router,
        session_store=session_store,
        feedback=feedback,
    )
```

In `format_result()`, add cost display:

```python
    if result.get("total_cost_usd"):
        parts.append(f"Cost: ${result['total_cost_usd']:.2f}")
```

- [ ] **Step 4: Run ALL CLI integration tests**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py -v`
Expected: All 5 tests PASS (3 existing + 2 new)

- [ ] **Step 5: Update __init__.py exports**

Keep `core/agent_sdk/__init__.py` lightweight — only eagerly import the two primary entry points. All other modules use lazy imports to avoid import-time SQLite/JSON overhead:

```python
from core.agent_sdk.config import AgentSDKConfig
from core.agent_sdk.controller import AuraController
from core.agent_sdk.cli_integration import build_controller_from_args, handle_agent_run

__all__ = [
    "AgentSDKConfig",
    "AuraController",
    "build_controller_from_args",
    "handle_agent_run",
]
```

- [ ] **Step 6: Commit**

```bash
git add core/agent_sdk/cli_integration.py core/agent_sdk/__init__.py tests/test_agent_sdk_cli_integration.py
git commit -m "feat: wire production subsystems into CLI integration"
```

---

## Task 9: CLI Commands (resume/status/cost) + Prompt Enhancements

**Files:**
- Modify: `core/agent_sdk/cli_integration.py`
- Modify: `core/agent_sdk/context_builder.py`
- Test: `tests/test_agent_sdk_cli_integration.py` (add new tests)

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_agent_sdk_cli_integration.py
import tempfile
from pathlib import Path

class TestCLICommands(unittest.TestCase):
    """Test resume, status, and cost CLI handlers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_handle_agent_status(self):
        from core.agent_sdk.cli_integration import handle_agent_status
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=Path(self.tmpdir) / "test.db")
        store.create_session("s1", "Fix bug", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        output = handle_agent_status(store, limit=10)
        self.assertIn("Fix bug", output)
        self.assertIn("active", output)

    def test_handle_agent_cost(self):
        from core.agent_sdk.cli_integration import handle_agent_cost
        from core.agent_sdk.session_persistence import SessionStore
        store = SessionStore(db_path=Path(self.tmpdir) / "test.db")
        pk = store.create_session("s1", "Fix bug", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        store.record_event(pk, "analyze", "analyze_goal", "claude-sonnet-4-6", 1000, 500, True, None)
        store.update_status(pk, "completed")
        output = handle_agent_cost(store, days=7)
        self.assertIn("bug_fix", output)

    def test_system_prompt_includes_workflow_info(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = {
            "recommended_skills": ["linter"],
            "workflow_info": "bug_fix: analyze_goal → create_plan → generate_code → verify_changes",
            "model_tier": "standard (claude-sonnet-4-6)",
        }
        prompt = builder.build_system_prompt(goal="Fix bug", goal_type="bug_fix", context=ctx)
        self.assertIn("Workflow", prompt)
        self.assertIn("analyze_goal", prompt)
        self.assertIn("Model Tier", prompt)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py::TestCLICommands -v`
Expected: FAIL

- [ ] **Step 3: Add CLI handler functions to cli_integration.py**

Add to `core/agent_sdk/cli_integration.py`:

```python
def handle_agent_status(session_store: Any, limit: int = 20) -> str:
    """Format recent sessions for CLI display."""
    sessions = session_store.list_sessions(limit=limit)
    if not sessions:
        return "No sessions found."
    lines = []
    for s in sessions:
        cost_str = f"${s['total_cost_usd']:.2f}" if s.get('total_cost_usd') else "$0.00"
        lines.append(f"  {s['session_id']}  {s['status']:<10} {cost_str:<8} {s['goal'][:50]}")
    header = f"{'Session ID':<14} {'Status':<10} {'Cost':<8} Goal"
    return f"{header}\n" + "\n".join(lines)


def handle_agent_cost(session_store: Any, days: int = 7) -> str:
    """Format cost summary for CLI display."""
    summary = session_store.get_cost_summary(days=days)
    if not summary.get("breakdown"):
        return f"No sessions in the last {days} days."
    lines = [f"Cost summary (last {days} days):"]
    for row in summary["breakdown"]:
        lines.append(
            f"  {row['goal_type']:<12} {row['model_tier']:<22} "
            f"{row['count']} sessions  ${row['total_cost']:.2f}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Add workflow_info and model_tier to context_builder prompt rendering**

In `core/agent_sdk/context_builder.py`, in `build_system_prompt()`, add after the `skill_weights` block:

```python
            if context.get("workflow_info"):
                parts.append(f"### Workflow\n{context['workflow_info']}")
            if context.get("model_tier"):
                parts.append(f"### Model Tier\n{context['model_tier']}")
```

- [ ] **Step 5: Run all CLI and context builder tests**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py tests/test_agent_sdk_context_builder.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add core/agent_sdk/cli_integration.py core/agent_sdk/context_builder.py tests/test_agent_sdk_cli_integration.py
git commit -m "feat: add resume/status/cost CLI handlers and workflow prompt enhancements"
```

---

## Task 10: Full Integration Test

**Files:**
- Modify: `tests/integration/test_agent_sdk_integration.py`

- [ ] **Step 1: Write the integration tests**

```python
# Append to tests/integration/test_agent_sdk_integration.py

class TestProductionLoopIntegration(unittest.TestCase):
    """Test the full production loop assembly."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_feedback_chain(self):
        """Goal → router → workflow → session → feedback."""
        from core.agent_sdk.config import AgentSDKConfig
        from core.agent_sdk.model_router import AdaptiveModelRouter
        from core.agent_sdk.workflow_templates import WorkflowExecutor, get_builtin_templates
        from core.agent_sdk.session_persistence import SessionStore
        from core.agent_sdk.feedback import SkillWeightUpdater, FeedbackCollector

        config = AgentSDKConfig(
            model_stats_path=Path(self.tmpdir) / "stats.json",
            session_db_path=Path(self.tmpdir) / "sessions.db",
            skill_weights_path=Path(self.tmpdir) / "weights.json",
        )

        router = AdaptiveModelRouter(stats_path=config.model_stats_path)
        session_store = SessionStore(db_path=config.session_db_path)
        skill_updater = SkillWeightUpdater(weights_path=config.skill_weights_path)
        feedback = FeedbackCollector(
            model_router=router, skill_updater=skill_updater,
            brain=None, session_store=session_store,
        )

        # 1. Router selects default model
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

        # 2. Create session
        pk = session_store.create_session("test-1", "Fix bug", "bug_fix", "bug_fix", model)
        self.assertGreater(pk, 0)

        # 3. Record a cycle event
        session_store.record_event(pk, "analyze_goal", "analyze_goal", model, 500, 200, True, None)

        # 4. Trigger feedback
        result = feedback.on_goal_complete(
            session_pk=pk, goal="Fix bug", goal_type="bug_fix",
            model=model, skills_used=["linter", "type_checker"],
            success=True, verification_result={"passed": True}, cost=0.05,
        )
        self.assertTrue(result["model_updated"])

        # 5. Router should now have stats
        stats = router.get_stats()
        self.assertIn("bug_fix", stats)

        # 6. Skill weights should be updated
        weights = skill_updater.get_weights()
        self.assertIn("linter", weights)

    def test_workflow_executor_with_mock_handlers(self):
        """Execute a workflow with mock tool handlers."""
        from core.agent_sdk.workflow_templates import (
            WorkflowExecutor, WorkflowTemplate, WorkflowPhase,
        )

        success_handler = lambda args: {"result": "ok"}
        handlers = {
            "analyze_goal": success_handler,
            "create_plan": success_handler,
            "generate_code": success_handler,
            "verify_changes": success_handler,
        }
        wf = WorkflowTemplate(
            name="test", goal_types=["test"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="generate_code"),
                WorkflowPhase(tool_name="verify_changes"),
            ],
        )
        executor = WorkflowExecutor(templates={"test": wf}, tool_handlers=handlers)
        result = executor.execute(wf, goal="Test", context={})
        self.assertTrue(result.success)
        self.assertEqual(result.phases_completed, 4)
```

- [ ] **Step 2: Run all integration tests**

Run: `python3 -m pytest tests/integration/test_agent_sdk_integration.py -v`
Expected: All 6 tests PASS (4 existing + 2 new)

- [ ] **Step 3: Run the complete test suite**

Run: `python3 -m pytest tests/test_agent_sdk_*.py tests/integration/test_agent_sdk_integration.py -v`
Expected: All tests PASS (~65 total)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_agent_sdk_integration.py
git commit -m "test: add production loop integration tests"
```
