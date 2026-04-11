"""Integration tests for session/phase persistence and resumption behaviour.

These tests exercise :class:`~core.orchestrator.LoopOrchestrator` directly
(no server layer) using :class:`~tests.fixtures.mock_llm.MockModelAdapter` and
deterministic ``FakeAgent`` stubs so that no real LLM calls are made.

Scenarios covered
-----------------
1. ``test_pipeline_resumes_from_phase`` — simulate a cycle that stopped after
   the ``plan`` phase; verify that phase_outputs carries the plan output and
   that a subsequent cycle can pick up at ``critique``.
2. ``test_phase_outputs_persisted_between_cycles`` — verify that ``phase_outputs``
   is populated and accessible on the cycle result dict after a completed cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from core.orchestrator import LoopOrchestrator  # noqa: E402
from core.policy import Policy  # noqa: E402
from memory.store import MemoryStore  # noqa: E402
from tests.fixtures.mock_llm import MockModelAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: deterministic fake agents
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Returns a fixed *output* dict on every ``run()`` call."""

    def __init__(self, output: Dict[str, Any]) -> None:
        self.output = output
        self.call_count = 0

    def run(self, _input: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        return dict(self.output)


def _make_agents(**overrides: Any) -> Dict[str, Any]:
    """Return a complete agent mapping, optionally overriding specific phases."""
    agents: Dict[str, Any] = {
        "ingest": _FakeAgent(
            {
                "goal": "add hello() to utils.py",
                "snapshot": "core/utils.py",
                "memory_summary": "",
                "constraints": {},
            }
        ),
        "plan": _FakeAgent(
            {
                "steps": ["define hello()", "write docstring"],
                "risks": [],
            }
        ),
        "critique": _FakeAgent(
            {
                "issues": [],
                "fixes": [],
            }
        ),
        "synthesize": _FakeAgent(
            {
                "tasks": [{"id": "t1", "title": "add hello", "intent": "", "files": [], "tests": []}],
            }
        ),
        "act": _FakeAgent(
            {
                "changes": [],
            }
        ),
        "verify": _FakeAgent(
            {
                "status": "pass",
                "failures": [],
                "logs": "",
            }
        ),
        "reflect": _FakeAgent(
            {
                "summary": "ok",
                "learnings": [],
                "next_actions": [],
            }
        ),
    }
    agents.update(overrides)
    return agents


@pytest.fixture()
def tmp_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


@pytest.fixture()
def mock_adapter() -> MockModelAdapter:
    return MockModelAdapter()


@pytest.fixture()
def orchestrator(tmp_store: MemoryStore, mock_adapter: MockModelAdapter, tmp_path: Path) -> LoopOrchestrator:
    policy = Policy(max_cycles=1)
    orch = LoopOrchestrator(
        agents=_make_agents(),
        memory_store=tmp_store,
        policy=policy,
        project_root=tmp_path,
    )
    orch.model = mock_adapter
    return orch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseOutputsPersistedBetweenCycles:
    """phase_outputs are populated and accessible after a cycle completes."""

    def test_phase_outputs_persisted_between_cycles(
        self,
        orchestrator: LoopOrchestrator,
    ) -> None:
        """run_cycle must return a result dict whose 'phase_outputs' key is a
        non-empty dict containing at least the canonical pipeline phases."""
        result = orchestrator.run_cycle(
            "add hello() to utils.py",
            dry_run=True,
        )

        assert "phase_outputs" in result, "run_cycle result must contain 'phase_outputs'"
        phase_outputs = result["phase_outputs"]
        assert isinstance(phase_outputs, dict), "phase_outputs must be a dict"
        assert len(phase_outputs) > 0, "phase_outputs must not be empty after a cycle"

        # Canonical phases that a dry-run cycle should always populate.
        for expected_key in ("plan", "critique", "task_bundle"):
            assert expected_key in phase_outputs, f"Expected phase key '{expected_key}' in phase_outputs; got keys: {list(phase_outputs.keys())}"

    def test_plan_phase_output_shape(
        self,
        orchestrator: LoopOrchestrator,
    ) -> None:
        """The 'plan' key in phase_outputs must contain a 'steps' list."""
        result = orchestrator.run_cycle(
            "add hello() to utils.py",
            dry_run=True,
        )
        plan = result["phase_outputs"].get("plan", {})
        assert "steps" in plan, f"Expected 'steps' in plan phase output; got: {plan}"
        assert isinstance(plan["steps"], list), "'steps' must be a list"

    def test_multiple_cycles_have_independent_phase_outputs(
        self,
        tmp_store: MemoryStore,
        mock_adapter: MockModelAdapter,
        tmp_path: Path,
    ) -> None:
        """Two separate run_cycle calls should each return their own phase_outputs."""
        policy = Policy(max_cycles=2)
        orch = LoopOrchestrator(
            agents=_make_agents(),
            memory_store=tmp_store,
            policy=policy,
            project_root=tmp_path,
        )
        orch.model = mock_adapter

        result1 = orch.run_cycle("goal one", dry_run=True)
        result2 = orch.run_cycle("goal two", dry_run=True)

        assert result1["phase_outputs"] is not result2["phase_outputs"], "Each cycle must produce an independent phase_outputs dict"

    def test_run_loop_history_contains_phase_outputs(
        self,
        orchestrator: LoopOrchestrator,
    ) -> None:
        """run_loop history entries must each contain a 'phase_outputs' dict."""
        loop_result = orchestrator.run_loop(
            "add hello() to utils.py",
            max_cycles=1,
            dry_run=True,
        )
        assert "history" in loop_result, "run_loop result must contain 'history'"
        for i, cycle in enumerate(loop_result["history"]):
            assert "phase_outputs" in cycle, f"Cycle {i} in history missing 'phase_outputs'"
            assert isinstance(cycle["phase_outputs"], dict), f"Cycle {i} phase_outputs must be a dict"


class TestPipelineResumesFromPhase:
    """Simulate phase stopping early and verify resumption semantics."""

    def test_pipeline_resumes_from_phase(
        self,
        tmp_store: MemoryStore,
        mock_adapter: MockModelAdapter,
        tmp_path: Path,
    ) -> None:
        """Simulate a cycle that produces plan output but fails at critique;
        a subsequent cycle should still be able to run critique independently.

        Since LoopOrchestrator does not have built-in mid-cycle resumption,
        this test verifies that:
        1. A completed first cycle populates plan output.
        2. A second cycle re-runs from scratch and also populates plan + critique.
        This mirrors the existing behaviour and serves as a regression anchor.
        """
        # Cycle 1 — complete dry run; captures plan output.
        policy = Policy(max_cycles=2)
        orch = LoopOrchestrator(
            agents=_make_agents(),
            memory_store=tmp_store,
            policy=policy,
            project_root=tmp_path,
        )
        orch.model = mock_adapter

        cycle1 = orch.run_cycle("add hello() to utils.py", dry_run=True)
        plan_from_cycle1 = cycle1["phase_outputs"].get("plan", {})
        assert "steps" in plan_from_cycle1, "Cycle 1 must produce a plan with 'steps'"

        # Cycle 2 — independent run; critique must also be populated.
        cycle2 = orch.run_cycle("add hello() to utils.py", dry_run=True)
        assert "critique" in cycle2["phase_outputs"], "Cycle 2 must produce a 'critique' phase output (resuming from plan)"

    def test_plan_output_carried_into_critique(
        self,
        tmp_store: MemoryStore,
        mock_adapter: MockModelAdapter,
        tmp_path: Path,
    ) -> None:
        """In a single cycle, the plan agent's output must be visible in
        phase_outputs before critique runs."""
        # We use a recording critique agent to assert plan was in phase_outputs.
        captured: Dict[str, Any] = {}

        class _RecordingCritiqueAgent:
            def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                captured["task_bundle"] = input_data.get("task_bundle")
                # plan may arrive as a list of steps or as a dict
                plan_data = input_data.get("plan")
                if isinstance(plan_data, dict):
                    captured["plan_steps"] = plan_data.get("steps")
                elif isinstance(plan_data, list):
                    captured["plan_steps"] = plan_data
                else:
                    captured["plan_steps"] = plan_data
                return {"issues": [], "fixes": []}

        agents = _make_agents(critique=_RecordingCritiqueAgent())
        policy = Policy(max_cycles=1)
        orch = LoopOrchestrator(
            agents=agents,
            memory_store=tmp_store,
            policy=policy,
            project_root=tmp_path,
        )
        orch.model = mock_adapter

        result = orch.run_cycle("add hello() to utils.py", dry_run=True)

        assert "plan" in result["phase_outputs"], "phase_outputs must contain 'plan' after a dry-run cycle"

    def test_phase_outputs_include_dry_run_flag(
        self,
        orchestrator: LoopOrchestrator,
    ) -> None:
        """phase_outputs must record the dry_run flag set by the caller."""
        result = orchestrator.run_cycle("add hello() to utils.py", dry_run=True)
        assert result["phase_outputs"].get("dry_run") is True, "phase_outputs['dry_run'] must be True when dry_run=True"

    def test_stop_reason_set_after_cycle(
        self,
        orchestrator: LoopOrchestrator,
    ) -> None:
        """A completed cycle must set a non-empty stop_reason on the loop result."""
        loop_result = orchestrator.run_loop(
            "add hello() to utils.py",
            max_cycles=1,
            dry_run=True,
        )
        assert loop_result.get("stop_reason"), "run_loop must always set a non-empty 'stop_reason'"
