"""Full 10-phase pipeline integration tests using MockModelAdapter.

These tests exercise the complete LoopOrchestrator pipeline without making any
real LLM or external-service calls.  Every agent is replaced with a deterministic
stub; the ``MockModelAdapter`` is wired in as the model so any path that falls
back to the adapter still returns structured, schema-valid JSON.

Phases exercised (in order):
    1.  Ingest        → phase_outputs["context"]
    2.  Skill dispatch → phase_outputs["skill_context"]
    3.  Plan          → phase_outputs["plan"]
    4.  Critique      → phase_outputs["critique"]
    5.  Synthesize    → phase_outputs["task_bundle"]
    6.  Act           → phase_outputs["change_set"]
    7.  Sandbox       → phase_outputs["sandbox"]  (internal, visible via retry)
    8.  Apply         → phase_outputs["apply_result"]
    9.  Verify        → phase_outputs["verification"]
    10. Reflect       → phase_outputs["reflection"]
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Iterator
from unittest.mock import MagicMock, patch

import pytest

from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore
from tests.fixtures.mock_llm import MockModelAdapter
from tests.fixtures.mock_responses import (
    CODER_RESPONSE,
    CRITIC_RESPONSE,
    PLANNER_RESPONSE,
    REFLECTOR_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers: Fake agents (deterministic, no I/O)
# ---------------------------------------------------------------------------

class _FakeAgent:
    """Returns a fixed *output* dict on every ``run()`` call."""

    def __init__(self, output: Dict[str, Any]) -> None:
        self.output = output
        self.call_count = 0

    def run(self, _input: Dict) -> Dict:
        self.call_count += 1
        return self.output


class _CallableAgent:
    """Delegates each ``run()`` call to a user-supplied callable."""

    def __init__(self, fn) -> None:
        self._fn = fn
        self.call_count = 0

    def run(self, input_data: Dict) -> Dict:
        self.call_count += 1
        return self._fn(input_data)


def _base_agents(**overrides) -> Dict[str, Any]:
    """Return a complete, schema-valid agent dict for LoopOrchestrator.

    All phases return the minimum structure required by downstream phases so
    that the pipeline can run to completion without schema errors.  Individual
    tests pass *overrides* to replace specific phases with custom behaviour.
    """
    agents: Dict[str, Any] = {
        "ingest": _FakeAgent({
            "goal": "Add a hello() function to core/utils.py",
            "snapshot": "core/utils.py",
            "memory_summary": "",
            "constraints": {},
        }),
        "plan": _FakeAgent({
            "steps": [
                {"id": "s1", "title": "Implement hello()", "description": "Add function.", "files": ["core/utils.py"]},
            ],
            "risks": [],
            "estimated_complexity": "low",
        }),
        "critique": _FakeAgent({
            "issues": [],
            "fixes": [],
            "blocking": False,
            "approved": True,
        }),
        "synthesize": _FakeAgent({
            "tasks": [{"id": "t1", "title": "Add hello()", "intent": "add function", "files": ["core/utils.py"], "tests": []}],
        }),
        "act": _FakeAgent({
            "changes": [],
        }),
        "sandbox": _FakeAgent({
            "passed": True,
            "summary": "ok",
        }),
        "verify": _FakeAgent({
            "status": "pass",
            "failures": [],
            "logs": "",
        }),
        "reflect": _FakeAgent({
            "summary": "Cycle completed successfully.",
            "learnings": ["Pattern-based mocking keeps tests hermetic."],
            "next_actions": [],
            "cycle_outcome": "success",
            "confidence": 0.95,
        }),
    }
    agents.update(overrides)
    return agents


def _make_orchestrator(
    agents: Dict[str, Any],
    tmp_path: Path,
    mock_adapter: MockModelAdapter | None = None,
    **kwargs,
) -> LoopOrchestrator:
    """Construct a LoopOrchestrator wired with mock dependencies.

    Args:
        agents:       Phase agent dict (use :func:`_base_agents` as the base).
        tmp_path:     pytest ``tmp_path`` fixture value for MemoryStore isolation.
        mock_adapter: Optional :class:`~tests.fixtures.mock_llm.MockModelAdapter`
                      instance to pass as ``model=``.  A default is created when
                      *None*.
        **kwargs:     Forwarded verbatim to :class:`LoopOrchestrator`.
    """
    if mock_adapter is None:
        mock_adapter = MockModelAdapter()

    store = MemoryStore(tmp_path)
    policy = Policy(max_cycles=1)

    brain = MagicMock(name="brain")
    # recall_with_budget is called during ingest; return empty list by default.
    brain.recall_with_budget.return_value = []
    brain.vector_store = MagicMock(name="vector_store")

    return LoopOrchestrator(
        agents,
        store,
        policy=policy,
        project_root=tmp_path,
        brain=brain,
        model=mock_adapter,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1 — Full 10-phase pipeline with MockModelAdapter
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_pipeline_completes_with_mock_adapter(tmp_path: Path) -> None:
    """10-phase pipeline runs to completion without raising and all key phase
    outputs are present in the cycle result."""
    goal = "Add a hello() function to core/utils.py"
    mock_adapter = MockModelAdapter()
    orchestrator = _make_orchestrator(_base_agents(), tmp_path, mock_adapter=mock_adapter)

    # --- run ---
    result = orchestrator.run_loop(goal, max_cycles=1, dry_run=True)

    # --- basic shape ---
    assert result["goal"] == goal
    assert result["stop_reason"] in {"PASS", "MAX_CYCLES"}, (
        f"Unexpected stop_reason: {result['stop_reason']!r}"
    )
    assert result["history"], "Expected at least one completed cycle."

    phases = result["history"][0]["phase_outputs"]

    # --- 10 phases produce their expected output keys ---
    expected_phase_keys = [
        "context",       # phase 1  – ingest
        "skill_context", # phase 2  – skill dispatch
        "plan",          # phase 3  – planner
        "critique",      # phase 4  – critic
        "task_bundle",   # phase 5  – synthesize
        "change_set",    # phase 6  – act
        "verification",  # phase 9  – verify
        "reflection",    # phase 10 – reflect
    ]
    for key in expected_phase_keys:
        assert key in phases, f"Expected phase_outputs['{key}'] to be set after pipeline run."

    # --- plan contains expected structure ---
    plan = phases["plan"]
    assert "steps" in plan, "plan output should have 'steps' key"

    # --- code (change_set) was produced ---
    change_set = phases["change_set"]
    assert "changes" in change_set, "change_set output should have 'changes' key"

    # --- verification passed ---
    verification = phases["verification"]
    assert verification.get("status") == "pass", (
        f"Expected verification to pass, got: {verification.get('status')!r}"
    )

    # --- reflection was written ---
    reflection = phases["reflection"]
    assert "summary" in reflection, "reflection output should contain 'summary'"

    # --- brain.remember was called (reflector / learn phase) ---
    brain: MagicMock = orchestrator.brain  # type: ignore[assignment]
    brain.remember.assert_called()


# ---------------------------------------------------------------------------
# Test 2 — Sandbox failure triggers act retry
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_sandbox_failure_triggers_retry(tmp_path: Path) -> None:
    """When the sandbox agent fails on the first two calls and passes on the
    third, the pipeline retries the act phase and eventually completes."""
    responses: Iterator[Dict] = iter([
        {"passed": False, "summary": "syntax error", "details": {"stderr": "SyntaxError: invalid syntax"}},
        {"passed": False, "summary": "runtime error", "details": {"stderr": "RuntimeError: boom"}},
        {"passed": True,  "summary": "all checks green"},
    ])

    sandbox_agent = _CallableAgent(lambda _: next(responses))

    # A coder that also tracks call count so we can verify it was re-invoked.
    coder_calls: list[int] = []
    original_act = _FakeAgent({"changes": []})
    act_agent = _CallableAgent(lambda inp: (coder_calls.append(1), original_act.run(inp))[1])

    agents = _base_agents(sandbox=sandbox_agent, act=act_agent)
    mock_adapter = MockModelAdapter()
    orchestrator = _make_orchestrator(agents, tmp_path, mock_adapter=mock_adapter)

    # dry_run=False is required: when dry_run=True the sandbox loop breaks on the
    # first call regardless of the result, so retries would never fire.
    # The change_set has no file changes (changes=[]) so no real writes occur.
    result = orchestrator.run_loop("fix sandbox error", max_cycles=1, dry_run=False)

    # Pipeline must produce at least one completed cycle.
    assert result["history"], "Expected at least one cycle."

    # Sandbox was called multiple times (fail → retry → pass).
    assert sandbox_agent.call_count >= 2, (
        f"Sandbox should have been retried; call_count={sandbox_agent.call_count}"
    )

    # Act should have been retried alongside the sandbox.
    assert len(coder_calls) >= 2, (
        f"Coder (act) should have been retried; call_count={len(coder_calls)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Cancel mid-run via running_runs API
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cancel_run_signals_stop_event(tmp_path: Path) -> None:
    """register_run / cancel_run round-trip: cancellation sets the stop event
    and any agent that honours it can observe the cancellation signal.

    The orchestrator itself delegates stop-event checking to agents; here we
    verify the API contract at the integration boundary — the stop event is set
    before a slow ingest agent finishes, and the agent can detect the signal.
    """
    from core.running_runs import cancel_run, deregister_run, get_stop_event, register_run

    run_id = "test-cancel-run-abc123"

    # ── Register the run and capture its stop event ──
    stop_event = register_run(run_id)

    # Confirm initial state.
    assert not stop_event.is_set(), "Stop event must not be set before cancel."
    assert get_stop_event(run_id) is stop_event

    # ── Simulate an agent that observes the stop event ──
    observed_cancellation: list[bool] = []

    ready_event = threading.Event()

    def _cancellable_ingest(_input: Dict) -> Dict:
        # Signal the main thread that we have entered the agent.
        ready_event.set()
        # Spin-wait for cancellation (bounded; should be near-instant in tests).
        stop_event.wait(timeout=2.0)
        observed_cancellation.append(stop_event.is_set())
        return {
            "goal": _input.get("goal", ""),
            "snapshot": "",
            "memory_summary": "",
            "constraints": {},
        }

    ingest_agent = _CallableAgent(_cancellable_ingest)
    agents = _base_agents(ingest=ingest_agent)
    mock_adapter = MockModelAdapter()
    orchestrator = _make_orchestrator(agents, tmp_path, mock_adapter=mock_adapter)
    orchestrator.brain.recall_with_budget.return_value = []

    # ── Run the pipeline in a background thread so we can cancel concurrently ──
    result_holder: list[Dict] = []
    exc_holder: list[BaseException] = []

    def _run() -> None:
        try:
            result_holder.append(
                orchestrator.run_cycle(
                    "Add a hello() function to core/utils.py",
                    dry_run=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            exc_holder.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Wait until the ingest agent is executing, then cancel.
    ready_event.wait(timeout=5.0)
    cancelled = cancel_run(run_id)
    assert cancelled, "cancel_run should return True for a known run_id."

    # ── Wait for the pipeline thread to finish ──
    thread.join(timeout=10.0)
    assert not thread.is_alive(), "Pipeline thread should finish within the timeout."

    # No unhandled exceptions from the orchestrator.
    if exc_holder:
        raise exc_holder[0]

    # Agent observed the cancellation signal.
    assert observed_cancellation, "Ingest agent never executed."
    assert observed_cancellation[0] is True, (
        "Ingest agent should have seen stop_event.is_set() == True after cancel_run."
    )

    # Cleanup.
    deregister_run(run_id)
    assert get_stop_event(run_id) is None, "Run should be deregistered after cleanup."
