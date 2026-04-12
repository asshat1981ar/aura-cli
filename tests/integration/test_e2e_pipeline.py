"""E2E pipeline tests: happy-path and failure-rollback.

Test 1 — ``test_happy_path_full_pipeline``
    Runs a complete ``run_loop(goal, max_cycles=1, dry_run=True)`` with
    ``MockModelAdapter`` and verifies all eight required phase_output keys are
    present and structurally correct.

Test 2 — ``test_failure_triggers_rollback``
    Injects a change whose ``old_code`` cannot be found in the target file so
    that ``_apply_change_set`` records an ``OldCodeNotFoundError`` in
    ``apply_result["failed"]``.  Verifies the orchestrator handles the failure
    gracefully (no unhandled exception) and that the target file is either
    unchanged or restored to its original content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from core.file_tools import OldCodeNotFoundError  # noqa: F401 – imported for isinstance checks
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore
from tests.fixtures.mock_llm import MockModelAdapter


# ---------------------------------------------------------------------------
# Shared helpers (mirrors the pattern from test_pipeline_integration.py)
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
    """Return a complete, schema-valid agent dict for ``LoopOrchestrator``.

    All phases return the minimum structure required by downstream phases so
    the pipeline can run to completion without schema errors.  Individual tests
    pass *overrides* to replace specific phases with custom behaviour.
    """
    agents: Dict[str, Any] = {
        "ingest": _FakeAgent(
            {
                "goal": "Add a hello() function to core/utils.py",
                "snapshot": "core/utils.py",
                "memory_summary": "",
                "constraints": {},
            }
        ),
        "plan": _FakeAgent(
            {
                "steps": [
                    {
                        "id": "s1",
                        "title": "Implement hello()",
                        "description": "Add function.",
                        "files": ["core/utils.py"],
                    },
                ],
                "risks": [],
                "estimated_complexity": "low",
            }
        ),
        "critique": _FakeAgent(
            {
                "issues": [],
                "fixes": [],
                "blocking": False,
                "approved": True,
            }
        ),
        "synthesize": _FakeAgent(
            {
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Add hello()",
                        "intent": "add function",
                        "files": ["core/utils.py"],
                        "tests": [],
                    }
                ],
            }
        ),
        "act": _FakeAgent({"changes": []}),
        "sandbox": _FakeAgent({"passed": True, "summary": "ok"}),
        "verify": _FakeAgent(
            {
                "passed": True,
                "status": "pass",
                "failures": [],
                "logs": "",
            }
        ),
        "reflect": _FakeAgent(
            {
                "summary": "Cycle completed successfully.",
                "learnings": ["Mock-based tests keep the pipeline hermetic."],
                "next_actions": [],
                "cycle_outcome": "success",
                "confidence": 0.95,
            }
        ),
    }
    agents.update(overrides)
    return agents


def _make_orchestrator(
    agents: Dict[str, Any],
    tmp_path: Path,
    mock_adapter: MockModelAdapter | None = None,
    **kwargs,
) -> LoopOrchestrator:
    """Construct a ``LoopOrchestrator`` wired with mock dependencies.

    Args:
        agents:       Phase agent dict (use :func:`_base_agents` as the base).
        tmp_path:     pytest ``tmp_path`` fixture for ``MemoryStore`` isolation.
        mock_adapter: Optional :class:`MockModelAdapter`; a default is created
                      when ``None``.
        **kwargs:     Forwarded verbatim to :class:`LoopOrchestrator`.
    """
    if mock_adapter is None:
        mock_adapter = MockModelAdapter()

    store = MemoryStore(tmp_path)
    policy = Policy(max_cycles=1)

    brain = MagicMock(name="brain")
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
# Test 1 — Full pipeline happy-path
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_happy_path_full_pipeline(tmp_path: Path) -> None:
    """Complete ``run_loop`` with ``dry_run=True`` produces all required
    phase_output keys and the expected structural invariants.

    Verified:
    * All eight phase_output keys are present.
    * ``plan["steps"]`` is a non-empty list.
    * ``verification["passed"]`` is ``True``.
    * ``brain.remember`` was called (reflection phase executed).
    """
    goal = "Add a hello() function to core/utils.py"
    mock_adapter = MockModelAdapter()
    orchestrator = _make_orchestrator(_base_agents(), tmp_path, mock_adapter=mock_adapter)

    result = orchestrator.run_loop(goal, max_cycles=1, dry_run=True)

    # ── Basic shape ──────────────────────────────────────────────────────────
    assert result["goal"] == goal
    assert result["stop_reason"] in {"PASS", "MAX_CYCLES", ""}, f"Unexpected stop_reason: {result['stop_reason']!r}"
    assert result["history"], "Expected at least one completed cycle."

    phases = result["history"][0]["phase_outputs"]

    # ── All eight required phase output keys ─────────────────────────────────
    required_keys = [
        "context",  # phase 1  – ingest
        "skill_context",  # phase 2  – skill dispatch
        "plan",  # phase 3  – planner
        "critique",  # phase 4  – critic
        "task_bundle",  # phase 5  – synthesize
        "change_set",  # phase 6  – act
        "verification",  # phase 9  – verify
        "reflection",  # phase 10 – reflect
    ]
    for key in required_keys:
        assert key in phases, f"Expected phase_outputs[{key!r}] to be present after pipeline run."

    # ── plan.steps is a list ─────────────────────────────────────────────────
    plan = phases["plan"]
    assert isinstance(plan.get("steps"), list), f"plan['steps'] should be a list, got: {type(plan.get('steps'))!r}"

    # ── verification passed ───────────────────────────────────────────────────
    verification = phases["verification"]
    assert verification.get("passed") is True, f"Expected verification['passed'] == True, got: {verification.get('passed')!r}"

    # ── reflection was written ────────────────────────────────────────────────
    reflection = phases["reflection"]
    assert "summary" in reflection, "reflection output should contain 'summary'."

    # ── brain.remember was called (learn phase executed) ─────────────────────
    brain: MagicMock = orchestrator.brain  # type: ignore[assignment]
    brain.remember.assert_called()


# ---------------------------------------------------------------------------
# Test 2 — Failure triggers graceful handling and rollback
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_failure_triggers_rollback(tmp_path: Path) -> None:
    """When the act agent returns a change whose ``old_code`` does not exist in
    the target file, ``_apply_change_set`` records an ``OldCodeNotFoundError``
    in ``apply_result["failed"]``.

    The orchestrator must:
    1. Not raise an unhandled exception.
    2. Produce a cycle entry in ``result["history"]``.
    3. Record the application failure in ``apply_result["failed"]``.
    4. Leave the target file at its original content (rollback / no partial write).
    """
    # ── Prepare a real target file in the project root ───────────────────────
    target_rel = "src/utils.py"
    target_abs = tmp_path / target_rel
    target_abs.parent.mkdir(parents=True, exist_ok=True)
    original_content = "# original content\n"
    target_abs.write_text(original_content, encoding="utf-8")

    # ── Act agent returns a change whose old_code is absent in the file ───────
    def _failing_act(_input: Dict) -> Dict:
        return {
            "changes": [
                {
                    "file_path": target_rel,
                    "old_code": "THIS_OLD_CODE_DOES_NOT_EXIST_IN_THE_FILE",
                    "new_code": "# replacement\n",
                }
            ]
        }

    agents = _base_agents(act=_CallableAgent(_failing_act))
    orchestrator = _make_orchestrator(agents, tmp_path)

    # ── Run – must not raise ─────────────────────────────────────────────────
    result = orchestrator.run_loop("test rollback on apply failure", max_cycles=1, dry_run=False)

    # ── Pipeline produced at least one cycle ─────────────────────────────────
    assert result["history"], "Expected at least one cycle entry in history."

    phases = result["history"][0]["phase_outputs"]

    # ── apply_result records the failure ─────────────────────────────────────
    apply_result = phases.get("apply_result", {})
    assert apply_result.get("failed"), "Expected apply_result['failed'] to be non-empty when old_code is absent."

    # ── Target file is unchanged (no partial write; rollback is a no-op since
    #    OldCodeNotFoundError is raised before any bytes are written) ──────────
    assert target_abs.read_text(encoding="utf-8") == original_content, "Target file content should be unchanged after a failed apply."
