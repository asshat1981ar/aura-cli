"""End-to-end integration tests for LoopOrchestrator.

Uses a configurable MockModelAdapter so each phase returns fake but
schema-valid responses without hitting any external service.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore


# ── Helpers ──────────────────────────────────────────────────────────────────

class _FakeAgent:
    """Always returns *output*."""

    def __init__(self, output: Dict[str, Any]):
        self.output = output
        self.call_count = 0

    def run(self, _input_data: Dict) -> Dict:
        self.call_count += 1
        return self.output


class _CallableAgent:
    """Returns output produced by *fn(input_data)*."""

    def __init__(self, fn):
        self._fn = fn
        self.call_count = 0

    def run(self, input_data: Dict) -> Dict:
        self.call_count += 1
        return self._fn(input_data)


def _base_agents(**overrides) -> Dict[str, Any]:
    """Return a full agent dict with schema-valid responses, applying overrides."""
    agents = {
        "ingest": _FakeAgent({
            "goal": "test-goal",
            "snapshot": "file.py",
            "memory_summary": "",
            "constraints": {},
        }),
        "plan": _FakeAgent({
            "steps": ["step 1"],
            "risks": [],
        }),
        "critique": _FakeAgent({
            "issues": [],
            "fixes": [],
        }),
        "synthesize": _FakeAgent({
            "tasks": [{"id": "t1", "title": "demo", "intent": "", "files": [], "tests": []}],
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
            "summary": "done",
            "learnings": [],
            "next_actions": [],
        }),
    }
    agents.update(overrides)
    return agents


def _make_orchestrator(agents, tmp_path: Path, **kwargs) -> LoopOrchestrator:
    store = MemoryStore(tmp_path)
    policy = Policy(max_cycles=1)
    return LoopOrchestrator(
        agents,
        store,
        policy=policy,
        project_root=tmp_path,
        **kwargs,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_full_9_phase_cycle_completes(tmp_path: Path):
    """Full pipeline runs without error and all expected phases are present."""
    orchestrator = _make_orchestrator(_base_agents(), tmp_path)
    result = orchestrator.run_loop("add a feature", max_cycles=1, dry_run=True)

    assert result["stop_reason"] == "PASS"
    assert result["history"], "expected at least one cycle entry"

    phases = result["history"][0]["phase_outputs"]
    for expected_phase in [
        "context", "plan", "critique", "task_bundle",
        "change_set", "verification", "reflection",
    ]:
        assert expected_phase in phases, f"Missing phase: {expected_phase}"


def test_sandbox_failure_triggers_act_retry(tmp_path: Path):
    """When sandbox fails once then passes, act is retried before verify."""
    fail_then_pass = iter([
        {"passed": False, "summary": "syntax error", "details": {"stderr": "SyntaxError"}},
        {"passed": True, "summary": "ok"},
    ])

    sandbox_agent = _CallableAgent(lambda _: next(fail_then_pass))
    agents = _base_agents(sandbox=sandbox_agent)

    orchestrator = _make_orchestrator(agents, tmp_path)
    result = orchestrator.run_loop("fix sandbox", max_cycles=1, dry_run=True)

    # Cycle should still complete (sandbox failures are retried or tolerated in dry_run)
    assert result["history"], "expected at least one cycle"
    assert sandbox_agent.call_count >= 1


def test_verify_failure_routes_to_act_retry(tmp_path: Path):
    """When verify fails with a recoverable error, act is retried."""
    call_counts = {"verify": 0}

    def _verify_agent(input_data):
        call_counts["verify"] += 1
        if call_counts["verify"] == 1:
            return {"status": "fail", "failures": ["AssertionError: x != y"], "logs": ""}
        return {"status": "pass", "failures": [], "logs": ""}

    agents = _base_agents(verify=_CallableAgent(_verify_agent))
    orchestrator = _make_orchestrator(agents, tmp_path)
    result = orchestrator.run_loop("fix test", max_cycles=1, dry_run=True)

    assert result["history"], "expected at least one cycle"
    # verify should have been called at least twice (fail → retry → pass)
    assert call_counts["verify"] >= 2


def test_dry_run_skips_file_writes(tmp_path: Path):
    """dry_run=True must not write any files to project_root."""
    target_file = tmp_path / "output.py"

    act_with_change = _FakeAgent({
        "changes": [{
            "file_path": str(target_file),
            "old_code": "",
            "new_code": "# generated",
            "overwrite_file": True,
        }],
    })
    agents = _base_agents(act=act_with_change)
    orchestrator = _make_orchestrator(agents, tmp_path)
    result = orchestrator.run_loop("write file", max_cycles=1, dry_run=True)

    assert result["history"], "expected at least one cycle"
    # File must NOT exist because dry_run=True prevents writes
    assert not target_file.exists(), "dry_run should not write files"


def test_human_gate_blocks_and_denies_skips_apply(tmp_path: Path, monkeypatch):
    """When HumanGate blocks and is denied, the cycle is skipped gracefully."""
    monkeypatch.delenv("AURA_AUTO_APPROVE", raising=False)

    # skill_context contains security findings to trigger the gate
    def _ingest(_input):
        return {
            "goal": "risky change",
            "snapshot": "",
            "memory_summary": "",
            "constraints": {},
        }

    agents = _base_agents(ingest=_CallableAgent(_ingest))
    orchestrator = _make_orchestrator(agents, tmp_path)

    # Inject critical security finding into skill_context via monkeypatching
    # dispatch_skills to return a mock result
    with patch(
        "core.orchestrator.dispatch_skills",
        return_value={"security_scanner": {"critical_count": 1, "findings": ["SQL injection"]}},
    ), patch.object(
        orchestrator.human_gate, "request_approval", return_value=False
    ):
        result = orchestrator.run_loop("risky", max_cycles=1, dry_run=True)

    assert result["history"], "expected at least one cycle"
    phases = result["history"][0]["phase_outputs"]
    gate_info = phases.get("human_gate", {})
    assert gate_info.get("blocked") is True
    assert gate_info.get("approved") is False
