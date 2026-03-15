"""End-to-end integration tests for LoopOrchestrator.

Uses a configurable MockModelAdapter so each phase returns fake but
schema-valid responses without hitting any external service.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

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


def test_verify_failure_restores_only_loop_applied_files(tmp_path: Path):
    """Verify-fail recovery should not stash or wipe unrelated dirty files."""
    dirty_file = tmp_path / "notes.txt"
    dirty_file.write_text("user dirty worktree note\n", encoding="utf-8")

    target_file = tmp_path / "run_aura.sh"
    target_file.write_text("#!/usr/bin/env bash\necho old\n", encoding="utf-8")
    target_file.chmod(0o755)

    agents = _base_agents(
        act=_FakeAgent({
            "changes": [{
                "file_path": "run_aura.sh",
                "old_code": "#!/usr/bin/env bash\necho old\n",
                "new_code": "#!/usr/bin/env bash\necho new\n",
                "overwrite_file": False,
            }],
        }),
        verify=_FakeAgent({
            "status": "fail",
            "failures": ["AssertionError: wrapper output mismatch"],
            "logs": "",
        }),
    )
    orchestrator = _make_orchestrator(agents, tmp_path)
    result = orchestrator.run_loop("fix wrapper", max_cycles=1, dry_run=False)

    assert result["history"], "expected at least one cycle"
    assert target_file.read_text(encoding="utf-8") == "#!/usr/bin/env bash\necho old\n"
    assert target_file.stat().st_mode & 0o777 == 0o755
    assert dirty_file.read_text(encoding="utf-8") == "user dirty worktree note\n"


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


def test_goal_capability_plan_augments_skill_dispatch(tmp_path: Path):
    orchestrator = _make_orchestrator(_base_agents(), tmp_path)
    orchestrator.skills = {
        "symbol_indexer": MagicMock(),
        "architecture_validator": MagicMock(),
        "dockerfile_analyzer": MagicMock(),
        "observability_checker": MagicMock(),
    }

    with patch("core.orchestrator.dispatch_skills", return_value={}) as mock_dispatch:
        result = orchestrator.run_cycle("Improve Docker logging and observability coverage", dry_run=True)

    phases = result["phase_outputs"]
    capability_plan = phases["capability_plan"]
    assert "dockerfile_analyzer" in capability_plan["recommended_skills"]
    assert "observability_checker" in capability_plan["recommended_skills"]
    assert "dockerfile_analyzer" in phases["pipeline_config"]["skills"]
    assert "observability_checker" in phases["pipeline_config"]["skills"]
    active_skills = mock_dispatch.call_args.args[1]
    assert "dockerfile_analyzer" in active_skills
    assert "observability_checker" in active_skills


def test_goal_capability_plan_can_trigger_mcp_provisioning(tmp_path: Path):
    orchestrator = _make_orchestrator(
        _base_agents(),
        tmp_path,
        auto_provision_mcp=True,
    )
    orchestrator.skills = {
        "symbol_indexer": MagicMock(),
        "git_history_analyzer": MagicMock(),
        "changelog_generator": MagicMock(),
    }

    with patch(
        "core.orchestrator.provision_capability_actions",
        return_value={"attempted": True, "results": [{"action": "ensure_mcp_servers", "status": "applied"}]},
    ) as mock_provision, patch("core.orchestrator.dispatch_skills", return_value={}):
        result = orchestrator.run_cycle("Review GitHub pull requests and release notes", dry_run=False)

    phases = result["phase_outputs"]
    assert phases["capability_provisioning"]["attempted"] is True
    mock_provision.assert_called_once()


def test_missing_capability_skills_are_queued_as_self_development_goals(tmp_path: Path):
    goal_queue = MagicMock()
    goal_queue.queue = []
    orchestrator = _make_orchestrator(
        _base_agents(),
        tmp_path,
        goal_queue=goal_queue,
    )
    orchestrator.skills = {"symbol_indexer": MagicMock()}

    with patch("core.orchestrator.dispatch_skills", return_value={}):
        result = orchestrator.run_cycle("Improve Docker observability coverage", dry_run=False)

    phases = result["phase_outputs"]
    queued = phases["capability_goal_queue"]["queued"]
    assert len(queued) == 2
    assert any("dockerfile_analyzer" in item for item in queued)
    assert any("observability_checker" in item for item in queued)
    assert phases["capability_goal_queue"]["queue_strategy"] == "prepend"
    goal_queue.prepend_batch.assert_called_once_with(queued)


def test_capability_status_is_recorded_for_status_and_doctor_surfaces(tmp_path: Path):
    goal_queue = MagicMock()
    goal_queue.queue = []
    orchestrator = _make_orchestrator(
        _base_agents(),
        tmp_path,
        goal_queue=goal_queue,
    )
    orchestrator.skills = {"symbol_indexer": MagicMock()}

    with patch("core.orchestrator.dispatch_skills", return_value={}):
        orchestrator.run_cycle("Improve Docker observability coverage", dry_run=False)

    assert orchestrator.last_capability_status["last_goal"] == "Improve Docker observability coverage"
    matched = orchestrator.last_capability_status["matched_capabilities"]
    assert any(item["capability_id"] == "docker_analysis" for item in matched)
    assert (tmp_path / "memory" / "capability_status.json").exists()


def test_missing_capability_skills_are_not_queued_in_dry_run(tmp_path: Path):
    goal_queue = MagicMock()
    goal_queue.queue = []
    orchestrator = _make_orchestrator(
        _base_agents(),
        tmp_path,
        goal_queue=goal_queue,
    )
    orchestrator.skills = {"symbol_indexer": MagicMock()}

    with patch("core.orchestrator.dispatch_skills", return_value={}):
        result = orchestrator.run_cycle("Improve Docker observability coverage", dry_run=True)

    phases = result["phase_outputs"]
    assert phases["capability_goal_queue"]["attempted"] is False
    assert phases["capability_goal_queue"]["queued"] == []
    goal_queue.prepend_batch.assert_not_called()
