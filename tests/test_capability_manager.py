from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.capability_manager import (
    analyze_capability_needs,
    build_capability_status_report,
    build_missing_skill_goals,
    capability_doctor_check,
    provision_capability_actions,
    queue_missing_capability_goals,
    record_capability_status,
)


def test_analyze_capability_needs_adds_goal_specific_skills():
    plan = analyze_capability_needs(
        "Improve Docker logging and database query performance",
        available_skills={
            "dockerfile_analyzer",
            "observability_checker",
            "database_query_analyzer",
            "symbol_indexer",
        },
        active_skills={"symbol_indexer"},
    )

    assert "dockerfile_analyzer" in plan["recommended_skills"]
    assert "observability_checker" in plan["recommended_skills"]
    assert "database_query_analyzer" in plan["recommended_skills"]
    assert plan["provisioning_actions"] == []


def test_analyze_capability_needs_identifies_mcp_setup_for_github_goals():
    plan = analyze_capability_needs(
        "Investigate GitHub pull request history for this repository",
        available_skills={"git_history_analyzer", "changelog_generator"},
        active_skills=(),
    )

    assert plan["mcp_tools"] == ["gh", "git", "fs"]
    assert plan["provisioning_actions"][0]["action"] == "ensure_mcp_servers"
    assert "git_history_analyzer" in plan["recommended_skills"]


def test_analyze_capability_needs_adds_skills_server_and_github_bridge_actions():
    plan = analyze_capability_needs(
        "Stand up the skills MCP server and the GitHub MCP bridge for copilot flows",
        available_skills={"skill_composer", "git_history_analyzer"},
        active_skills=(),
    )

    action_names = {item["action"] for item in plan["provisioning_actions"]}
    assert "start_skills_mcp_server" in action_names
    assert "start_github_mcp_bridge" in action_names
    assert "skill_composer" in plan["recommended_skills"]


def test_build_missing_skill_goals_uses_goal_text():
    goals = build_missing_skill_goals(["dockerfile_analyzer"], "Harden Docker workflows")

    assert goals == [
        "Add AURA skill 'dockerfile_analyzer' so AURA can better handle goal: Harden Docker workflows"
    ]


def test_queue_missing_capability_goals_prioritizes_and_dedupes():
    queue = MagicMock()
    queue.queue = [
        "Add AURA skill 'dockerfile_analyzer' so AURA can better handle goal: Harden Docker workflows"
    ]

    result = queue_missing_capability_goals(
        goal_queue=queue,
        missing_skills=["dockerfile_analyzer", "observability_checker"],
        goal="Harden Docker workflows",
        enabled=True,
        dry_run=False,
    )

    queue.prepend_batch.assert_called_once_with([
        "Add AURA skill 'observability_checker' so AURA can better handle goal: Harden Docker workflows"
    ])
    assert result["queued"] == [
        "Add AURA skill 'observability_checker' so AURA can better handle goal: Harden Docker workflows"
    ]
    assert result["queue_strategy"] == "prepend"
    assert result["skipped"] == [
        {
            "goal": "Add AURA skill 'dockerfile_analyzer' so AURA can better handle goal: Harden Docker workflows",
            "reason": "already_queued",
        }
    ]


def test_queue_missing_capability_goals_skips_in_dry_run():
    result = queue_missing_capability_goals(
        goal_queue=MagicMock(),
        missing_skills=["dockerfile_analyzer"],
        goal="Harden Docker workflows",
        enabled=True,
        dry_run=True,
    )

    assert result["attempted"] is False
    assert result["queued"] == []
    assert result["skipped"][0]["reason"] == "dry_run"


def test_recorded_capability_status_reports_pending_and_running_actions(tmp_path: Path):
    queue = MagicMock()
    queue.queue = [
        "Add AURA skill 'dockerfile_analyzer' so AURA can better handle goal: Harden Docker workflows"
    ]

    stored = record_capability_status(
        project_root=tmp_path,
        goal="Harden Docker workflows",
        capability_plan={
            "matched_capabilities": [{"capability_id": "docker_analysis", "reason": "containers"}],
            "recommended_skills": [],
            "missing_skills": ["dockerfile_analyzer"],
            "provisioning_actions": [{"action": "start_skills_mcp_server"}],
        },
        capability_goal_queue={
            "queued": list(queue.queue),
            "skipped": [],
            "queue_strategy": "prepend",
        },
        capability_provisioning={
            "results": [
                {"action": "ensure_mcp_servers", "status": "planned"},
                {"action": "start_skills_mcp_server", "status": "already_running"},
            ]
        },
        goal_queue=queue,
    )

    report = build_capability_status_report(tmp_path, goal_queue=queue)
    doctor_status, doctor_detail = capability_doctor_check(tmp_path, goal_queue=queue)

    assert stored["last_goal"] == "Harden Docker workflows"
    assert report["matched_capability_ids"] == ["docker_analysis"]
    assert report["pending_self_development_goals"] == list(queue.queue)
    assert report["pending_bootstrap_actions"] == ["ensure_mcp_servers"]
    assert report["running_bootstrap_actions"] == ["start_skills_mcp_server"]
    assert report["queue_strategy"] == "prepend"
    assert doctor_status == "PASS"
    assert "matched: docker_analysis" in doctor_detail
    assert "bootstrap pending: ensure_mcp_servers" in doctor_detail


def test_provision_capability_actions_skips_when_auto_provision_disabled(tmp_path: Path):
    result = provision_capability_actions(
        project_root=tmp_path,
        provisioning_actions=[{"action": "ensure_mcp_servers", "capability_id": "github_mcp"}],
        auto_provision=False,
        start_servers=False,
        dry_run=False,
    )

    assert result["attempted"] is False
    assert result["results"][0]["status"] == "planned"
    assert result["results"][0]["skipped_reason"] == "auto_provision_disabled"


def test_provision_capability_actions_runs_mcp_setup_script_without_starting_servers(tmp_path: Path):
    script_path = tmp_path / "scripts" / "mcp_server_setup.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    completed = MagicMock(returncode=0, stdout="configured", stderr="")
    with patch("core.capability_manager.subprocess.run", return_value=completed) as mock_run:
        result = provision_capability_actions(
            project_root=tmp_path,
            provisioning_actions=[{"action": "ensure_mcp_servers", "capability_id": "github_mcp"}],
            auto_provision=True,
            start_servers=False,
            dry_run=False,
        )

    assert result["attempted"] is True
    assert result["results"][0]["status"] == "applied"
    assert mock_run.call_args.args[0] == ["bash", str(script_path)]
    assert mock_run.call_args.kwargs["env"]["MCP_SETUP_NO_START"] == "1"


def test_provision_capability_actions_plans_server_start_when_auto_start_disabled(tmp_path: Path):
    result = provision_capability_actions(
        project_root=tmp_path,
        provisioning_actions=[{"action": "start_skills_mcp_server", "capability_id": "skills_mcp_server"}],
        auto_provision=True,
        start_servers=False,
        dry_run=False,
    )

    assert result["attempted"] is True
    assert result["results"][0]["status"] == "planned"
    assert result["results"][0]["skipped_reason"] == "auto_start_disabled"


def test_provision_capability_actions_starts_skills_server_when_enabled(tmp_path: Path):
    script_path = tmp_path / "scripts" / "run_mcp_server.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    fake_proc = MagicMock(pid=4321)
    with patch("core.capability_manager._listening", return_value=False), \
         patch("core.capability_manager.subprocess.Popen", return_value=fake_proc) as mock_popen:
        result = provision_capability_actions(
            project_root=tmp_path,
            provisioning_actions=[{"action": "start_skills_mcp_server", "capability_id": "skills_mcp_server"}],
            auto_provision=True,
            start_servers=True,
            dry_run=False,
        )

    assert result["results"][0]["status"] == "started"
    assert result["results"][0]["pid"] == 4321
    assert mock_popen.call_args.args[0] == ["bash", str(script_path)]


def test_provision_capability_actions_starts_github_bridge_when_enabled(tmp_path: Path):
    script_path = tmp_path / "scripts" / "start_mcp_github.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    fake_proc = MagicMock(pid=9876)
    with patch("core.capability_manager._listening", return_value=False), \
         patch("core.capability_manager.subprocess.Popen", return_value=fake_proc) as mock_popen:
        result = provision_capability_actions(
            project_root=tmp_path,
            provisioning_actions=[{"action": "start_github_mcp_bridge", "capability_id": "github_mcp_bridge"}],
            auto_provision=True,
            start_servers=True,
            dry_run=False,
        )

    assert result["results"][0]["status"] == "started"
    assert result["results"][0]["pid"] == 9876
    assert mock_popen.call_args.args[0] == ["bash", str(script_path)]


def test_provision_capability_actions_marks_dry_run_as_planned(tmp_path: Path):
    result = provision_capability_actions(
        project_root=tmp_path,
        provisioning_actions=[{"action": "ensure_mcp_servers", "capability_id": "github_mcp"}],
        auto_provision=True,
        start_servers=True,
        dry_run=True,
    )

    assert result["attempted"] is False
    assert result["results"][0]["status"] == "planned"
    assert result["results"][0]["skipped_reason"] == "dry_run"
