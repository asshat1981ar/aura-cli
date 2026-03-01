import io
import json
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from aura_cli.commands import _handle_status
from core.task_manager import Task


def test_handle_status_renders_task_hierarchy_without_crashing():
    goal_queue = SimpleNamespace(queue=["Investigate crash"])
    goal_archive = SimpleNamespace(completed=[("Fix regression", 8.5)])
    loop = SimpleNamespace(
        current_score=7.25,
        regression_count=0,
        stable_convergence_count=2,
        current_goal="Investigate crash",
    )
    task_manager = SimpleNamespace(
        root_tasks=[
            Task(
                id="root-1",
                title="Investigate crash",
                status="in_progress",
                subtasks=[Task(id="child-1", title="Trace stack", status="completed")],
            )
        ]
    )

    out = io.StringIO()
    capability_report = {
        "last_goal": "Investigate crash",
        "matched_capability_ids": ["docker_analysis"],
        "pending_self_development_goals": ["Add AURA skill 'dockerfile_analyzer' so AURA can better handle goal: Investigate crash"],
        "pending_bootstrap_actions": ["ensure_mcp_servers"],
        "running_bootstrap_actions": ["start_skills_mcp_server"],
    }
    with patch("aura_cli.commands.TaskManager", return_value=task_manager), \
         patch("aura_cli.commands.build_capability_status_report", return_value=capability_report), \
         redirect_stdout(out):
        _handle_status(goal_queue, goal_archive, loop, as_json=False, project_root=".")

    rendered = out.getvalue()
    assert "--- Capability Bootstrap ---" in rendered
    assert "Matched capability rules: docker_analysis" in rendered
    assert "MCP bootstrap running: start_skills_mcp_server" in rendered
    assert "--- Task Hierarchy ---" in rendered
    assert "- [in progress] Investigate crash" in rendered
    assert "  - [completed] Trace stack" in rendered
    assert "Current Loop Score: 7.25" in rendered


def test_handle_status_json_emits_clean_json_on_stdout():
    goal_queue = SimpleNamespace(queue=["Investigate crash"])
    goal_archive = SimpleNamespace(completed=[("Fix regression", 8.5)])
    loop = SimpleNamespace()
    capability_report = {
        "last_goal": None,
        "matched_capability_ids": [],
        "pending_self_development_goals": [],
        "pending_bootstrap_actions": [],
        "running_bootstrap_actions": [],
    }

    out = io.StringIO()
    err = io.StringIO()
    with patch("aura_cli.commands.build_capability_status_report", return_value=capability_report), \
         redirect_stdout(out), redirect_stderr(err):
        _handle_status(goal_queue, goal_archive, loop, as_json=True, project_root=".")

    assert json.loads(out.getvalue()) == {
        "queue_length": 1,
        "queue": ["Investigate crash"],
        "completed_count": 1,
        "completed": [{"goal": "Fix regression", "score": 8.5}],
        "capabilities": capability_report,
    }
    assert "aura_status_requested" in err.getvalue()
