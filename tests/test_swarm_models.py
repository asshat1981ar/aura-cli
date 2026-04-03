"""Tests for swarm Pydantic models in core/swarm_models.py."""

from __future__ import annotations

from core.swarm_models import (
    CycleLesson,
    CycleReport,
    DebugReport,
    PRGateDecision,
    SupervisorConfig,
    SwarmTask,
    SwarmTopology,
    TaskResult,
    TaskState,
    AgentRole,
)


def test_swarm_task_default_state_is_pending() -> None:
    task = SwarmTask(
        task_id="t1",
        title="Sample task",
        role=AgentRole.CODER,
        story_id="s1",
        description="Do something",
    )
    assert task.state == TaskState.PENDING


def test_swarm_task_depends_on_default_empty() -> None:
    task = SwarmTask(
        task_id="t2",
        title="Sample task",
        role=AgentRole.ARCHITECT,
        story_id="s1",
        description="Design something",
    )
    assert task.depends_on == []


def test_task_result_default_tests_passed_false() -> None:
    result = TaskResult(
        task_id="r1",
        role=AgentRole.TESTER,
        state=TaskState.FAILED,
        summary="No tests ran",
    )
    assert result.tests_passed is False


def test_cycle_lesson_default_confidence() -> None:
    lesson = CycleLesson(
        cycle_number=1,
        lesson="Always write tests first.",
        source_task_id="t-source",
    )
    assert lesson.confidence == 0.8


def test_supervisor_config_defaults() -> None:
    cfg = SupervisorConfig()
    assert cfg.topology == SwarmTopology.HIERARCHICAL
    assert cfg.learning_interval == 5
    assert cfg.max_parallel_tasks == 3


def test_pr_gate_decision_github_port_default() -> None:
    decision = PRGateDecision(should_open_pr=True, reason="All tests passed")
    assert decision.github_server_port == 8007


def test_cycle_report_empty_collections() -> None:
    report = CycleReport(cycle_number=1, story_id="s-empty")
    assert report.tasks == []
    assert report.results == []
    assert report.lessons_injected == []


def test_debug_report_should_retry_default() -> None:
    report = DebugReport(task_id="t-debug", failure_summary="Something broke")
    assert report.should_retry is True
