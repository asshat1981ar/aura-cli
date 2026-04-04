"""Unit tests for agents/hierarchical_coordinator.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.swarm_models import (
    AgentRole,
    CycleLesson,
    SupervisorConfig,
    SwarmTask,
    TaskResult,
    TaskState,
)
from agents.hierarchical_coordinator import HierarchicalCoordinatorAgent


def _make_config(**kwargs) -> SupervisorConfig:
    defaults = dict(
        learning_interval=1,
        retros_dir="/tmp/test_retros",
        github_delivery_enabled=False,
    )
    defaults.update(kwargs)
    return SupervisorConfig(**defaults)


def _make_lesson_store(lessons=None) -> MagicMock:
    store = MagicMock()
    store.injectable_lessons = AsyncMock(return_value=lessons or [])
    store.record_cycle = AsyncMock(return_value=None)
    return store


def _make_passing_worker() -> MagicMock:
    worker = MagicMock()
    worker.execute = AsyncMock(
        return_value=TaskResult(
            task_id="t",
            role=AgentRole.TESTER,
            state=TaskState.COMPLETE,
            summary="all good",
            tests_passed=True,
        )
    )
    return worker


def _make_failing_worker() -> MagicMock:
    worker = MagicMock()
    worker.execute = AsyncMock(
        return_value=TaskResult(
            task_id="t",
            role=AgentRole.TESTER,
            state=TaskState.FAILED,
            summary="tests failed",
            tests_passed=False,
        )
    )
    return worker


class TestHierarchicalCoordinatorPlan:
    @patch("agents.hierarchical_coordinator.log_json")
    def test_plan_returns_three_tasks(self, _):
        config = _make_config()
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=_make_lesson_store())
        tasks = asyncio.run(
            coordinator.plan("story-1", "Build a feature")
        )
        assert len(tasks) == 3

    @patch("agents.hierarchical_coordinator.log_json")
    def test_plan_roles_are_architect_coder_tester(self, _):
        config = _make_config()
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=_make_lesson_store())
        tasks = asyncio.run(
            coordinator.plan("story-1", "some story")
        )
        roles = [t.role for t in tasks]
        assert AgentRole.ARCHITECT in roles
        assert AgentRole.CODER in roles
        assert AgentRole.TESTER in roles

    @patch("agents.hierarchical_coordinator.log_json")
    def test_plan_dependency_chain(self, _):
        config = _make_config()
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=_make_lesson_store())
        tasks = asyncio.run(
            coordinator.plan("story-1", "some story")
        )
        architect_id = tasks[0].task_id
        coder_id = tasks[1].task_id
        assert architect_id in tasks[1].depends_on
        assert coder_id in tasks[2].depends_on


class TestInvokeWorker:
    @patch("agents.hierarchical_coordinator.log_json")
    def test_missing_worker_returns_failed_result(self, _):
        config = _make_config()
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=_make_lesson_store())
        task = SwarmTask(
            task_id="t1", title="t", role=AgentRole.CODER,
            story_id="s1", description="desc"
        )
        result = asyncio.run(
            coordinator._invoke_worker(task=task, context={}, workers={})
        )
        assert result.state == TaskState.FAILED
        assert "worker_missing" in result.error_message

    @patch("agents.hierarchical_coordinator.log_json")
    def test_worker_without_execute_or_run_returns_failed(self, _):
        config = _make_config()
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=_make_lesson_store())
        task = SwarmTask(
            task_id="t1", title="t", role=AgentRole.CODER,
            story_id="s1", description="desc"
        )
        bad_worker = MagicMock(spec=[])  # no execute, no run
        result = asyncio.run(
            coordinator._invoke_worker(task=task, context={}, workers={AgentRole.CODER: bad_worker})
        )
        assert result.state == TaskState.FAILED
        assert "worker_contract_invalid" in result.error_message


class TestLessonsForCycle:
    @patch("agents.hierarchical_coordinator.log_json")
    def test_lessons_returned_on_interval(self, _):
        store = _make_lesson_store(
            lessons=[CycleLesson(cycle_number=5, lesson="Learn X", source_task_id="t1")]
        )
        config = _make_config(learning_interval=1)
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)
        lessons = asyncio.run(
            coordinator._lessons_for_cycle(1)
        )
        assert len(lessons) == 1
        store.injectable_lessons.assert_awaited_once()

    @patch("agents.hierarchical_coordinator.log_json")
    def test_no_lessons_between_intervals(self, _):
        store = _make_lesson_store()
        config = _make_config(learning_interval=5)
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)
        lessons = asyncio.run(
            coordinator._lessons_for_cycle(3)
        )
        assert lessons == []
        store.injectable_lessons.assert_not_awaited()


class TestReflect:
    @patch("agents.hierarchical_coordinator.log_json")
    def test_reflect_with_debug_report_returns_lessons(self, _):
        from core.swarm_models import CycleReport, DebugReport, SDLCFinding, SDLCLens
        config = _make_config()
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=_make_lesson_store())
        debug_report = DebugReport(
            task_id="t1",
            failure_summary="fail",
            recovery_plan=["Fix A", "Fix B", "Fix C"],
        )
        report = CycleReport(
            cycle_number=1,
            story_id="s1",
            debug_report=debug_report,
        )
        lessons = asyncio.run(coordinator.reflect(report))
        assert len(lessons) <= 3
        assert all(l.lesson in ["Fix A", "Fix B", "Fix C"] for l in lessons)

    @patch("agents.hierarchical_coordinator.log_json")
    def test_reflect_without_debug_report_returns_injected_lessons(self, _):
        from core.swarm_models import CycleReport
        config = _make_config()
        injected = [CycleLesson(cycle_number=1, lesson="Lesson 1", source_task_id="t1")]
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=_make_lesson_store())
        report = CycleReport(cycle_number=1, story_id="s1", lessons_injected=injected)
        lessons = asyncio.run(coordinator.reflect(report))
        assert lessons == injected
