"""Unit tests for agents/sdlc_debugger.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from core.swarm_models import (
    AgentRole,
    DebugReport,
    SDLCLens,
    SwarmTask,
    TaskResult,
    TaskState,
)
from agents.sdlc_debugger import SDLCDebuggerAgent


def _make_task(task_id: str = "task-1") -> SwarmTask:
    return SwarmTask(
        task_id=task_id,
        title="Test task",
        role=AgentRole.CODER,
        story_id="story-1",
        description="do something",
    )


def _make_result(summary: str, error: str = "") -> TaskResult:
    return TaskResult(
        task_id="task-1",
        role=AgentRole.CODER,
        state=TaskState.FAILED,
        summary=summary,
        error_message=error or None,
    )


class TestSDLCDebuggerAgentInit:
    def test_name(self):
        agent = SDLCDebuggerAgent()
        assert agent.name == "sdlc_debugger"


class TestPlan:
    def test_plan_returns_four_steps(self):
        agent = SDLCDebuggerAgent()
        task = _make_task()
        result = _make_result("fail")
        plan = asyncio.get_event_loop().run_until_complete(agent.plan(task, result))
        assert len(plan) == 4
        assert all(isinstance(s, str) for s in plan)


class TestExecute:
    def setup_method(self):
        self.agent = SDLCDebuggerAgent()

    def _run(self, summary: str, error: str = "") -> DebugReport:
        task = _make_task()
        result = _make_result(summary, error)
        return asyncio.get_event_loop().run_until_complete(self.agent.execute(task, result))

    @patch("agents.sdlc_debugger.log_json")
    def test_implementation_lens_on_exception(self, _):
        report = self._run("Traceback: TypeError at line 5")
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.IMPLEMENTATION in lenses

    @patch("agents.sdlc_debugger.log_json")
    def test_requirements_lens_on_acceptance_criteria(self, _):
        report = self._run("acceptance criteria not met — unclear story")
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.REQUIREMENTS in lenses

    @patch("agents.sdlc_debugger.log_json")
    def test_integration_lens_on_timeout(self, _):
        report = self._run("connection timeout to mcp port 8001")
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.INTEGRATION in lenses

    @patch("agents.sdlc_debugger.log_json")
    def test_testing_lens_on_pytest(self, _):
        report = self._run("pytest assert failed — coverage missing")
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.TESTING in lenses

    @patch("agents.sdlc_debugger.log_json")
    def test_security_lens_on_token(self, _):
        report = self._run("auth token injection failed — permission denied")
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.SECURITY in lenses

    @patch("agents.sdlc_debugger.log_json")
    def test_performance_lens_on_slow(self, _):
        report = self._run("task slow due to high memory usage")
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.PERFORMANCE in lenses

    @patch("agents.sdlc_debugger.log_json")
    def test_fallback_finding_when_no_signals(self, _):
        report = self._run("something went wrong with no details")
        assert len(report.findings) >= 1
        assert report.findings[0].lens == SDLCLens.IMPLEMENTATION

    @patch("agents.sdlc_debugger.log_json")
    def test_report_has_recovery_plan(self, _):
        report = self._run("SyntaxError: bad code")
        assert len(report.recovery_plan) >= 2
        assert report.should_retry is True

    @patch("agents.sdlc_debugger.log_json")
    def test_recovery_plan_no_duplicates(self, _):
        report = self._run("exception traceback typeerror attributeerror")
        assert len(report.recovery_plan) == len(set(report.recovery_plan))


class TestReflect:
    @patch("agents.sdlc_debugger.log_json")
    def test_reflect_returns_actions(self, _):
        agent = SDLCDebuggerAgent()
        task = _make_task()
        result = _make_result("SyntaxError: x")
        report = asyncio.get_event_loop().run_until_complete(agent.execute(task, result))
        lessons = asyncio.get_event_loop().run_until_complete(agent.reflect(report))
        assert isinstance(lessons, list)
        assert len(lessons) >= 1
        assert all(isinstance(l, str) for l in lessons)


class TestBuildRecoveryPlan:
    @patch("agents.sdlc_debugger.log_json")
    def test_plan_always_starts_with_restate_criteria(self, _):
        agent = SDLCDebuggerAgent()
        task = _make_task("my-task")
        result = _make_result("exception")
        report = asyncio.get_event_loop().run_until_complete(agent.execute(task, result))
        assert any("Re-state acceptance criteria" in step for step in report.recovery_plan)

    @patch("agents.sdlc_debugger.log_json")
    def test_plan_ends_with_rerun_tester(self, _):
        agent = SDLCDebuggerAgent()
        task = _make_task()
        result = _make_result("exception")
        report = asyncio.get_event_loop().run_until_complete(agent.execute(task, result))
        assert any("tester validation" in step for step in report.recovery_plan)
