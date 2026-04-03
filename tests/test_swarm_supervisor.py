"""Tests for the hierarchical supervisor workflow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agents.hierarchical_coordinator import HierarchicalCoordinatorAgent
from core.swarm_models import AgentRole, SupervisorConfig, TaskResult, TaskState, SwarmTask, CycleLesson
from core.swarm_supervisor import install_swarm_runtime
from memory.learning_loop import LessonStore


class FakeWorker:
    """Captures contexts and returns a deterministic task result."""

    def __init__(self, role: AgentRole, state: TaskState, tests_passed: bool = True, summary: str = "ok") -> None:
        self.role = role
        self.state = state
        self.tests_passed = tests_passed
        self.summary = summary
        self.contexts: list[dict] = []

    async def execute(self, task: SwarmTask, context: dict) -> TaskResult:
        self.contexts.append(context)
        return TaskResult(
            task_id=task.task_id,
            role=self.role,
            state=self.state,
            summary=self.summary,
            tests_passed=self.tests_passed,
            error_message=None if self.state == TaskState.COMPLETE else "pytest assertion failure",
            output={"role": self.role.value},
        )


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def register(self, name: str, factory: object) -> None:
        self.calls.append((name, factory))


class FakeOrchestrator:
    pass


def test_learning_is_injected_on_every_fifth_cycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LessonStore(tmp_path / "memory")
        from core.swarm_models import CycleReport, PRGateDecision
        seed = CycleReport(
            cycle_number=1,
            story_id="seed",
            lessons_injected=[CycleLesson(cycle_number=1, lesson="Prefer smaller patches.", source_task_id="seed-1")],
            pr_gate=PRGateDecision(should_open_pr=False, reason="seed"),
        )
        await store.record_cycle(seed)

        config = SupervisorConfig(learning_interval=5, retros_dir=str(tmp_path / "retros"))
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)
        coordinator._cycle_counter = 4  # next execute call becomes cycle 5

        architect = FakeWorker(AgentRole.ARCHITECT, TaskState.COMPLETE)
        coder = FakeWorker(AgentRole.CODER, TaskState.COMPLETE)
        tester = FakeWorker(AgentRole.TESTER, TaskState.COMPLETE, tests_passed=True)

        await coordinator.execute(
            story_id="story-1",
            story_text="Build feature X",
            workers={
                AgentRole.ARCHITECT: architect,
                AgentRole.CODER: coder,
                AgentRole.TESTER: tester,
            },
        )

        assert architect.contexts[0]["lessons"] == ["Prefer smaller patches."]
        assert coder.contexts[0]["lessons"] == ["Prefer smaller patches."]
        assert tester.contexts[0]["lessons"] == ["Prefer smaller patches."]

    asyncio.run(scenario())


def test_tester_failure_creates_debug_report_and_retrospective(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LessonStore(tmp_path / "memory")
        config = SupervisorConfig(retros_dir=str(tmp_path / "retros"))
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)

        report = await coordinator.execute(
            story_id="story-2",
            story_text="Implement feature Y",
            workers={
                AgentRole.ARCHITECT: FakeWorker(AgentRole.ARCHITECT, TaskState.COMPLETE),
                AgentRole.CODER: FakeWorker(AgentRole.CODER, TaskState.COMPLETE),
                AgentRole.TESTER: FakeWorker(
                    AgentRole.TESTER,
                    TaskState.FAILED,
                    tests_passed=False,
                    summary="pytest assertion failure in integration test",
                ),
            },
        )

        retro_file = tmp_path / "retros" / "story-2-cycle-1.json"
        assert report.debug_report is not None
        assert report.pr_gate is not None
        assert report.pr_gate.should_open_pr is False
        assert retro_file.exists()
        parsed = json.loads(retro_file.read_text("utf-8"))
        assert parsed["task_id"].startswith("story-2-tester-")

    asyncio.run(scenario())


def test_pr_gate_only_opens_after_tester_success(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LessonStore(tmp_path / "memory")
        config = SupervisorConfig(retros_dir=str(tmp_path / "retros"))
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)

        report = await coordinator.execute(
            story_id="story-3",
            story_text="Implement feature Z",
            workers={
                AgentRole.ARCHITECT: FakeWorker(AgentRole.ARCHITECT, TaskState.COMPLETE),
                AgentRole.CODER: FakeWorker(AgentRole.CODER, TaskState.COMPLETE),
                AgentRole.TESTER: FakeWorker(AgentRole.TESTER, TaskState.COMPLETE, tests_passed=True),
            },
        )

        assert report.pr_gate is not None
        assert report.pr_gate.should_open_pr is True
        assert report.pr_gate.github_server_port == 8007

    asyncio.run(scenario())


def test_runtime_installs_into_orchestrator_and_registry(tmp_path: Path) -> None:
    registry = FakeRegistry()
    orchestrator = FakeOrchestrator()

    install_swarm_runtime(
        orchestrator=orchestrator,
        registry=registry,
        lessons_root=tmp_path / "memory",
    )

    assert hasattr(orchestrator, "run_hierarchical_story")
    names = {name for name, _factory in registry.calls}
    assert names == {"hierarchical_coordinator", "sdlc_debugger"}


# ---------------------------------------------------------------------------
# Additional tests (Wave 1 Sprint S004)
# ---------------------------------------------------------------------------


def test_missing_worker_returns_failed_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LessonStore(tmp_path / "memory")
        config = SupervisorConfig(retros_dir=str(tmp_path / "retros"))
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)

        # Only ARCHITECT and TESTER present — CODER is missing
        report = await coordinator.execute(
            story_id="story-missing",
            story_text="Missing worker test",
            workers={
                AgentRole.ARCHITECT: FakeWorker(AgentRole.ARCHITECT, TaskState.COMPLETE),
                AgentRole.TESTER: FakeWorker(AgentRole.TESTER, TaskState.COMPLETE, tests_passed=True),
            },
        )

        coder_result = next(
            (r for r in report.results if r.role == AgentRole.CODER), None
        )
        assert coder_result is not None
        assert coder_result.state == TaskState.FAILED
        assert coder_result.error_message == "worker_missing"

    asyncio.run(scenario())


def test_worker_without_execute_or_run_fails(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LessonStore(tmp_path / "memory")
        config = SupervisorConfig(retros_dir=str(tmp_path / "retros"))
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)

        class BrokenWorker:
            """Has neither execute nor run."""

        report = await coordinator.execute(
            story_id="story-broken",
            story_text="Broken worker test",
            workers={
                AgentRole.ARCHITECT: BrokenWorker(),
                AgentRole.CODER: FakeWorker(AgentRole.CODER, TaskState.COMPLETE),
                AgentRole.TESTER: FakeWorker(AgentRole.TESTER, TaskState.COMPLETE, tests_passed=True),
            },
        )

        architect_result = next(
            (r for r in report.results if r.role == AgentRole.ARCHITECT), None
        )
        assert architect_result is not None
        assert architect_result.state == TaskState.FAILED
        assert architect_result.error_message == "worker_contract_invalid"

    asyncio.run(scenario())


def test_lesson_not_injected_on_non_interval_cycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LessonStore(tmp_path / "memory")
        from core.swarm_models import CycleReport, PRGateDecision

        seed = CycleReport(
            cycle_number=1,
            story_id="seed",
            lessons_injected=[CycleLesson(cycle_number=1, lesson="Old lesson.", source_task_id="seed-1")],
            pr_gate=PRGateDecision(should_open_pr=False, reason="seed"),
        )
        await store.record_cycle(seed)

        config = SupervisorConfig(learning_interval=5, retros_dir=str(tmp_path / "retros"))
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)
        coordinator._cycle_counter = 2  # next call becomes cycle 3 — not divisible by 5

        architect = FakeWorker(AgentRole.ARCHITECT, TaskState.COMPLETE)
        coder = FakeWorker(AgentRole.CODER, TaskState.COMPLETE)
        tester = FakeWorker(AgentRole.TESTER, TaskState.COMPLETE, tests_passed=True)

        await coordinator.execute(
            story_id="story-no-lessons",
            story_text="No lessons test",
            workers={
                AgentRole.ARCHITECT: architect,
                AgentRole.CODER: coder,
                AgentRole.TESTER: tester,
            },
        )

        assert architect.contexts[0]["lessons"] == []
        assert coder.contexts[0]["lessons"] == []
        assert tester.contexts[0]["lessons"] == []

    asyncio.run(scenario())


def test_learning_interval_boundary_at_cycle_10(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LessonStore(tmp_path / "memory")
        from core.swarm_models import CycleReport, PRGateDecision

        seed = CycleReport(
            cycle_number=1,
            story_id="seed",
            lessons_injected=[CycleLesson(cycle_number=1, lesson="Boundary lesson.", source_task_id="seed-1")],
            pr_gate=PRGateDecision(should_open_pr=False, reason="seed"),
        )
        await store.record_cycle(seed)

        config = SupervisorConfig(learning_interval=5, retros_dir=str(tmp_path / "retros"))
        coordinator = HierarchicalCoordinatorAgent(config=config, lesson_store=store)
        coordinator._cycle_counter = 9  # next call becomes cycle 10 — divisible by 5

        architect = FakeWorker(AgentRole.ARCHITECT, TaskState.COMPLETE)
        coder = FakeWorker(AgentRole.CODER, TaskState.COMPLETE)
        tester = FakeWorker(AgentRole.TESTER, TaskState.COMPLETE, tests_passed=True)

        await coordinator.execute(
            story_id="story-boundary",
            story_text="Boundary lesson test",
            workers={
                AgentRole.ARCHITECT: architect,
                AgentRole.CODER: coder,
                AgentRole.TESTER: tester,
            },
        )

        assert architect.contexts[0]["lessons"] == ["Boundary lesson."]
        assert coder.contexts[0]["lessons"] == ["Boundary lesson."]
        assert tester.contexts[0]["lessons"] == ["Boundary lesson."]

    asyncio.run(scenario())


def test_sdlc_debugger_requirements_lens(tmp_path: Path) -> None:
    from agents.sdlc_debugger import SDLCDebuggerAgent
    from core.swarm_models import SDLCLens

    async def scenario() -> None:
        agent = SDLCDebuggerAgent()
        task = SwarmTask(
            task_id="t-req-1",
            title="Requirements lens test",
            role=AgentRole.TESTER,
            story_id="s-req",
            description="Check requirements lens",
        )
        result = TaskResult(
            task_id="t-req-1",
            role=AgentRole.TESTER,
            state=TaskState.FAILED,
            summary="The acceptance criteria were not met",
            tests_passed=False,
            error_message="acceptance criteria mismatch",
        )
        report = await agent.execute(task, result)
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.REQUIREMENTS in lenses

    asyncio.run(scenario())


def test_sdlc_debugger_testing_lens(tmp_path: Path) -> None:
    from agents.sdlc_debugger import SDLCDebuggerAgent
    from core.swarm_models import SDLCLens

    async def scenario() -> None:
        agent = SDLCDebuggerAgent()
        task = SwarmTask(
            task_id="t-test-1",
            title="Testing lens test",
            role=AgentRole.TESTER,
            story_id="s-test",
            description="Check testing lens",
        )
        result = TaskResult(
            task_id="t-test-1",
            role=AgentRole.TESTER,
            state=TaskState.FAILED,
            summary="pytest fixture failed",
            tests_passed=False,
            error_message="pytest assertion failure",
        )
        report = await agent.execute(task, result)
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.TESTING in lenses

    asyncio.run(scenario())


def test_sdlc_debugger_implementation_lens(tmp_path: Path) -> None:
    from agents.sdlc_debugger import SDLCDebuggerAgent
    from core.swarm_models import SDLCLens

    async def scenario() -> None:
        agent = SDLCDebuggerAgent()
        task = SwarmTask(
            task_id="t-impl-1",
            title="Implementation lens test",
            role=AgentRole.CODER,
            story_id="s-impl",
            description="Check implementation lens",
        )
        result = TaskResult(
            task_id="t-impl-1",
            role=AgentRole.CODER,
            state=TaskState.FAILED,
            summary="TypeError raised during execution",
            tests_passed=False,
            error_message="typeerror in function call",
        )
        report = await agent.execute(task, result)
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.IMPLEMENTATION in lenses

    asyncio.run(scenario())


def test_sdlc_debugger_no_match_defaults_to_implementation(tmp_path: Path) -> None:
    from agents.sdlc_debugger import SDLCDebuggerAgent
    from core.swarm_models import SDLCLens

    async def scenario() -> None:
        agent = SDLCDebuggerAgent()
        task = SwarmTask(
            task_id="t-noop-1",
            title="No match test",
            role=AgentRole.CODER,
            story_id="s-noop",
            description="No keywords present",
        )
        result = TaskResult(
            task_id="t-noop-1",
            role=AgentRole.CODER,
            state=TaskState.FAILED,
            summary="something went wrong",
            tests_passed=False,
            error_message=None,
        )
        report = await agent.execute(task, result)
        assert len(report.findings) == 1
        assert report.findings[0].lens == SDLCLens.IMPLEMENTATION

    asyncio.run(scenario())


def test_sdlc_debugger_security_lens(tmp_path: Path) -> None:
    from agents.sdlc_debugger import SDLCDebuggerAgent
    from core.swarm_models import SDLCLens

    async def scenario() -> None:
        agent = SDLCDebuggerAgent()
        task = SwarmTask(
            task_id="t-sec-1",
            title="Security lens test",
            role=AgentRole.CODER,
            story_id="s-sec",
            description="Check security lens",
        )
        result = TaskResult(
            task_id="t-sec-1",
            role=AgentRole.CODER,
            state=TaskState.FAILED,
            summary="expired token caused auth failure",
            tests_passed=False,
            error_message="token validation failed",
        )
        report = await agent.execute(task, result)
        lenses = [f.lens for f in report.findings]
        assert SDLCLens.SECURITY in lenses

    asyncio.run(scenario())


def test_sdlc_debugger_recovery_plan_always_starts_with_restate(tmp_path: Path) -> None:
    from agents.sdlc_debugger import SDLCDebuggerAgent

    async def scenario() -> None:
        agent = SDLCDebuggerAgent()
        task = SwarmTask(
            task_id="t-plan-1",
            title="Recovery plan test",
            role=AgentRole.CODER,
            story_id="s-plan",
            description="Check recovery plan",
        )
        result = TaskResult(
            task_id="t-plan-1",
            role=AgentRole.CODER,
            state=TaskState.FAILED,
            summary="Build failed for unknown reason",
            tests_passed=False,
            error_message=None,
        )
        report = await agent.execute(task, result)
        assert len(report.recovery_plan) > 0
        assert report.recovery_plan[0].startswith("Re-state")

    asyncio.run(scenario())


def test_install_swarm_runtime_sets_swarm_supervisor_attr(tmp_path: Path) -> None:
    registry = FakeRegistry()
    orchestrator = FakeOrchestrator()

    coordinator = install_swarm_runtime(
        orchestrator=orchestrator,
        registry=registry,
        lessons_root=tmp_path / "memory",
    )

    assert hasattr(orchestrator, "swarm_supervisor")
    assert orchestrator.swarm_supervisor is coordinator
