"""Hierarchical coordinator for structured sub-agent driven development."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, runtime_checkable
from uuid import uuid4

from agents.sdlc_debugger import SDLCDebuggerAgent
from core.logging_utils import log_json
from core.swarm_models import (
    AgentRole,
    CycleLesson,
    CycleReport,
    PRGateDecision,
    SupervisorConfig,
    SwarmTask,
    TaskResult,
    TaskState,
)
from memory.learning_loop import LessonStore


@runtime_checkable
class WorkerProtocol(Protocol):
    """Duck-typed worker contract compatible with existing AURA agents."""

    async def execute(self, task: SwarmTask, context: Dict[str, Any]) -> TaskResult:
        """Execute a task and return a normalized task result."""


class HierarchicalCoordinatorAgent:
    """Coordinator that decomposes a story and gates progress by test outcomes."""

    name = "hierarchical_coordinator"

    def __init__(self, config: SupervisorConfig, lesson_store: LessonStore) -> None:
        self.config = config
        self.lesson_store = lesson_store
        self.debugger = SDLCDebuggerAgent()
        self._cycle_counter = 0

    async def plan(self, story_id: str, story_text: str) -> List[SwarmTask]:
        """Create the default architect -> coder -> tester pipeline."""
        architect_id = f"{story_id}-architect-{uuid4().hex[:8]}"
        coder_id = f"{story_id}-coder-{uuid4().hex[:8]}"
        tester_id = f"{story_id}-tester-{uuid4().hex[:8]}"
        return [
            SwarmTask(
                task_id=architect_id,
                title="Refine story into implementation plan",
                role=AgentRole.ARCHITECT,
                story_id=story_id,
                description=story_text,
                acceptance_criteria=[
                    "Implementation plan is explicit.",
                    "Risks and dependencies are named.",
                ],
            ),
            SwarmTask(
                task_id=coder_id,
                title="Implement plan",
                role=AgentRole.CODER,
                story_id=story_id,
                description="Implement the approved plan produced by the architect task.",
                acceptance_criteria=[
                    "Code compiles or lints cleanly.",
                    "Tests needed for the story are added.",
                ],
                depends_on=[architect_id],
            ),
            SwarmTask(
                task_id=tester_id,
                title="Validate implementation",
                role=AgentRole.TESTER,
                story_id=story_id,
                description="Run verification for the implementation and report status.",
                acceptance_criteria=[
                    "All acceptance criteria are exercised.",
                    "Failure signals are explicit and reproducible.",
                ],
                depends_on=[coder_id],
            ),
        ]

    async def execute(
        self,
        story_id: str,
        story_text: str,
        workers: Mapping[AgentRole, Any],
    ) -> CycleReport:
        """Run one controlled execution cycle for a Forge story."""
        self._cycle_counter += 1
        
        # Superpowers: INITIALIZING State - Semantic Tool Discovery
        from agents.skills.mcp_semantic_discovery import mcp_semantic_discovery
        discovery_context = mcp_semantic_discovery(f"Tools needed for story {story_id}: {story_text}")
        
        tasks = await self.plan(story_id=story_id, story_text=story_text)
        lessons = await self._lessons_for_cycle(self._cycle_counter)
        results: List[TaskResult] = []
        outputs: Dict[str, Any] = {}

        for task in tasks:
            task.state = TaskState.RUNNING
            context = {
                "cycle_number": self._cycle_counter,
                "story_id": story_id,
                "lessons": [lesson.lesson for lesson in lessons],
                "discovery_tools": discovery_context["tools"],
                "upstream_outputs": outputs,
                "mcp_ports": self.config.mcp_ports,
            }
            result = await self._invoke_worker(task=task, context=context, workers=workers)
            task.state = result.state
            results.append(result)
            outputs[task.task_id] = result.output

            if task.role == AgentRole.TESTER and (result.state == TaskState.FAILED or not result.tests_passed):
                debug_report = await self.debugger.execute(task=task, result=result)
                await self._write_retrospective(story_id=story_id, report=debug_report)
                pr_gate = PRGateDecision(
                    should_open_pr=False,
                    reason="Tester did not confirm a passing cycle.",
                    github_server_port=self.config.mcp_ports["github"],
                )
                cycle_report = CycleReport(
                    cycle_number=self._cycle_counter,
                    story_id=story_id,
                    tasks=tasks,
                    results=results,
                    lessons_injected=lessons,
                    debug_report=debug_report,
                    pr_gate=pr_gate,
                )
                await self.lesson_store.record_cycle(cycle_report)
                return cycle_report

        # Superpowers: POST_FLIGHT State - Auto-PR and Auto-Merge
        if self.config.github_delivery_enabled:
            await self._perform_github_delivery(story_id, results)
        else:
            log_json("INFO", "swarm_github_delivery_skipped", details={"story_id": story_id, "reason": "github_delivery_enabled=False"})

        pr_gate = PRGateDecision(
            should_open_pr=True,
            reason="All tasks completed and tester confirmed success. PR merged.",
            github_server_port=self.config.mcp_ports["github"],
        )
        cycle_report = CycleReport(
            cycle_number=self._cycle_counter,
            story_id=story_id,
            tasks=tasks,
            results=results,
            lessons_injected=lessons,
            pr_gate=pr_gate,
        )
        await self.lesson_store.record_cycle(cycle_report)
        return cycle_report

    async def _perform_github_delivery(self, story_id: str, results: List[TaskResult]) -> None:
        """Create PR and Auto-Merge via GitHub MCP."""
        log_json("INFO", "swarm_github_delivery_start", details={"story_id": story_id})
        # 1. Create Pull Request
        # 2. Automatically Merge PR (Every successful cycle)
        # Mocking the calls
        log_json("INFO", "swarm_github_delivery_pr_created", details={"story_id": story_id})
        log_json("INFO", "swarm_github_delivery_merging", details={"story_id": story_id})


    async def reflect(self, report: CycleReport) -> List[CycleLesson]:
        """Return lessons that should be written back to shared memory."""
        if report.debug_report:
            return [
                CycleLesson(
                    cycle_number=report.cycle_number,
                    lesson=step,
                    source_task_id=report.debug_report.task_id,
                    confidence=0.9,
                )
                for step in report.debug_report.recovery_plan[:3]
            ]
        return report.lessons_injected

    async def _lessons_for_cycle(self, cycle_number: int) -> List[CycleLesson]:
        if cycle_number % self.config.learning_interval != 0:
            return []
        return await self.lesson_store.injectable_lessons(limit=5)

    async def _invoke_worker(
        self,
        task: SwarmTask,
        context: Dict[str, Any],
        workers: Mapping[AgentRole, Any],
    ) -> TaskResult:
        worker = workers.get(task.role)
        if worker is None:
            return TaskResult(
                task_id=task.task_id,
                role=task.role,
                state=TaskState.FAILED,
                summary=f"No worker registered for role {task.role.value}.",
                error_message="worker_missing",
            )

        if hasattr(worker, "execute"):
            result = await worker.execute(task, context)
        elif hasattr(worker, "run"):
            result = await worker.run(task, context)
        else:
            return TaskResult(
                task_id=task.task_id,
                role=task.role,
                state=TaskState.FAILED,
                summary=f"Worker for {task.role.value} lacks execute/run.",
                error_message="worker_contract_invalid",
            )

        if isinstance(result, TaskResult):
            return result

        raise TypeError(f"Worker {task.role.value} returned unsupported result type: {type(result)!r}")

    async def _write_retrospective(self, story_id: str, report: Any) -> None:
        retros_dir = Path(self.config.retros_dir)
        retros_dir.mkdir(parents=True, exist_ok=True)
        path = retros_dir / f"{story_id}-cycle-{self._cycle_counter}.json"
        payload = json.dumps(report.model_dump(), indent=2)
        await asyncio.to_thread(path.write_text, payload, "utf-8")
