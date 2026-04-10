"""Core orchestration engine for the AURA autonomous coding loop.

This module implements :class:`LoopOrchestrator`, the central coordinator that
drives a multi-phase *plan -> critique -> synthesize -> act -> sandbox -> verify ->
reflect* pipeline.  Each phase is handled by a dedicated agent; the orchestrator
sequences them, routes failures, manages retries, and persists cycle history via
:class:`~memory.store.MemoryStore`.

Typical usage::

    from core.orchestrator import LoopOrchestrator
    from memory.store import MemoryStore

    orchestrator = LoopOrchestrator(agents=default_agents(brain, model),
                                     memory_store=MemoryStore())
    result = orchestrator.run_loop("Add a retry mechanism to the HTTP client",
                                   max_cycles=3)
    print(result["stop_reason"])
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.logging_utils import log_json
from core.schema import validate_phase_output
from core.operator_runtime import build_cycle_summary
from core.hooks import HookEngine
from core.phase_result import ConfidenceRouter
from core.quality_trends import QualityTrendAnalyzer
from core.config_manager import config
from core.policy import Policy
from core.file_tools import (
    MISMATCH_OVERWRITE_BLOCK_EVENT,
    MismatchOverwriteBlockedError,
    OldCodeNotFoundError,
    apply_change_with_explicit_overwrite_policy,
    mismatch_overwrite_block_log_details,
)
from core.skill_dispatcher import classify_goal
from core.human_gate import HumanGate
from core.types import TaskRequest, TaskResult, ExecutionContext
from core.mcp_agent_registry import agent_registry
from core.mcp_client import MCPAsyncClient
from memory.controller import memory_controller
from core.phase_dispatcher import PhaseDispatcher
from core.orchestrator_phases import PhasesMixin
from core.orchestrator_verify import VerifyMixin
from core.orchestrator_learn import LearnMixin
from core.orchestrator_capabilities import CapabilitiesMixin

MAX_SANDBOX_RETRIES = 3


class BeadsSyncLoop:
    """Triggers beads synchronization (dolt push/pull) periodically."""

    EVERY_N = 5

    def __init__(self, beads_skill):
        self._skill = beads_skill
        self._n = 0

    def on_cycle_complete(self, _entry):
        if isinstance(_entry, dict) and bool(_entry.get("dry_run")):
            return
        self._n += 1
        if self._n % self.EVERY_N == 0:
            log_json("INFO", "beads_sync_loop_starting")
            # Try to pull latest changes from remote
            self._skill.run({"cmd": "dolt", "args": ["pull"]})
            # Push local changes to remote
            self._skill.run({"cmd": "dolt", "args": ["push"]})


class LoopOrchestrator(PhasesMixin, VerifyMixin, LearnMixin, CapabilitiesMixin):
    """Coordinates the full AURA autonomous-coding pipeline across one or more cycles.

    Each *cycle* executes the following phases in order:

    1. **Ingest** -- gather project context and memory hints.
    2. **Skill dispatch** -- run adaptive static-analysis skills.
    3. **Plan** -- generate a step-by-step implementation plan (with retries).
    4. **Critique** -- adversarially review the plan for flaws.
    5. **Synthesize** -- merge plan + critique into an actionable task bundle.
    6. **Act** -- generate code changes (with retries on failure).
    7. **Sandbox** -- execute the generated snippet in an isolated subprocess.
    8. **Apply** -- write file changes to disk atomically.
    9. **Verify** -- run tests / linters against the applied changes.
    10. **Reflect** -- summarise outcomes and update skill weights.

    Failure routing (:meth:`_route_failure`) decides whether a failed
    verification warrants retrying the *act* phase, escalating to a full
    re-plan, or skipping (when the cause is environmental/external).

    Attributes:
        agents: Mapping of phase-name -> agent instance used for each pipeline step.
        memory_controller: Centralized memory authority.
        policy: Stopping-condition evaluator; defaults to :class:`~core.policy.Policy`
            with an empty config.
        project_root: Filesystem root against which all relative file paths are
            resolved.  Defaults to the current working directory.
        strict_schema: When ``True``, any phase that produces output not
            matching the expected JSON schema immediately aborts the cycle.
        debugger: Optional :class:`~agents.debugger.DebuggerAgent` invoked
            when :class:`~agents.coder.CoderAgent` emits an invalid change-set.
        skills: Dict of skill-name -> skill instance, loaded lazily from
            :mod:`agents.skills.registry`.
    """

    def __init__(
        self,
        agents: Dict[str, object],
        memory_store: Any = None,  # Deprecated in favor of global memory_controller
        policy: Policy = None,
        project_root: Path = None,
        strict_schema: bool = False,
        debugger=None,
        auto_add_capabilities: bool = True,
        auto_queue_missing_capabilities: bool = True,
        auto_provision_mcp: bool = False,
        auto_start_mcp_servers: bool = False,
        goal_queue=None,
        goal_archive=None,
        brain: Any = None,
        model: Any = None,
        runtime_mode: str = "full",
        beads_bridge: Any = None,
        beads_enabled: bool = False,
        beads_required: bool = False,
        beads_scope: str = "goal_run",
    ):
        """Initialise the orchestrator with its agents and supporting services.

        Args:
            agents: Dict mapping phase names (``"ingest"``, ``"plan"``, ``"act"``,
                etc.) to agent instances that implement a ``run(input_data)`` method.
            memory_store: [Deprecated] Now uses the global memory_controller.
            policy: Optional policy evaluator used to decide when to stop the
                :meth:`run_loop`.  Defaults to a permissive :class:`~core.policy.Policy`.
            project_root: Root directory of the target project.  All relative
                file paths in change-sets are resolved against this path.
                Defaults to the current working directory.
            strict_schema: When ``True``, any phase output that fails schema
                validation causes the cycle to stop immediately with
                ``stop_reason="INVALID_OUTPUT"``.
            debugger: Optional :class:`~agents.debugger.DebuggerAgent` consulted
                when the coder produces an invalid change-set.
        """
        self.agents = agents
        self.memory_controller = memory_controller
        if memory_store is not None:
            self.memory_controller.set_store(memory_store)
        self.policy = policy or Policy.from_config({})
        self.project_root = Path(project_root or ".")
        self.strict_schema = strict_schema
        self.config = config
        self.debugger = debugger  # Optional DebuggerAgent for auto-recovery
        self.human_gate = HumanGate()
        self.lesson_store = None
        try:
            from memory.lesson_store import LessonStore

            self.lesson_store = LessonStore()
        except (OSError, ImportError):
            pass
        self.auto_add_capabilities = auto_add_capabilities
        self.auto_queue_missing_capabilities = auto_queue_missing_capabilities
        self.auto_provision_mcp = auto_provision_mcp
        self.auto_start_mcp_servers = auto_start_mcp_servers
        self.goal_queue = goal_queue
        self.goal_archive = goal_archive
        self.brain = brain
        self.model = model
        self.runtime_mode = runtime_mode
        self.beads_bridge = beads_bridge
        self.beads_enabled = beads_enabled
        self.beads_required = beads_required
        self.beads_scope = beads_scope
        self.self_correction_agent = agents.get("self_correction")
        self.investigation_agent = agents.get("investigation")
        self.root_cause_analysis_agent = agents.get("root_cause_analysis")
        self.last_capability_plan: dict = {}
        self.last_capability_goal_queue: dict = {}
        self.last_capability_provisioning: dict = {}
        self.last_capability_status: dict = {}
        self.current_goal: str | None = None
        self.active_cycle_summary: dict | None = None
        self.last_cycle_summary: dict | None = None

        # Lazy-load skills so missing optional deps don't break startup
        try:
            from agents.skills.registry import all_skills

            self.skills = all_skills(brain=self.brain, model=self.model)
        except Exception as exc:  # pragma: no cover
            log_json("WARN", "skills_load_failed", details={"error": str(exc)})
            self.skills = {}

        # ── Innovation modules ──
        # Hook engine: guaranteed-execution lifecycle hooks at phase boundaries
        self.hook_engine = HookEngine(self._load_config_file())
        # Phase dispatcher: thin delegation layer for phase -> agent mapping + hooks
        self._phase_dispatcher = PhaseDispatcher(
            agents=self.agents,
            hook_engine=self.hook_engine,
            config=self.config,
            project_root=str(self.project_root),
        )
        # Confidence router: data-driven phase routing based on confidence scores
        self.confidence_router = ConfidenceRouter()
        # Quality trend analyzer: cross-cycle regression detection
        self.quality_trends = QualityTrendAnalyzer()
        # Skill correlation matrix: self-organizing skill system
        try:
            from core.skill_correlation import SkillCorrelationMatrix

            self.skill_correlation = SkillCorrelationMatrix()
        except (OSError, ImportError):
            self.skill_correlation = None

        # Self-improvement loops (all optional -- never block the main loop)
        self._improvement_loops: list = []

        # CASPA-W components -- set via attach_caspa() after construction
        self.adaptive_pipeline = None  # AdaptivePipeline
        self.propagation_engine = None  # PropagationEngine
        self.context_graph = None  # ContextGraph
        self._consecutive_fails: int = 0
        self._ui_callbacks: list = []

    def attach_ui_callback(self, callback) -> None:
        """Register a UI callback (e.g., AuraStudio) to receive real-time updates."""
        self._ui_callbacks.append(callback)

    def _notify_ui(self, method_name: str, *args, **kwargs):
        """Internal helper to safely trigger UI callbacks."""
        for cb in self._ui_callbacks:
            method = getattr(cb, method_name, None)
            if method:
                try:
                    method(*args, **kwargs)
                except (TypeError, AttributeError):
                    pass

    def attach_improvement_loops(self, *loops) -> None:
        """Register one or more self-improvement loops to be called after each cycle.

        Each *loop* object must expose an ``on_cycle_complete(entry: dict)``
        method.  Errors raised by individual loops are swallowed so they never
        interrupt the main pipeline.

        Args:
            *loops: Variadic sequence of improvement-loop instances to append.
                Multiple calls are additive -- loops are never de-duplicated.

        Example::

            orchestrator.attach_improvement_loops(SkillWeightUpdater(),
                                                   KnowledgeDistiller())
        """
        self._improvement_loops.extend(loops)
        log_json("INFO", "orchestrator_loops_attached", details={"count": len(self._improvement_loops), "types": [type(l).__name__ for l in self._improvement_loops]})

    def attach_caspa(
        self,
        adaptive_pipeline=None,
        propagation_engine=None,
        context_graph=None,
    ) -> None:
        """Attach CASPA-W components for contextually adaptive self-propagation.

        CASPA-W (Contextually Adaptive Self-Propagating Architecture -- Wired)
        extends the base pipeline with dynamic intensity scaling, cross-cycle
        knowledge propagation, and a live context dependency graph.

        All three components are optional; pass ``None`` to disable any of
        them.  Components can be attached after construction and will take
        effect on the next :meth:`run_cycle` call.

        Args:
            adaptive_pipeline: An :class:`~core.adaptive_pipeline.AdaptivePipeline`
                instance that overrides per-cycle phase configuration (intensity,
                skill set, retry limits, etc.).
            propagation_engine: A
                :class:`~core.propagation_engine.PropagationEngine` invoked after
                all improvement loops complete, allowing insights to flow into the
                next cycle's context.
            context_graph: A :class:`~core.context_graph.ContextGraph` that is
                updated after each cycle to track dependency relationships between
                files and goals.
        """
        self.adaptive_pipeline = adaptive_pipeline
        self.propagation_engine = propagation_engine
        self.context_graph = context_graph
        log_json(
            "INFO",
            "orchestrator_caspa_attached",
            details={
                "adaptive_pipeline": adaptive_pipeline is not None,
                "propagation_engine": propagation_engine is not None,
                "context_graph": context_graph is not None,
            },
        )

    def _retrieve_hints(self, goal: str, limit: int = 5) -> list[dict]:
        """Return the most relevant past cycle summaries for *goal*.

        Ranks by a weighted score: keyword overlap (50%), recency (30%),
        past outcome success (20%) -- replacing the old plain substring filter.

        Args:
            goal: Natural-language description of the current coding goal.
                Used to extract keywords for relevance scoring.
            limit: Maximum number of hint dicts to return.  Defaults to 5.

        Returns:
            A list of up to *limit* cycle-summary dicts from
            the memory controller, ordered from most to least relevant.
            Returns ``[]`` when the memory store is unavailable or empty.
        """
        if not self.memory_controller or not self.memory_controller.persistent_store:
            return []
        try:
            summaries = self.memory_controller.persistent_store.query("cycle_summaries", limit=200)
        except (OSError, AttributeError):
            return []
        if not summaries:
            return []

        goal_words = set(goal.lower().split())

        def _score(s: dict, rank: int, total: int) -> float:
            if not isinstance(s, dict):
                return 0.0
            summary_words = set(str(s).lower().split())
            kw_score = len(goal_words & summary_words) / max(len(goal_words), 1)
            recency = 1.0 - (rank / max(total, 1))
            outcome = 1.0 if s.get("status") == "success" else 0.3
            return kw_score * 0.5 + recency * 0.3 + outcome * 0.2

        total = len(summaries)
        ranked = sorted(
            enumerate(summaries),
            key=lambda iv: _score(iv[1], iv[0], total),
            reverse=True,
        )
        return [s for _, s in ranked[:limit]]

    def _load_config_file(self) -> dict:
        """Load aura.config.json for hook engine and other config."""
        config_path = self.project_root / "aura.config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                log_json("WARN", "config_file_load_failed", details={"path": str(config_path), "error": str(exc)})
        return {}

    def _run_phase(self, name: str, input_data: Dict) -> Dict:
        """Dispatch a single pipeline phase to its registered agent.

        Delegates to :attr:`_phase_dispatcher` which handles hook wrapping,
        canary routing, and agent lookup.  Shadow-mode comparison is still
        performed here because it requires access to :meth:`_dispatch_task`
        and :meth:`_run_shadow_check` which live on the orchestrator.

        Args:
            name: The phase identifier (e.g. ``"plan"``, ``"act"``, ``"verify"``).
            input_data: Arbitrary dict passed directly to ``agent.run()``.

        Returns:
            The dict returned by ``agent.run()``, or an empty dict ``{}`` when
            no agent is registered for *name*.
        """
        result = self._phase_dispatcher.dispatch(name, input_data)

        # Shadow mode comparison (M4-005, M5-005) -- kept here as it needs
        # access to _dispatch_task which is an orchestrator-level concern.
        if self.config.get("new_orchestrator_shadow_mode") and isinstance(result, dict) and not result.get("_blocked_by_hook"):
            self._run_shadow_check(name, input_data, result)

        return result

    def _run_shadow_check(self, name: str, input_data: Dict, sync_result: Dict):
        """Execute the async pipeline in the background and compare results (M4-005)."""
        import anyio
        import asyncio

        async def perform_check():
            req = TaskRequest(task_id=f"shadow_{uuid.uuid4().hex[:8]}", agent_name=name, input_data=input_data, context=ExecutionContext(project_root=str(self.project_root)))
            shadow_res = await self._dispatch_task(req)

            # Compare basic metadata
            mismatch = False
            if shadow_res.status != "success":
                mismatch = True
            elif set(sync_result.keys()) != set(shadow_res.output.keys()):
                mismatch = True

            log_json("INFO", "orchestrator_shadow_comparison", details={"phase": name, "status": shadow_res.status, "mismatch": mismatch, "sync_keys": list(sync_result.keys()), "shadow_keys": list(shadow_res.output.keys()) if shadow_res.status == "success" else []})

        try:
            # Try to run in current loop or create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We are in a sync method but maybe on an async thread?
                    # For shadow mode, we can block briefly or use a thread.
                    anyio.from_thread.run(perform_check)
                else:
                    anyio.run(perform_check)
            except RuntimeError:
                anyio.run(perform_check)
        except Exception as e:
            log_json("WARN", "orchestrator_shadow_check_failed", details={"error": str(e)})

    async def _dispatch_task(self, request: TaskRequest) -> TaskResult:
        """Async resolution and invocation pipeline (M4-002)."""
        import anyio

        agent_spec = agent_registry.get_agent(request.agent_name)

        # Fallback to local agents if not in typed registry
        if not agent_spec:
            legacy_agent = self.agents.get(request.agent_name)
            if not legacy_agent:
                return TaskResult(task_id=request.task_id, status="error", output={}, error=f"Agent {request.agent_name} not found")

            try:
                result = await anyio.to_thread.run_sync(legacy_agent.run, request.input_data)
                return TaskResult(task_id=request.task_id, status="success", output=result)
            except Exception as e:
                return TaskResult(task_id=request.task_id, status="error", output={}, error=str(e))

        if agent_spec.source == "mcp":
            from core.mcp_registry import get_registered_service

            try:
                service = get_registered_service(agent_spec.mcp_server)
                client = MCPAsyncClient(service["url"])
                result = await client.call_tool(agent_spec.name, request.input_data)
                return TaskResult(task_id=request.task_id, status="success", output=result)
            except Exception as e:
                return TaskResult(task_id=request.task_id, status="error", output={}, error=str(e))
        else:
            legacy_agent = self.agents.get(request.agent_name)
            try:
                result = await anyio.to_thread.run_sync(legacy_agent.run, request.input_data)
                return TaskResult(task_id=request.task_id, status="success", output=result)
            except Exception as e:
                return TaskResult(task_id=request.task_id, status="error", output={}, error=str(e))

    def _snapshot_file_state(self, file_path: str) -> Dict:
        """Capture a restorable snapshot of a target file before mutation."""
        target = (self.project_root / file_path).resolve()
        snapshot = {
            "file": file_path,
            "target": str(target),
            "existed": target.exists(),
            "content": None,
            "mode": None,
        }
        if target.exists():
            snapshot["content"] = target.read_text(encoding="utf-8", errors="ignore")
            snapshot["mode"] = target.stat().st_mode & 0o777
        return snapshot

    def _apply_change_set(self, change_set: Dict, dry_run: bool) -> Dict[str, List]:
        """Apply each change in *change_set* independently to the filesystem.

        Unlike the old all-or-nothing approach, a failure on one file is
        recorded and the loop continues so that other files are still applied.
        Callers should inspect ``result["failed"]`` to decide whether the
        verification phase should run.

        Args:
            change_set: A dict in one of two forms:

                * A single change: ``{"file_path": ..., "old_code": ...,
                  "new_code": ...}``.
                * A batch: ``{"changes": [{"file_path": ..., ...}, ...]}``.

                An optional ``"overwrite_file": True`` key on any individual
                change forces a full-file overwrite instead of a targeted
                ``old_code`` replacement.
            dry_run: When ``True``, changes are logged but **not** written to
                disk.  All paths are recorded in ``"applied"`` as if they
                succeeded.

        Returns:
            A dict with two keys:

            * ``"applied"`` (``List[str]``) -- file paths successfully written.
            * ``"failed"``  (``List[dict]``) -- dicts with ``"file"`` and
              ``"error"`` keys for each path that could not be written.
            * ``"snapshots"`` (``List[dict]``) -- pre-apply file snapshots for
              restoring loop-owned writes when verification fails.
        """
        changes: List[Dict] = []
        if isinstance(change_set, dict):
            if all(k in change_set for k in ["file_path", "old_code", "new_code"]):
                changes.append(change_set)
            elif "changes" in change_set and isinstance(change_set["changes"], list):
                changes.extend(change_set["changes"])

        applied: List[str] = []
        failed: List[Dict] = []
        snapshots: List[Dict] = []
        snapshotted_files: set[str] = set()

        for change in changes:
            file_path = change.get("file_path")
            old_code = change.get("old_code")
            new_code = change.get("new_code")
            overwrite_file = change.get("overwrite_file", False)

            if not file_path:
                log_json("WARN", "apply_change_skipped_missing_path", details={"change": change})
                continue

            if dry_run:
                log_json("INFO", "replace_code_skipped", details={"reason": "dry_run", "file": file_path})
                applied.append(file_path)
                continue

            try:
                if file_path not in snapshotted_files:
                    snapshots.append(self._snapshot_file_state(file_path))
                    snapshotted_files.add(file_path)
                apply_change_with_explicit_overwrite_policy(
                    self.project_root,
                    file_path,
                    old_code,
                    new_code,
                    overwrite_file=overwrite_file,
                )
                applied.append(file_path)
            except MismatchOverwriteBlockedError as exc:
                log_json("ERROR", MISMATCH_OVERWRITE_BLOCK_EVENT, details=mismatch_overwrite_block_log_details(exc, file_path))
                failed.append({"file": file_path, "error": str(exc)})
            except OldCodeNotFoundError as exc:
                log_json("ERROR", "old_code_not_found", details={"error": str(exc), "file": file_path})
                failed.append({"file": file_path, "error": str(exc)})
            except Exception as exc:
                log_json("ERROR", "apply_change_failed", details={"error": str(exc), "file": file_path})
                failed.append({"file": file_path, "error": str(exc)})

        return {"applied": applied, "failed": failed, "snapshots": snapshots}

    # ── Phase execution, verification, learning, and capability methods are
    # ── provided by PhasesMixin, VerifyMixin, LearnMixin, CapabilitiesMixin.

    def _run_pre_plan_phases(
        self,
        goal: str,
        goal_type: str,
        cycle_id: str,
        pipeline_cfg: Dict,
        phase_outputs: Dict,
        started_at: float,
        dry_run: bool,
    ) -> tuple:
        """Run ingest, MCP discovery, skill dispatch, beads gate, and coverage backfill.

        Returns (context, skill_context, early_return) where early_return is
        non-None if the cycle should stop early.
        """
        # Inject lessons from previous cycles into planner context
        if self.lesson_store:
            try:
                lessons = self.lesson_store.injectable_lessons()
                if lessons:
                    phase_outputs["injected_lessons"] = lessons
            except (OSError, TypeError):
                pass

        context = self._run_ingest_phase(goal, cycle_id, phase_outputs)
        if self.strict_schema and validate_phase_output("context", context):
            return (
                context,
                {},
                self._build_early_stop_entry(
                    cycle_id=cycle_id,
                    goal=goal,
                    goal_type=goal_type,
                    phase_outputs=phase_outputs,
                    started_at=started_at,
                    stop_reason="INVALID_OUTPUT",
                ),
            )

        # Autonomous MCP capability injection
        mcp_discovery = self._run_mcp_discovery_phase(phase_outputs)
        if mcp_discovery.get("status") == "success" and mcp_discovery.get("discovered"):
            log_json("INFO", "mcp_servers_discovered", details={"count": len(mcp_discovery["discovered"])})

        skill_context = self._dispatch_skills(goal_type, pipeline_cfg, phase_outputs)

        # Beads gate
        early = self._check_beads_gate(goal, goal_type, context, skill_context, cycle_id, phase_outputs, started_at)
        if early is not None:
            return context, skill_context, early

        # Coverage Backfill
        gaps = skill_context.get("structural_analyzer", {}).get("coverage_gaps", [])
        if getattr(self, "auto_backfill_coverage", False) and self.goal_queue:
            for gap in gaps:
                f = gap.get("file")
                priority = gap.get("risk_priority", "MEDIUM")
                backfill_goal = f"test_backfill: Write missing unit tests for '{f}' to resolve coverage gap (priority: {priority})"
                self.goal_queue.add(backfill_goal)
                log_json("INFO", "backfill_goal_enqueued", details={"file": f, "priority": priority})

        if gaps:
            context["backfill_context"] = gaps

        return context, skill_context, None

    def _check_beads_gate(
        self,
        goal: str,
        goal_type: str,
        context: Dict,
        skill_context: Dict,
        cycle_id: str,
        phase_outputs: Dict,
        started_at: float,
    ) -> Optional[Dict]:
        """Run beads gate checks and return an early-stop entry if blocked, else None."""
        if not self._beads_gate_applies():
            return None

        beads_gate = self._run_beads_gate(goal, goal_type, context, skill_context)
        phase_outputs["beads_gate"] = beads_gate

        if not beads_gate.get("ok") and self.beads_required:
            return self._build_early_stop_entry(
                cycle_id=cycle_id,
                goal=goal,
                goal_type=goal_type,
                phase_outputs=phase_outputs,
                started_at=started_at,
                stop_reason="BEADS_UNAVAILABLE",
                beads=beads_gate,
            )

        status = beads_gate.get("status")
        if status == "block":
            return self._build_early_stop_entry(
                cycle_id=cycle_id,
                goal=goal,
                goal_type=goal_type,
                phase_outputs=phase_outputs,
                started_at=started_at,
                stop_reason=beads_gate.get("stop_reason") or "BEADS_BLOCKED",
                beads=beads_gate,
            )
        if status == "revise":
            return self._build_early_stop_entry(
                cycle_id=cycle_id,
                goal=goal,
                goal_type=goal_type,
                phase_outputs=phase_outputs,
                started_at=started_at,
                stop_reason=beads_gate.get("stop_reason") or "BEADS_REVISE_REQUIRED",
                beads=beads_gate,
            )
        return None

    def _run_post_verify_phases(
        self,
        goal: str,
        goal_type: str,
        cycle_id: str,
        phase_outputs: Dict,
        started_at: float,
        verification: Dict,
        skill_context: Dict,
    ) -> Optional[Dict]:
        """Run reflection, learning, confidence recording, and outcome persistence.

        Returns the early-stop entry if reflection validation fails, else None
        (caller should use _record_cycle_outcome result).
        """
        reflection = self._run_reflection_phase(verification, skill_context, goal_type, cycle_id, phase_outputs)
        if self.strict_schema and validate_phase_output("reflection", reflection):
            return self._build_early_stop_entry(
                cycle_id=cycle_id,
                goal=goal,
                goal_type=goal_type,
                phase_outputs=phase_outputs,
                started_at=started_at,
                stop_reason="INVALID_OUTPUT",
                beads=phase_outputs.get("beads_gate"),
            )

        phase_outputs["cycle_confidence"] = self.confidence_router.get_cycle_confidence()
        phase_outputs.pop("_failure_context", None)
        return None

    def run_cycle(self, goal: str, dry_run: bool = False, context_injection: Optional[Dict] = None) -> Dict:
        """Execute a single complete plan-act-verify cycle for *goal*.

        Parameters
        ----------
        context_injection:
            Optional dict merged into the ingest phase input_data.  Used by
            SADD SubAgentRunner to pass dependency context from completed
            workstreams.
        """
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        started_at = time.time()
        phase_outputs = {"retry_count": 0, "dry_run": dry_run}
        if context_injection:
            phase_outputs["context_injection"] = context_injection
        self.confidence_router.reset()
        self._notify_ui("on_cycle_start", goal)

        bead_id = self._parse_bead_id(goal)
        if bead_id and not dry_run:
            self._claim_bead(bead_id)

        goal_type = classify_goal(goal)
        self.current_goal = goal
        self.active_cycle_summary = build_cycle_summary(
            cycle_id=cycle_id,
            goal=goal,
            goal_type=goal_type,
            phase_outputs=phase_outputs,
            state="running",
            started_at=started_at,
        )
        pipeline_cfg = self._configure_pipeline(goal, goal_type, phase_outputs)
        self._handle_capabilities(goal, pipeline_cfg, phase_outputs, dry_run)

        # ── Pre-plan phases: ingest, MCP, skills, beads, backfill ──
        context, skill_context, early_return = self._run_pre_plan_phases(
            goal,
            goal_type,
            cycle_id,
            pipeline_cfg,
            phase_outputs,
            started_at,
            dry_run,
        )
        if early_return is not None:
            return early_return

        # ── Plan loop ──
        verification, early_return = self._run_plan_loop(
            goal=goal,
            context=context,
            skill_context=skill_context,
            pipeline_cfg=pipeline_cfg,
            cycle_id=cycle_id,
            phase_outputs=phase_outputs,
            dry_run=dry_run,
        )
        if early_return is not None:
            log_json("INFO", "n8n_early_exit_attempting", details={"cycle_id": cycle_id, "goal": goal[:80]})
            try:
                self._notify_n8n_feedback(goal, cycle_id, False, phase_outputs)
                log_json("INFO", "n8n_early_exit_feedback_ok", details={"cycle_id": cycle_id})
            except Exception as exc:
                log_json("WARN", "n8n_early_exit_feedback_failed", details={"error": str(exc), "cycle_id": cycle_id})
            return early_return

        # ── Post-verify phases: reflection, confidence ──
        early_stop = self._run_post_verify_phases(
            goal,
            goal_type,
            cycle_id,
            phase_outputs,
            started_at,
            verification,
            skill_context,
        )
        if early_stop is not None:
            return early_stop

        result = self._record_cycle_outcome(cycle_id, goal, goal_type, phase_outputs, started_at)

        # Record lesson from this cycle
        if self.lesson_store:
            try:
                self.lesson_store.record_cycle(result)
            except (OSError, AttributeError):
                pass

        return result

    def _estimate_confidence(self, output: Dict, phase: str) -> float:
        """Heuristically estimate confidence for a phase output.

        Returns a float between 0.0 and 1.0 based on phase-specific signals.
        """
        if not isinstance(output, dict):
            return 0.3
        confidence = 0.5  # baseline

        if phase == "plan":
            steps = output.get("steps", [])
            if isinstance(steps, list):
                if len(steps) >= 2:
                    confidence += 0.2
                if any("test" in str(s).lower() for s in steps):
                    confidence += 0.1
            if output.get("estimated_complexity"):
                confidence += 0.05
        elif phase == "act":
            changes = output.get("changes", [])
            if isinstance(changes, list) and len(changes) > 0:
                confidence += 0.2
                if all(c.get("file_path") for c in changes if isinstance(c, dict)):
                    confidence += 0.1
        elif phase == "verify":
            status = output.get("status", "")
            if status == "pass":
                confidence = 0.95
            elif status == "skip":
                confidence = 0.6
            elif status == "fail":
                confidence = 0.1
        elif phase == "critique":
            if output.get("issues") or output.get("suggestions"):
                confidence += 0.2

        return min(1.0, max(0.0, confidence))

    def poll_external_goals(self) -> List[str]:
        """Poll external systems (like BEADS) for new goals.

        Returns:
            A list of goal description strings.
        """
        new_goals = []

        beads_skill = self._get_beads_skill()
        if beads_skill is not None:
            try:
                log_json("INFO", "orchestrator_polling_beads")
                result = beads_skill.run({"cmd": "ready"})

                # 'bd ready --json' returns a list of beads or an object with a list
                beads = []
                if isinstance(result, list):
                    beads = result
                elif isinstance(result, dict) and "beads" in result:
                    beads = result["beads"]
                elif isinstance(result, dict) and "ready" in result:
                    beads = result["ready"]

                for bead in beads:
                    if isinstance(bead, dict):
                        title = bead.get("title") or bead.get("summary")
                        bead_id = bead.get("id")
                        if title and bead_id:
                            goal = f"bead:{bead_id}: {title}"
                            new_goals.append(goal)
            except Exception as exc:
                log_json("WARN", "beads_poll_failed", details={"error": str(exc)})

        return new_goals

    def run_loop(self, goal: str, max_cycles: int = 5, dry_run: bool = False, context_injection: Optional[Dict] = None) -> Dict:
        """Run :meth:`run_cycle` repeatedly until a stopping condition is met.

        Stopping conditions (checked after every cycle in priority order):

        1. The cycle itself sets ``stop_reason`` (e.g. ``"INVALID_OUTPUT"``).
        2. :attr:`policy` returns a non-empty stop reason based on cycle history.
        3. *max_cycles* have been executed.

        Args:
            goal: Natural-language description of the coding task to complete.
            max_cycles: Hard upper limit on the number of cycles to run.
                Defaults to 5.
            dry_run: Passed through to each :meth:`run_cycle` call.  When
                ``True``, no filesystem changes are made.

        Returns:
            A dict with the following keys:

            * ``"goal"`` (str) -- the original goal string.
            * ``"stop_reason"`` (str) -- why the loop terminated.  One of the
              policy-defined reasons, a cycle-level ``"INVALID_OUTPUT"``, or
              ``"MAX_CYCLES"`` when the hard limit was reached.
            * ``"history"`` (list[dict]) -- list of cycle-result dicts in
              execution order, each as returned by :meth:`run_cycle`.

        Example::

            orchestrator = LoopOrchestrator(agents, memory_store)
            result = orchestrator.run_loop("Refactor auth module", max_cycles=3)
            for cycle in result["history"]:
                print(cycle["cycle_id"], cycle["phase_outputs"]["verification"])
        """
        history = []
        stop_reason = ""
        started_at = time.time()
        try:
            for _ in range(max_cycles):
                entry = self.run_cycle(goal, dry_run=dry_run, context_injection=context_injection)
                history.append(entry)
                if entry.get("stop_reason"):
                    stop_reason = entry["stop_reason"]
                    break
                stop_reason = self.policy.evaluate(history, entry.get("phase_outputs", {}).get("verification", {}), started_at=started_at)
                if stop_reason:
                    entry["stop_reason"] = stop_reason
                    self._refresh_cycle_summary(entry)
                    break
            if not stop_reason and history:
                history[-1]["stop_reason"] = "MAX_CYCLES"
                self._refresh_cycle_summary(history[-1])

            # If goal was a bead and we passed, close it
            if stop_reason == "PASS" and not dry_run:
                bead_id = self._parse_bead_id(goal)
                if bead_id:
                    self._close_bead(bead_id, reason="AURA successfully completed the goal.")

            return {
                "goal": goal,
                "stop_reason": stop_reason or "MAX_CYCLES",
                "history": history,
            }
        finally:
            # M7-003: Cleanup async resources
            import anyio

            try:
                anyio.run(self.shutdown)
            except (RuntimeError, OSError):
                pass

    async def shutdown(self):
        """Shut down the orchestrator and clean up resources (M7-003)."""
        await MCPAsyncClient.close_all()
        log_json("INFO", "orchestrator_shutdown_complete")
