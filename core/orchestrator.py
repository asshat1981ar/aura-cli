"""Core orchestration engine for the AURA autonomous coding loop.

This module implements :class:`LoopOrchestrator`, the central coordinator that
drives a multi-phase *plan → critique → synthesize → act → sandbox → verify →
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

import os
import json
import dataclasses
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.logging_utils import log_json
from core.beads_bridge import build_beads_runtime_input
from core.operator_runtime import build_cycle_summary
from core.cycle_outcome import CycleOutcome
from core.hooks import HookEngine
from core.phase_result import PhaseResult, ConfidenceRouter
from core.quality_trends import QualityTrendAnalyzer
from core.config_manager import config
from core.capability_manager import (
    analyze_capability_needs,
    provision_capability_actions,
    queue_missing_capability_goals,
    record_capability_status,
)
from core.policy import Policy
from core.file_tools import (
    MISMATCH_OVERWRITE_BLOCK_EVENT,
    MismatchOverwriteBlockedError,
    OldCodeNotFoundError,
    apply_change_with_explicit_overwrite_policy,
    mismatch_overwrite_block_log_details,
)
from core.schema import validate_phase_output
from core.skill_dispatcher import classify_goal, dispatch_skills
from core.human_gate import HumanGate
from core.types import TaskRequest, TaskResult, ExecutionContext
from core.mcp_agent_registry import agent_registry
from core.mcp_client import MCPAsyncClient
from memory.controller import memory_controller, MemoryTier
from core.failure_router import FailureAction, FailureRouter

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


class LoopOrchestrator:
    """Coordinates the full AURA autonomous-coding pipeline across one or more cycles.

    Each *cycle* executes the following phases in order:

    1. **Ingest** — gather project context and memory hints.
    2. **Skill dispatch** — run adaptive static-analysis skills.
    3. **Plan** — generate a step-by-step implementation plan (with retries).
    4. **Critique** — adversarially review the plan for flaws.
    5. **Synthesize** — merge plan + critique into an actionable task bundle.
    6. **Act** — generate code changes (with retries on failure).
    7. **Sandbox** — execute the generated snippet in an isolated subprocess.
    8. **Apply** — write file changes to disk atomically.
    9. **Verify** — run tests / linters against the applied changes.
    10. **Reflect** — summarise outcomes and update skill weights.

    Failure routing (:meth:`_route_failure`) decides whether a failed
    verification warrants retrying the *act* phase, escalating to a full
    re-plan, or skipping (when the cause is environmental/external).

    Attributes:
        agents: Mapping of phase-name → agent instance used for each pipeline step.
        memory_controller: Centralized memory authority.
        policy: Stopping-condition evaluator; defaults to :class:`~core.policy.Policy`
            with an empty config.
        project_root: Filesystem root against which all relative file paths are
            resolved.  Defaults to the current working directory.
        strict_schema: When ``True``, any phase that produces output not
            matching the expected JSON schema immediately aborts the cycle.
        debugger: Optional :class:`~agents.debugger.DebuggerAgent` invoked
            when :class:`~agents.coder.CoderAgent` emits an invalid change-set.
        skills: Dict of skill-name → skill instance, loaded lazily from
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
        # Confidence router: data-driven phase routing based on confidence scores
        self.confidence_router = ConfidenceRouter()
        # Quality trend analyzer: cross-cycle regression detection
        self.quality_trends = QualityTrendAnalyzer()
        # Skill correlation matrix: self-organizing skill system
        try:
            from core.skill_correlation import SkillCorrelationMatrix

            self.skill_correlation = SkillCorrelationMatrix()
        except Exception:
            self.skill_correlation = None

        # Self-improvement loops (all optional — never block the main loop)
        self._improvement_loops: list = []

        # CASPA-W components — set via attach_caspa() after construction
        self.adaptive_pipeline = None  # AdaptivePipeline
        self.propagation_engine = None  # PropagationEngine
        self.context_graph = None  # ContextGraph
        self._consecutive_fails: int = 0
        self._ui_callbacks: list = []

        # Swarm integration — set by install_swarm_runtime when AURA_ENABLE_SWARM=1
        self.lesson_store: Any = None

        # Failure routing — delegates to dedicated FailureRouter module
        self._failure_router = FailureRouter(max_act_retries=MAX_SANDBOX_RETRIES)

    def _get_beads_skill(self):
        """Return the BEADS skill only when runtime BEADS integration is enabled."""
        if not self.beads_enabled:
            return None
        return self.skills.get("beads_skill")

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
                except Exception:
                    pass

    def _analyze_error(self, error: str, context: Optional[dict] = None) -> Optional[str]:
        """Optionally use SelfCorrectionAgent to analyze and suggest fixes."""
        if not self.self_correction_agent:
            return None
        try:
            return self.self_correction_agent.analyze_error(error, context or {})
        except Exception as exc:
            log_json("WARN", "self_correction_analysis_failed", details={"error": str(exc)})
            return None

    def _run_root_cause_analysis(
        self,
        failures: List[str],
        logs: str,
        context: Optional[dict] = None,
        *,
        history: Optional[List[dict]] = None,
    ) -> Optional[dict]:
        """Optionally produce a structured RCA report for a failed phase."""
        if not self.root_cause_analysis_agent:
            return None
        try:
            return self.root_cause_analysis_agent.run(
                {
                    "failures": failures,
                    "logs": logs,
                    "context": context or {},
                    "history": history or [],
                }
            )
        except Exception as exc:
            log_json("WARN", "root_cause_analysis_failed", details={"error": str(exc)})
            return None

    def _run_investigation(
        self,
        *,
        goal: str,
        verification: Dict[str, Any],
        context: Optional[dict] = None,
        route: str = "act",
        analysis_suggestion: str | None = None,
        root_cause_analysis: Optional[dict] = None,
        previous_test_count: int | None = None,
        current_test_count: int | None = None,
    ) -> Optional[dict]:
        """Optionally produce a structured investigation report for a failed phase."""
        if not self.investigation_agent:
            return None
        try:
            return self.investigation_agent.run(
                {
                    "goal": goal,
                    "verification": verification,
                    "context": context or {},
                    "route": route,
                    "analysis_suggestion": analysis_suggestion,
                    "root_cause_analysis": root_cause_analysis or {},
                    "history": self._failure_history(),
                    "previous_test_count": previous_test_count,
                    "current_test_count": current_test_count,
                }
            )
        except Exception as exc:
            log_json("WARN", "investigation_failed", details={"error": str(exc)})
            return None

    def _failure_history(self, limit: int = 5) -> List[dict]:
        """Return recent cycle summaries to help classify repeated failures."""
        if not self.memory_controller or not self.memory_controller.persistent_store:
            return []
        try:
            return list(self.memory_controller.read_log()[-limit:])
        except Exception:
            return []

    def attach_improvement_loops(self, *loops) -> None:
        """Register one or more self-improvement loops to be called after each cycle.

        Each *loop* object must expose an ``on_cycle_complete(entry: dict)``
        method.  Errors raised by individual loops are swallowed so they never
        interrupt the main pipeline.

        Args:
            *loops: Variadic sequence of improvement-loop instances to append.
                Multiple calls are additive — loops are never de-duplicated.

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

        CASPA-W (Contextually Adaptive Self-Propagating Architecture — Wired)
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
        past outcome success (20%) — replacing the old plain substring filter.

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
        except Exception:
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
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _run_phase(self, name: str, input_data: Dict) -> Dict:
        """Dispatch a single pipeline phase to its registered agent.

        Wraps execution with guaranteed-execution pre/post hooks and
        records confidence scores for routing decisions.

        Args:
            name: The phase identifier (e.g. ``"plan"``, ``"act"``, ``"verify"``).
            input_data: Arbitrary dict passed directly to ``agent.run()``.

        Returns:
            The dict returned by ``agent.run()``, or an empty dict ``{}`` when
            no agent is registered for *name*.
        """
        # Emergency bypass (M5 rollback)
        if self.config.get("force_legacy_orchestrator"):
            agent = self.agents.get(name)
            return agent.run(input_data) if agent else {}

        # Pre-phase hooks (guaranteed execution — cannot be bypassed by model)
        should_proceed, input_data = self.hook_engine.run_pre_hooks(name, input_data)
        if not should_proceed:
            log_json("WARN", "phase_blocked_by_hook", details={"phase": name})
            return {"_blocked_by_hook": True, "phase": name}

        # M5-002, M5-003: Canary Waves - Route low-risk and core tooling to async path
        canary_phases = ["mcp_discovery", "mcp_health", "code_search", "investigation"]
        if name in canary_phases and self.config.get("enable_new_orchestrator"):
            log_json("INFO", "orchestrator_canary_routing", details={"phase": name})
            try:
                import anyio
                import asyncio

                req = TaskRequest(task_id=f"canary_{uuid.uuid4().hex[:8]}", agent_name=name, input_data=input_data, context=ExecutionContext(project_root=str(self.project_root)))

                async def call_dispatch():
                    return await self._dispatch_task(req)

                try:
                    asyncio.get_running_loop()
                    task_res = anyio.from_thread.run(call_dispatch)
                except RuntimeError:
                    task_res = anyio.run(call_dispatch)

                if task_res.status == "success":
                    return task_res.output
                log_json("ERROR", "orchestrator_canary_failed", details={"phase": name, "error": task_res.error})
            except Exception as e:
                log_json("ERROR", "orchestrator_canary_exception", details={"phase": name, "error": str(e)})

        agent = self.agents.get(name)
        if not agent:
            return {}

        result = agent.run(input_data)

        # Shadow mode comparison (M4-005, M5-005)
        if self.config.get("new_orchestrator_shadow_mode"):
            self._run_shadow_check(name, input_data, result)

        # Post-phase hooks (observational)
        self.hook_engine.run_post_hooks(name, result if isinstance(result, dict) else {})

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

    def _restore_applied_changes(self, snapshots: List[Dict]) -> None:
        """Restore only the files mutated by the current loop attempt.

        This avoids touching unrelated user changes elsewhere in the repo.
        """
        restored: list[str] = []
        failed: list[Dict[str, str]] = []

        for snapshot in reversed(snapshots):
            target = Path(snapshot["target"])
            try:
                if snapshot.get("existed"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(snapshot.get("content") or "", encoding="utf-8")
                    if snapshot.get("mode") is not None:
                        os.chmod(target, int(snapshot["mode"]))
                else:
                    if target.exists():
                        target.unlink()
                restored.append(snapshot["file"])
            except Exception as exc:
                failed.append({"file": snapshot["file"], "error": str(exc)})

        if restored:
            log_json("INFO", "verify_fail_restore_ok", details={"files": restored})
        if failed:
            log_json("WARN", "verify_fail_restore_failed", details={"failures": failed})

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

            * ``"applied"`` (``List[str]``) — file paths successfully written.
            * ``"failed"``  (``List[dict]``) — dicts with ``"file"`` and
              ``"error"`` keys for each path that could not be written.
            * ``"snapshots"`` (``List[dict]``) — pre-apply file snapshots for
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

    def _route_failure(self, verification: Dict) -> str:
        """Classify a verification failure and return the recommended re-entry point.

        Delegates to :class:`~core.failure_router.FailureRouter` and maps the
        returned :class:`~core.failure_router.FailureAction` back to the legacy
        string literals expected by the rest of the orchestrator loop.

        Args:
            verification: The dict returned by the ``"verify"`` phase.  The
                keys ``"failures"`` (list) and ``"logs"`` (str) are inspected;
                all other keys are ignored.

        Returns:
            One of three string literals:

            * ``"act"``  — recoverable code-level error; retry the act phase.
            * ``"plan"`` — structural or design error; re-plan from scratch.
            * ``"skip"`` — external / environment issue that cannot be
              self-fixed (e.g. missing dependency, network error).
        """
        failures = " ".join(str(f) for f in verification.get("failures", []))
        logs = str(verification.get("logs", ""))
        error = (failures + " " + logs).strip()
        verification_output = logs or None

        action = self._failure_router.route_failure(
            phase="verify",
            attempt=1,
            error=error,
            verification_output=verification_output,
        )

        _action_to_route = {
            FailureAction.RETRY_ACT: "act",
            FailureAction.REPLAN: "plan",
            FailureAction.SKIP: "skip",
            FailureAction.ABORT: "skip",
        }
        return _action_to_route.get(action, "act")

    def _normalize_verification_result(self, verification: Dict) -> Dict:
        """Accept both legacy ``passed`` and canonical ``status`` verification payloads."""
        if not isinstance(verification, dict):
            return {"status": "fail", "failures": ["invalid verification payload"], "logs": str(verification)}
        if verification.get("status") in ("pass", "fail", "skip"):
            return verification
        if "passed" in verification:
            normalized = dict(verification)
            normalized["status"] = "pass" if bool(verification.get("passed")) else "fail"
            normalized.setdefault("failures", [])
            normalized.setdefault("logs", "")
            return normalized
        return verification

    # ── Inner loop helpers ────────────────────────────────────────────────────

    def _run_sandbox_loop(
        self,
        goal: str,
        act: Dict,
        task_bundle: Dict,
        dry_run: bool,
        phase_outputs: Dict,
    ):
        """Run the sandbox pre-apply check, retrying up to MAX_SANDBOX_RETRIES.

        On each failure, injects stderr as a fix_hint and re-generates code.

        Returns:
            Tuple of (final_act_dict, sandbox_passed, act_attempt_delta).
        """
        sandbox_passed = False
        sandbox_result = {}
        act_attempts_used = 0

        for _sandbox_try in range(MAX_SANDBOX_RETRIES):
            self._notify_ui("on_phase_start", "sandbox")
            t0_sandbox = time.time()
            sandbox_result = (
                self._run_phase(
                    "sandbox",
                    {
                        "act": act,
                        "dry_run": dry_run,
                        "project_root": str(self.project_root),
                    },
                )
                or {}
            )
            self._notify_ui("on_phase_complete", "sandbox", (time.time() - t0_sandbox) * 1000)

            phase_outputs["sandbox"] = sandbox_result
            sandbox_passed = sandbox_result.get("passed", True)
            if sandbox_passed or dry_run:
                break

            stderr_hint = (sandbox_result.get("details") or {}).get("stderr", "") or sandbox_result.get("summary", "sandbox_failed")
            failure_context = {"goal": goal, "phase": "sandbox"}
            analysis_suggestion = self._analyze_error(stderr_hint, failure_context)
            root_cause_analysis = self._run_root_cause_analysis(
                [stderr_hint],
                stderr_hint,
                failure_context,
                history=self._failure_history(),
            )
            investigation = self._run_investigation(
                goal=goal,
                verification={"failures": [stderr_hint], "logs": stderr_hint},
                context=failure_context,
                route="act",
                analysis_suggestion=analysis_suggestion,
                root_cause_analysis=root_cause_analysis,
            )
            failure_investigation = (investigation or {}).get("verification_investigation", {})
            remediation_plan = (investigation or {}).get("remediation_plan", {})
            fix_hints = remediation_plan.get("fix_hints", []) or [stderr_hint]
            if root_cause_analysis:
                sandbox_result["root_cause_analysis"] = root_cause_analysis
            sandbox_result["failure_investigation"] = failure_investigation
            sandbox_result["remediation_plan"] = remediation_plan
            if investigation:
                sandbox_result["investigation"] = investigation

            log_json(
                "WARN",
                "sandbox_pre_apply_failed",
                details={"try": _sandbox_try + 1, "summary": sandbox_result.get("summary", ""), "suggestion": analysis_suggestion, "root_cause_patterns": (root_cause_analysis or {}).get("patterns", []), "investigation_signals": failure_investigation.get("signals", [])},
            )

            if _sandbox_try < MAX_SANDBOX_RETRIES - 1:
                phase_outputs["retry_count"] = phase_outputs.get("retry_count", 0) + 1
                task_bundle["fix_hints"] = fix_hints
                act = self._run_phase(
                    "act",
                    {
                        "task": goal,
                        "task_bundle": task_bundle,
                        "dry_run": dry_run,
                        "project_root": str(self.project_root),
                        "fix_hints": fix_hints,
                    },
                )
                act_attempts_used += 1
                phase_outputs["change_set"] = act
            else:
                log_json("WARN", "sandbox_max_retries_exceeded", details={"max": MAX_SANDBOX_RETRIES, "continuing_with_best_attempt": True})

        if not sandbox_passed and not dry_run:
            task_bundle["fix_hints"] = [(sandbox_result.get("details") or {}).get("stderr", "") or sandbox_result.get("summary", "sandbox_failed")]

        return act, sandbox_passed, act_attempts_used

    def _select_act_agent(self, goal: str) -> str:
        """Return the agents-dict key for the best code-generation agent for *goal*.

        Uses resolve_agent_for_goal() when a model_adapter is available and
        the resolved agent is actually registered in self.agents; falls back
        to "act" otherwise so existing behaviour is preserved.
        """
        try:
            from core.skill_dispatcher import resolve_agent_for_goal

            model_adapter = getattr(self, "model_adapter", None) or getattr(self, "model", None)
            spec = resolve_agent_for_goal(goal, model_adapter=model_adapter)
            if spec and spec.name in self.agents:
                return spec.name
        except Exception:
            pass
        return "act"

    def _execute_act_verify_attempt(self, goal: str, plan: Dict, task_bundle: Dict, cycle_id: str, phase_outputs: Dict, dry_run: bool):
        """Execute one attempt of act -> sandbox -> apply -> verify.

        When ``n_best_candidates`` > 1 in config, generates multiple code
        variants and uses critic tournament to select the best one.
        """
        config = self._load_config_file()
        n_best = config.get("n_best_candidates", 1)

        self._notify_ui("on_phase_start", "act")
        t0_act = time.time()

        # RAG: retrieve similar past implementations as context
        rag_context = None
        try:
            from core.code_rag import CodeRAG

            code_rag = CodeRAG(
                vector_store=getattr(self, "vector_store", None),
                brain=self.brain,
            )
            rag_context = code_rag.retrieve_context(goal, task_bundle)
            if rag_context and rag_context.examples:
                task_bundle = dict(task_bundle) if isinstance(task_bundle, dict) else {}
                task_bundle["rag_examples"] = [e.get("content", "")[:300] for e in rag_context.examples[:3]]
                task_bundle["rag_anti_patterns"] = rag_context.anti_patterns[:3]
        except Exception:
            pass

        if n_best > 1 and self.model and not dry_run:
            # N-Best code generation with critic tournament
            from core.nbest import NBestEngine

            engine = NBestEngine(
                n_candidates=n_best,
                temperature_spread=tuple(config.get("n_best_temperature_spread", [0.2, 0.5, 0.8])),
            )
            act_prompt = f"Goal: {goal}\nTask bundle: {json.dumps(task_bundle, default=str)[:3000]}"
            candidates = engine.generate_candidates(self.model, act_prompt)
            sandbox_agent = self.agents.get("sandbox")
            if sandbox_agent:
                candidates = engine.sandbox_all(sandbox_agent, candidates)
            try:
                winner = engine.critic_tournament(self.model, candidates, goal)
                act = {"changes": winner.changes, "confidence": winner.total_score, "variant_id": winner.variant_id, "n_best": True}
            except ValueError:
                # Fallback to single-path if no valid candidates
                act = self._run_phase(
                    self._select_act_agent(goal),
                    {
                        "task": goal,
                        "task_bundle": task_bundle,
                        "dry_run": dry_run,
                        "project_root": str(self.project_root),
                        "fix_hints": task_bundle.get("fix_hints", []),
                    },
                )
        else:
            act = self._run_phase(
                self._select_act_agent(goal),
                {
                    "task": goal,
                    "task_bundle": task_bundle,
                    "dry_run": dry_run,
                    "project_root": str(self.project_root),
                    "fix_hints": task_bundle.get("fix_hints", []),
                },
            )

        self._notify_ui("on_phase_complete", "act", (time.time() - t0_act) * 1000)

        # Record act confidence
        act_confidence = self._estimate_confidence(act, "act")
        act_result_pr = PhaseResult(phase="act", output=act, confidence=act_confidence)
        self.confidence_router.record(act_result_pr)
        phase_outputs["act_confidence"] = act_confidence

        if validate_phase_output("change_set", act) and self.debugger:
            debug_hint = self.debugger.diagnose(
                error_message="CoderAgent produced invalid change_set",
                current_goal=goal,
                context=json.dumps(
                    {"goal": goal, "plan": plan, "task_bundle": task_bundle},
                    default=str,
                ),
            )
            act = self._run_phase(self._select_act_agent(goal), {"task": goal, "task_bundle": task_bundle, "dry_run": dry_run, "project_root": str(self.project_root), "debug_hint": debug_hint})

        phase_outputs["change_set"] = act
        act, _passed, extra_uses = self._run_sandbox_loop(goal, act, task_bundle, dry_run, phase_outputs)

        self._notify_ui("on_phase_start", "apply")
        t0_apply = time.time()
        apply_result = self._apply_change_set(act, dry_run=dry_run)
        self._notify_ui("on_phase_complete", "apply", (time.time() - t0_apply) * 1000)
        phase_outputs["apply_result"] = apply_result

        tests = task_bundle.get("tasks", [{}])[0].get("tests", []) if isinstance(task_bundle, dict) else []
        self._notify_ui("on_phase_start", "verify")
        t0_verify = time.time()
        if apply_result.get("failed"):
            verification = {
                "status": "fail",
                "failures": [f"{item['file']}: {item['error']}" for item in apply_result["failed"]],
                "logs": "\n".join(str(item.get("error", "")) for item in apply_result["failed"]),
            }
        else:
            verification = self._run_phase("verify", {"change_set": act, "dry_run": dry_run, "project_root": str(self.project_root), "tests": tests})
        verification = self._normalize_verification_result(verification)
        self._notify_ui("on_phase_complete", "verify", (time.time() - t0_verify) * 1000, success=(verification.get("status") in ("pass", "skip")))

        phase_outputs["verification"] = verification
        return act, apply_result, verification, extra_uses

    def _run_act_loop(self, goal: str, plan: Dict, task_bundle: Dict, pipeline_cfg, cycle_id: str, phase_outputs: Dict, dry_run: bool, plan_attempt: int, max_plan_retries: int, skill_context: Dict):
        max_act_attempts = pipeline_cfg.max_act_attempts
        act_attempt = 0
        verification: Dict = {}
        replan_needed = False

        while act_attempt < max_act_attempts:
            act_attempt += 1
            if act_attempt > 1:
                phase_outputs["retry_count"] = phase_outputs.get("retry_count", 0) + 1
                time.sleep(min(2 ** (act_attempt - 2), 16))

            act, apply_result, verification, extra_uses = self._execute_act_verify_attempt(goal, plan, task_bundle, cycle_id, phase_outputs, dry_run)
            act_attempt += extra_uses

            if verification.get("status") in ("pass", "skip"):
                blocked, gate_reason = self.human_gate.should_block(verification, skill_context)
                if blocked:
                    if not self.human_gate.request_approval(gate_reason, {"cycle_id": cycle_id, "goal": goal, "changes": len(act.get("changes", []))}):
                        phase_outputs["human_gate"] = {"blocked": True, "reason": gate_reason, "approved": False}
                        break
                    phase_outputs["human_gate"] = {"blocked": True, "reason": gate_reason, "approved": True}
                break

            if apply_result.get("applied") and not dry_run:
                self._restore_applied_changes(apply_result.get("snapshots", []))

            route = self._route_failure(verification)
            failure_context = {"goal": goal, "phase": "verify", "route": route}
            failure_logs = verification.get("logs", "")
            failure_text = "\n".join(verification.get("failures", []))
            analysis_suggestion = self._analyze_error(failure_text, failure_context)
            root_cause_analysis = self._run_root_cause_analysis(
                verification.get("failures", []),
                failure_logs,
                failure_context,
                history=self._failure_history(),
            )
            current_test_count = None
            previous_test_count = None
            quality = phase_outputs.get("quality", {}) if isinstance(phase_outputs.get("quality"), dict) else {}
            if "test_count" in quality:
                current_test_count = quality.get("test_count")
            recent_history = self._failure_history(limit=2)
            if recent_history:
                latest_quality = recent_history[-1].get("phase_outputs", {}).get("quality", {}) if isinstance(recent_history[-1], dict) else {}
                if isinstance(latest_quality, dict) and "test_count" in latest_quality:
                    previous_test_count = latest_quality.get("test_count")
            investigation = self._run_investigation(
                goal=goal,
                verification=verification,
                context=failure_context,
                route=route,
                analysis_suggestion=analysis_suggestion,
                root_cause_analysis=root_cause_analysis,
                previous_test_count=previous_test_count,
                current_test_count=current_test_count,
            )
            failure_investigation = (investigation or {}).get("verification_investigation", {})
            remediation_plan = (investigation or {}).get("remediation_plan", {})
            fix_hints = remediation_plan.get("fix_hints", []) or verification.get("failures", [])
            verification["failure_investigation"] = failure_investigation
            verification["remediation_plan"] = remediation_plan
            if investigation:
                verification["investigation"] = investigation
            if root_cause_analysis:
                verification["root_cause_analysis"] = root_cause_analysis

            if route == "plan" and plan_attempt < max_plan_retries:
                phase_outputs["retry_count"] = phase_outputs.get("retry_count", 0) + 1
                phase_outputs["_failure_context"] = {
                    "failures": verification.get("failures", []),
                    "logs": verification.get("logs", ""),
                    "route": "plan",
                    "suggestion": analysis_suggestion,
                    "failure_investigation": failure_investigation,
                    "root_cause_analysis": root_cause_analysis,
                    "remediation_plan": remediation_plan,
                    "investigation": investigation,
                }
                replan_needed = True
                break
            elif route == "skip":
                break
            task_bundle["fix_hints"] = fix_hints

        return verification, replan_needed, None

    def _execute_plan_critique_synthesize(self, goal: str, context: Dict, skill_context: Dict, pipeline_cfg: Any, phase_outputs: Dict) -> Tuple[Dict, Dict]:
        """Section 3-5: PLAN -> CRITIQUE -> SYNTHESIZE.

        When tree_of_thought_candidates > 1 in config, generates N plan
        candidates with varied strategies and selects the best one.
        """
        beads_decision = phase_outputs.get("beads_gate", {})
        config = self._load_config_file()
        tot_candidates = config.get("tree_of_thought_candidates", 1)

        self._notify_ui("on_phase_start", "plan")
        t0 = time.time()

        # Tree-of-Thought: generate N plans with varied strategies
        if tot_candidates > 1 and self.model:
            try:
                from core.tree_of_thought import TreeOfThoughtPlanner

                tot = TreeOfThoughtPlanner(n_candidates=tot_candidates)
                candidates = tot.generate_plans(
                    self.model,
                    goal,
                    {
                        "memory_snapshot": context.get("memory_summary", ""),
                        "known_weaknesses": "",
                        "skill_context": skill_context,
                    },
                )
                winner = tot.score_plans(self.model, candidates, goal)
                plan = {"steps": winner.steps, "strategy": winner.strategy, "confidence": winner.total_score, "tree_of_thought": True}
                self._notify_ui("on_phase_complete", "plan", (time.time() - t0) * 1000)
                phase_outputs["plan"] = plan
                phase_outputs["tot_strategy"] = winner.strategy

                # Record high confidence — may skip critique
                plan_confidence = min(winner.total_score, 0.95)
                plan_result = PhaseResult(phase="plan", output=plan, confidence=plan_confidence)
                self.confidence_router.record(plan_result)
                phase_outputs["plan_confidence"] = plan_confidence

                # Synthesize directly (ToT already self-scored)
                self._notify_ui("on_phase_start", "synthesize")
                t0 = time.time()
                task_bundle = self._run_phase(
                    "synthesize",
                    {
                        "goal": goal,
                        "plan": plan,
                        "critique": {"status": "skipped", "reason": "tree_of_thought_self_scored"},
                        "beads_decision": beads_decision,
                    },
                )
                self._notify_ui("on_phase_complete", "synthesize", (time.time() - t0) * 1000)
                phase_outputs["task_bundle"] = task_bundle
                return plan, task_bundle
            except Exception as exc:
                log_json("WARN", "tree_of_thought_failed_fallback", details={"error": str(exc)})
                # Fall through to standard planning

        plan = self._run_phase(
            "plan",
            {
                "goal": goal,
                "memory_snapshot": context.get("memory_summary", ""),
                "similar_past_problems": context.get("hints_summary", ""),
                "known_weaknesses": "",
                "skill_context": skill_context,
                "failure_context": phase_outputs.get("_failure_context", {}),
                "extra_context": getattr(pipeline_cfg, "extra_plan_ctx", {}),
                "backfill_context": context.get("backfill_context", []),
                "beads_decision": beads_decision,
                "beads_constraints": beads_decision.get("required_constraints", []),
                "beads_required_tests": beads_decision.get("required_tests", []),
                "beads_required_skills": beads_decision.get("required_skills", []),
            },
        )
        self._notify_ui("on_phase_complete", "plan", (time.time() - t0) * 1000)
        phase_outputs["plan"] = plan

        # Record plan confidence and check routing
        plan_confidence = self._estimate_confidence(plan, "plan")
        plan_result = PhaseResult(phase="plan", output=plan, confidence=plan_confidence)
        self.confidence_router.record(plan_result)
        phase_outputs["plan_confidence"] = plan_confidence

        # Skip critique if plan confidence is very high
        if self.confidence_router.should_skip_optional(plan_result, "critique"):
            log_json("INFO", "critique_skipped_high_confidence", details={"plan_confidence": plan_confidence})
            critique = {"status": "skipped", "reason": "high_plan_confidence"}
            phase_outputs["critique"] = critique
            self._notify_ui("on_phase_start", "synthesize")
            t0 = time.time()
            task_bundle = self._run_phase(
                "synthesize",
                {
                    "goal": goal,
                    "plan": plan,
                    "critique": critique,
                    "beads_decision": beads_decision,
                },
            )
            self._notify_ui("on_phase_complete", "synthesize", (time.time() - t0) * 1000)
            phase_outputs["task_bundle"] = task_bundle
            return plan, task_bundle

        self._notify_ui("on_phase_start", "critique")
        t0 = time.time()
        critique = self._run_phase(
            "critique",
            {
                "task": goal,
                "plan": plan.get("steps", []),
            },
        )
        self._notify_ui("on_phase_complete", "critique", (time.time() - t0) * 1000)
        phase_outputs["critique"] = critique

        self._notify_ui("on_phase_start", "synthesize")
        t0 = time.time()
        task_bundle = self._run_phase(
            "synthesize",
            {
                "goal": goal,
                "plan": plan,
                "critique": critique,
                "beads_decision": beads_decision,
            },
        )
        self._notify_ui("on_phase_complete", "synthesize", (time.time() - t0) * 1000)
        phase_outputs["task_bundle"] = task_bundle
        return plan, task_bundle

    def _run_plan_loop(self, goal: str, context: Dict, skill_context: Dict, pipeline_cfg: Any, cycle_id: str, phase_outputs: Dict, dry_run: bool) -> Tuple[Dict, Optional[Dict]]:
        max_plan_retries = getattr(pipeline_cfg, "plan_retries", 3)
        plan_attempt = 0
        verification: Dict = {}

        while plan_attempt < max_plan_retries:
            plan_attempt += 1
            plan, task_bundle = self._execute_plan_critique_synthesize(goal, context, skill_context, pipeline_cfg, phase_outputs)

            verification, replan_needed, early_return = self._run_act_loop(
                goal=goal, plan=plan, task_bundle=task_bundle, pipeline_cfg=pipeline_cfg, cycle_id=cycle_id, phase_outputs=phase_outputs, dry_run=dry_run, plan_attempt=plan_attempt, max_plan_retries=max_plan_retries, skill_context=skill_context
            )
            if early_return:
                return verification, early_return
            if not replan_needed:
                break

        return verification, None

    def _configure_pipeline(self, goal: str, goal_type: str, phase_outputs: Dict) -> Any:
        """Section 0: ADAPTIVE PIPELINE CONFIG."""
        if self.adaptive_pipeline:
            pipeline_cfg = self.adaptive_pipeline.configure(
                goal,
                goal_type,
                consecutive_fails=self._consecutive_fails,
                past_failures=list(phase_outputs.get("_failure_context", {}).get("failures", [])),
            )
        else:
            from core.adaptive_pipeline import AdaptivePipeline

            pipeline_cfg = AdaptivePipeline()._default_config(goal_type)

        phase_outputs["pipeline_config"] = {
            "intensity": pipeline_cfg.intensity,
            "phases": pipeline_cfg.phases,
            "skills": pipeline_cfg.skill_set,
        }
        self._notify_ui("on_pipeline_configured", phase_outputs["pipeline_config"])
        return pipeline_cfg

    def _handle_capabilities(self, goal: str, pipeline_cfg: Any, phase_outputs: Dict, dry_run: bool):
        """Section 0.1: CAPABILITY MANAGEMENT."""
        capability_plan = {"matched_capabilities": [], "recommended_skills": [], "missing_skills": [], "mcp_tools": [], "provisioning_actions": []}
        capability_goal_queue = {"attempted": False, "queued": [], "skipped": [], "queue_strategy": None}
        capability_provisioning = {"attempted": False, "results": []}

        if self.auto_add_capabilities:
            capability_plan = analyze_capability_needs(goal, available_skills=self.skills.keys(), active_skills=pipeline_cfg.skill_set)
            if capability_plan["recommended_skills"]:
                pipeline_cfg.skill_set = list(dict.fromkeys(list(pipeline_cfg.skill_set) + list(capability_plan["recommended_skills"])))
                phase_outputs["pipeline_config"]["skills"] = pipeline_cfg.skill_set
            phase_outputs["capability_plan"] = capability_plan
            capability_goal_queue = queue_missing_capability_goals(goal_queue=self.goal_queue, missing_skills=capability_plan["missing_skills"], goal=goal, enabled=self.auto_queue_missing_capabilities, dry_run=dry_run)
            if capability_goal_queue["queued"] or capability_goal_queue["skipped"]:
                phase_outputs["capability_goal_queue"] = capability_goal_queue
            if capability_plan["provisioning_actions"]:
                capability_provisioning = provision_capability_actions(project_root=self.project_root, provisioning_actions=capability_plan["provisioning_actions"], auto_provision=self.auto_provision_mcp, start_servers=self.auto_start_mcp_servers, dry_run=dry_run)
                phase_outputs["capability_provisioning"] = capability_provisioning

        self.last_capability_plan = capability_plan
        self.last_capability_goal_queue = capability_goal_queue
        self.last_capability_provisioning = capability_provisioning
        self.last_capability_status = record_capability_status(project_root=self.project_root, goal=goal, capability_plan=capability_plan, capability_goal_queue=capability_goal_queue, capability_provisioning=capability_provisioning, goal_queue=self.goal_queue)

    def _run_ingest_phase(self, goal: str, cycle_id: str, phase_outputs: Dict) -> Dict:
        """Section 1: INGEST."""
        self._notify_ui("on_phase_start", "ingest")
        t0 = time.time()
        working_memory = self.memory_controller.retrieve(MemoryTier.WORKING)
        session_memory = self.memory_controller.retrieve(MemoryTier.SESSION)
        ingest_input = {"goal": goal, "project_root": str(self.project_root), "hints": self._retrieve_hints(goal), "working_memory": working_memory, "session_memory": session_memory}
        # SADD context injection: merge dependency context from completed workstreams
        ctx_inject = phase_outputs.get("context_injection")
        if ctx_inject:
            ingest_input["dependency_context"] = ctx_inject
        context = self._run_phase("ingest", ingest_input)
        self._notify_ui("on_phase_complete", "ingest", (time.time() - t0) * 1000)
        if "bundle" in context:
            self._notify_ui("on_context_assembled", context["bundle"])

        errors = validate_phase_output("context", context)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "context", "errors": errors})
        phase_outputs["context"] = context
        return context

    def _run_mcp_discovery_phase(self, phase_outputs: Dict) -> Dict:
        """Section 1.5: MCP DISCOVERY."""
        self._notify_ui("on_phase_start", "mcp_discovery")
        t0 = time.time()
        mcp_discovery_output = self._run_phase("mcp_discovery", {"project_root": str(self.project_root)})
        self._notify_ui("on_phase_complete", "mcp_discovery", (time.time() - t0) * 1000)
        phase_outputs["mcp_discovery"] = mcp_discovery_output
        return mcp_discovery_output

    def _dispatch_skills(self, goal_type: str, pipeline_cfg: Any, phase_outputs: Dict) -> Dict:
        """Section 2: SKILL DISPATCH."""
        self._notify_ui("on_phase_start", "skill_dispatch")
        t0 = time.time()
        skill_context: Dict = {}
        if self.skills and pipeline_cfg.skill_set:
            active_skills = {k: self.skills[k] for k in pipeline_cfg.skill_set if k in self.skills}
            # Check skill correlation for suggested additions
            if self.skill_correlation:
                try:
                    base_skill_names = list(active_skills.keys())
                    suggestions = self.skill_correlation.suggest_skills(base_skill_names, goal_type)
                    for suggested_name, corr_score in suggestions:
                        if suggested_name in self.skills and suggested_name not in active_skills:
                            active_skills[suggested_name] = self.skills[suggested_name]
                            log_json("INFO", "skill_correlation_added", details={"skill": suggested_name, "correlation": round(corr_score, 3)})
                except Exception:
                    pass

            skill_context = dispatch_skills(goal_type, active_skills, str(self.project_root))
        phase_outputs["skill_context"] = skill_context
        self._notify_ui("on_phase_complete", "skill_dispatch", (time.monotonic() - t0) * 1000)
        return skill_context

    def _beads_gate_applies(self) -> bool:
        if not self.beads_enabled or self.beads_bridge is None:
            return False
        if self.beads_scope == "all_runtime":
            return True
        return self.beads_scope == "goal_run"

    def _run_beads_gate(
        self,
        goal: str,
        goal_type: str,
        context: Dict,
        skill_context: Dict,
    ) -> Dict[str, Any]:
        active_context = {
            "context": context,
            "skill_context": skill_context,
            "capability_plan": self.last_capability_plan,
            "capability_goal_queue": self.last_capability_goal_queue,
            "capability_provisioning": self.last_capability_provisioning,
        }
        payload = build_beads_runtime_input(
            goal=goal,
            goal_type=goal_type,
            project_root=self.project_root,
            runtime_mode=self.runtime_mode,
            goal_queue=self.goal_queue,
            goal_archive=self.goal_archive,
            active_goal=goal,
            active_context=active_context,
        )
        log_json("INFO", "beads_gate_start", details={"goal": goal, "goal_type": goal_type, "scope": self.beads_scope})
        result = self.beads_bridge.run(payload)
        decision = result.get("decision") or {}
        beads_state = {
            "ok": bool(result.get("ok")),
            "status": decision.get("status") if decision else ("error" if not result.get("ok") else None),
            "decision_id": decision.get("decision_id"),
            "summary": decision.get("summary"),
            "required_constraints": list(decision.get("required_constraints", [])) if isinstance(decision, dict) else [],
            "required_skills": list(decision.get("required_skills", [])) if isinstance(decision, dict) else [],
            "required_tests": list(decision.get("required_tests", [])) if isinstance(decision, dict) else [],
            "follow_up_goals": list(decision.get("follow_up_goals", [])) if isinstance(decision, dict) else [],
            "stop_reason": decision.get("stop_reason") if isinstance(decision, dict) else None,
            "error": result.get("error"),
            "stderr": result.get("stderr"),
            "duration_ms": result.get("duration_ms", 0),
        }
        log_json(
            "INFO" if beads_state["ok"] else "WARN",
            "beads_gate_complete",
            details={
                "goal": goal,
                "ok": beads_state["ok"],
                "status": beads_state["status"],
                "decision_id": beads_state["decision_id"],
                "error": beads_state["error"],
            },
        )
        return beads_state

    def _build_early_stop_entry(
        self,
        *,
        cycle_id: str,
        goal: str,
        goal_type: str,
        phase_outputs: Dict,
        started_at: float,
        stop_reason: str,
        beads: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        entry = {
            "cycle_id": cycle_id,
            "goal": goal,
            "goal_type": goal_type,
            "phase_outputs": phase_outputs,
            "stop_reason": stop_reason,
            "started_at": started_at,
            "completed_at": time.time(),
        }
        if beads is not None:
            entry["beads"] = beads
        self._refresh_cycle_summary(entry)
        self.current_goal = None
        self.active_cycle_summary = None
        return entry

    def _run_reflection_phase(self, verification: Dict, skill_context: Dict, goal_type: str, cycle_id: str, phase_outputs: Dict) -> Dict:
        """Section 7: REFLECT."""
        self._notify_ui("on_phase_start", "reflect")
        t0 = time.time()
        reflection = self._run_phase("reflect", {"verification": verification, "skill_context": skill_context, "goal_type": goal_type})
        self._notify_ui("on_phase_complete", "reflect", (time.time() - t0) * 1000)
        errors = validate_phase_output("reflection", reflection)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "reflection", "errors": errors})
        phase_outputs["reflection"] = reflection
        if reflection.get("summary"):
            self.memory_controller.store(MemoryTier.SESSION, reflection["summary"], metadata={"cycle_id": cycle_id, "type": "reflection"})
        return reflection

    def _refresh_cycle_summary(self, entry: Dict, *, state: str = "complete", current_phase: str | None = None) -> Dict:
        """Rebuild and cache the canonical operator-facing cycle summary."""
        summary = build_cycle_summary(entry, state=state, current_phase=current_phase)
        entry["cycle_summary"] = summary
        if state == "running":
            self.active_cycle_summary = summary
        else:
            self.last_cycle_summary = summary
        return summary

    def _record_cycle_outcome(self, cycle_id: str, goal: str, goal_type: str, phase_outputs: Dict, started_at: float):
        """Final persistence and loop notification."""
        from core.quality_snapshot import run_quality_snapshot

        verify_status = phase_outputs.get("verification", {}).get("status", "skip")
        passed = verify_status in ("pass", "skip")
        if passed:
            self._consecutive_fails = 0
        elif verify_status == "fail":
            self._consecutive_fails += 1

        # Phase 8: measure()
        self._notify_ui("on_phase_start", "measure")
        t0_measure = time.time()
        changed_files = phase_outputs.get("apply_result", {}).get("applied", [])
        quality = run_quality_snapshot(self.project_root, changed_files=changed_files)
        phase_outputs["quality"] = quality
        self._notify_ui("on_phase_complete", "measure", (time.time() - t0_measure) * 1000)

        # ── Quality Trend Analysis ──
        try:
            alerts = self.quality_trends.record_from_cycle(
                {
                    "cycle_id": cycle_id,
                    "goal": goal,
                    "completed_at": time.time(),
                    "duration_s": time.time() - started_at,
                    "phase_outputs": phase_outputs,
                }
            )
            if alerts and self.goal_queue:
                for goal_text in self.quality_trends.get_remediation_goals():
                    self.goal_queue.add(goal_text)
                    log_json("INFO", "quality_remediation_goal_enqueued", details={"goal": goal_text[:100]})
        except Exception as exc:
            log_json("WARN", "quality_trend_record_failed", details={"error": str(exc)})

        # ── Learning Loop: CycleOutcome ──
        outcome = CycleOutcome(
            goal=goal,
            goal_type=goal_type,
            started_at=started_at,
            phases_completed=list(phase_outputs.keys()),
            changes_applied=len(changed_files),
            tests_after=quality.get("test_count", 0),
            strategy_used=phase_outputs.get("pipeline_config", {}).get("intensity", "normal"),
        )

        if self.adaptive_pipeline:
            try:
                strategy = outcome.strategy_used
                self.adaptive_pipeline.record_outcome(goal_type, strategy, passed)
            except Exception as exc:
                log_json("WARN", "adaptive_pipeline_outcome_record_failed", details={"error": str(exc)})

        completed_at = time.time()
        outcome.mark_complete(success=passed)
        entry = {
            "cycle_id": cycle_id,
            "goal": goal,
            "goal_type": goal_type,
            "phase_outputs": phase_outputs,
            "dry_run": bool(phase_outputs.get("dry_run")),
            "beads": phase_outputs.get("beads_gate"),
            "stop_reason": None,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_s": outcome.duration_s(),
            "outcome": dataclasses.asdict(outcome),
        }

        # Phase 9: learn()
        self._notify_ui("on_phase_start", "learn")
        t0_learn = time.time()
        summary = self._refresh_cycle_summary(entry)
        self._notify_ui("on_cycle_complete", summary)
        self.current_goal = None
        self.active_cycle_summary = None
        if self.memory_controller.persistent_store:
            self.memory_controller.persistent_store.append_log(entry)
            # Store structured outcome for learning
            self.memory_controller.store(MemoryTier.PROJECT, json.dumps(summary), metadata={"type": "cycle_summary", "goal": goal, "cycle_id": cycle_id})

        if self.brain:
            try:
                self.brain.set(f"outcome:{cycle_id}", outcome.to_json())
                self.brain.remember(f"Cycle completed: {goal} -> {'SUCCESS' if passed else 'FAILED'}")
            except Exception as exc:
                log_json("WARN", "brain_outcome_storage_failed", details={"error": str(exc)})

        self._notify_ui("on_phase_complete", "learn", (time.time() - t0_learn) * 1000)

        if self.context_graph is not None:
            try:
                self.context_graph.update_from_cycle(entry)
            except Exception as exc:
                log_json("WARN", "context_graph_update_failed", details={"error": str(exc)})

        for loop in self._improvement_loops:
            try:
                loop.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "improvement_loop_error", details={"loop": type(loop).__name__, "error": str(exc)})

        # Phase 10: discover()
        self._notify_ui("on_phase_start", "discover")
        # Discovery now happens via on_cycle_complete (TRIGGER_EVERY_N=15)
        self._notify_ui("on_phase_complete", "discover", 0)

        # Phase 11: evolve()
        self._notify_ui("on_phase_start", "evolve")
        # Evolution now happens via on_cycle_complete (TRIGGER_EVERY_N=20)
        self._notify_ui("on_phase_complete", "evolve", 0)

        if self.propagation_engine is not None:
            try:
                self.propagation_engine.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "propagation_engine_error", details={"error": str(exc)})

        # ── Memory Consolidation (periodic) ──
        # Run every 10 cycles to prune, merge, and summarize old memories
        cycle_num = int(cycle_id.split("_")[-1], 16) if "_" in cycle_id else 0
        if self.brain and cycle_num % 10 == 0:
            try:
                from memory.consolidation import MemoryConsolidator, MemoryEntry

                consolidator = MemoryConsolidator()
                # Convert brain memories to MemoryEntry format
                raw_memories = self.brain.recall_with_budget(max_tokens=50000)
                entries = [MemoryEntry(id=str(i), content=m, memory_type="decision") for i, m in enumerate(raw_memories)]
                if len(entries) > 50:
                    retained, result = consolidator.consolidate(entries)
                    log_json("INFO", "memory_consolidation_complete", details={"before": result.memories_before, "after": result.memories_after, "compression": f"{result.compression_ratio:.1%}"})
            except Exception as exc:
                log_json("WARN", "memory_consolidation_error", details={"error": str(exc)})

        return entry

    def _parse_bead_id(self, goal: str) -> Optional[str]:
        """Extract bead ID from goal string if present (format: 'bead:ID: Title')."""
        if goal.startswith("bead:"):
            parts = goal.split(":", 2)
            if len(parts) >= 2:
                return parts[1]
        return None

    def _claim_bead(self, bead_id: str):
        """Mark a bead as in_progress using BeadsSkill."""
        beads_skill = self._get_beads_skill()
        if beads_skill is not None:
            log_json("INFO", "orchestrator_claiming_bead", details={"bead_id": bead_id})
            beads_skill.run({"cmd": "update", "id": bead_id, "args": ["--status", "in_progress"]})

    def _close_bead(self, bead_id: str, reason: str):
        """Close a bead using BeadsSkill."""
        beads_skill = self._get_beads_skill()
        if beads_skill is not None:
            log_json("INFO", "orchestrator_closing_bead", details={"bead_id": bead_id})
            beads_skill.run({"cmd": "close", "id": bead_id, "args": ["--reason", reason]})

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

        context = self._run_ingest_phase(goal, cycle_id, phase_outputs)
        if self.strict_schema and validate_phase_output("context", context):
            return self._build_early_stop_entry(
                cycle_id=cycle_id,
                goal=goal,
                goal_type=goal_type,
                phase_outputs=phase_outputs,
                started_at=started_at,
                stop_reason="INVALID_OUTPUT",
            )

        # Autonomous MCP capability injection
        mcp_discovery = self._run_mcp_discovery_phase(phase_outputs)
        if mcp_discovery.get("status") == "success" and mcp_discovery.get("discovered"):
            log_json("INFO", "mcp_servers_discovered", details={"count": len(mcp_discovery["discovered"])})

        skill_context = self._dispatch_skills(goal_type, pipeline_cfg, phase_outputs)

        if self._beads_gate_applies():
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
            if beads_gate.get("status") == "block":
                return self._build_early_stop_entry(
                    cycle_id=cycle_id,
                    goal=goal,
                    goal_type=goal_type,
                    phase_outputs=phase_outputs,
                    started_at=started_at,
                    stop_reason=beads_gate.get("stop_reason") or "BEADS_BLOCKED",
                    beads=beads_gate,
                )
            if beads_gate.get("status") == "revise":
                return self._build_early_stop_entry(
                    cycle_id=cycle_id,
                    goal=goal,
                    goal_type=goal_type,
                    phase_outputs=phase_outputs,
                    started_at=started_at,
                    stop_reason=beads_gate.get("stop_reason") or "BEADS_REVISE_REQUIRED",
                    beads=beads_gate,
                )

        # ── Coverage Backfill ──
        gaps = skill_context.get("structural_analyzer", {}).get("coverage_gaps", [])
        if getattr(self, "auto_backfill_coverage", False) and self.goal_queue:
            for gap in gaps:
                f = gap.get("file")
                priority = gap.get("risk_priority", "MEDIUM")
                backfill_goal = f"test_backfill: Write missing unit tests for '{f}' to resolve coverage gap (priority: {priority})"
                self.goal_queue.add(backfill_goal)
                log_json("INFO", "backfill_goal_enqueued", details={"file": f, "priority": priority})

        # Inject gaps into context for the planner
        if gaps:
            context["backfill_context"] = gaps

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
            return early_return

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

        # Record cycle confidence for telemetry
        phase_outputs["cycle_confidence"] = self.confidence_router.get_cycle_confidence()

        phase_outputs.pop("_failure_context", None)
        return self._record_cycle_outcome(cycle_id, goal, goal_type, phase_outputs, started_at)

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

            * ``"goal"`` (str) — the original goal string.
            * ``"stop_reason"`` (str) — why the loop terminated.  One of the
              policy-defined reasons, a cycle-level ``"INVALID_OUTPUT"``, or
              ``"MAX_CYCLES"`` when the hard limit was reached.
            * ``"history"`` (list[dict]) — list of cycle-result dicts in
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
            except Exception:
                pass

    async def shutdown(self):
        """Shut down the orchestrator and clean up resources (M7-003)."""
        await MCPAsyncClient.close_all()
        log_json("INFO", "orchestrator_shutdown_complete")
