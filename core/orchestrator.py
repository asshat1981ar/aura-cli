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
import heapq
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.logging_utils import log_json
from core.circuit_breaker import CircuitBreaker
from core.beads_bridge import build_beads_runtime_input
from core.fot_arbiter import FoTArbiter
from core.operator_runtime import build_cycle_summary
from core.cycle_outcome import CycleOutcome
from core.capability_coordinator import CapabilityCoordinator
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
from core.phases import build_phase_registry
from core.codex_manager import CodexManager
from core.git_tools import GitTools
from memory.controller import memory_controller, MemoryTier
from core.config_manager import config as _config

MAX_SANDBOX_RETRIES = _config.get("max_sandbox_retries", 3)
USE_CODEX_PARALLEL = _config.get("aura_use_codex_parallel", False) or os.environ.get("AURA_USE_CODEX_PARALLEL") == "1"


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
        memory_store: Any = None, # Deprecated in favor of global memory_controller
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
        ralph_enabled: bool = True,
        ralph_mode: str = "propose",
        ralph_max_proposals_per_cycle: int = 3,
        ralph_max_auto_queue_per_cycle: int = 2,
        discovery_loop: Any = None,
        evolution_loop: Any = None,
        fot_enabled: bool = True,
        fot_max_candidates_per_cycle: int | None = None,
        fot_max_auto_queue_per_cycle: int | None = None,
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
        self.debugger = debugger  # Optional DebuggerAgent for auto-recovery
        self.human_gate = HumanGate()
        self.capabilities = CapabilityCoordinator(
            auto_add=auto_add_capabilities,
            auto_queue_missing=auto_queue_missing_capabilities,
            auto_provision_mcp=auto_provision_mcp,
            auto_start_mcp_servers=auto_start_mcp_servers,
            project_root=self.project_root,
        )
        self.goal_queue = goal_queue
        self.goal_archive = goal_archive
        self.brain = brain
        self.model = model
        self.runtime_mode = runtime_mode
        self.beads_bridge = beads_bridge
        self.beads_enabled = beads_enabled
        self.beads_required = beads_required
        self.beads_scope = beads_scope
        self.ralph_enabled = bool(ralph_enabled)
        self.ralph_mode = str(ralph_mode or "propose")
        self.ralph_max_proposals_per_cycle = max(1, int(ralph_max_proposals_per_cycle or 1))
        self.ralph_max_auto_queue_per_cycle = max(1, int(ralph_max_auto_queue_per_cycle or 1))
        self.discovery_loop = discovery_loop
        self.evolution_loop = evolution_loop
        self.fot_enabled = bool(fot_enabled)
        self.fot_max_candidates_per_cycle = max(
            1,
            int(
                fot_max_candidates_per_cycle
                if fot_max_candidates_per_cycle is not None
                else max(self.ralph_max_proposals_per_cycle, 5)
            ),
        )
        self.fot_max_auto_queue_per_cycle = max(
            0,
            int(
                fot_max_auto_queue_per_cycle
                if fot_max_auto_queue_per_cycle is not None
                else self.ralph_max_auto_queue_per_cycle
            ),
        )
        self.git_tools = GitTools(str(self.project_root))
        self.codex_manager = CodexManager(self.model, self.git_tools, str(self.project_root), brain=self.brain)
        # Backward-compat properties — delegate to self.capabilities
        # (kept for UI callbacks / telemetry that inspect these)
        self.current_goal: str | None = None
        self.active_cycle_summary: dict | None = None
        self.last_cycle_summary: dict | None = None
        self.phase_registry = build_phase_registry(self)

        # Lazy-load skills so missing optional deps don't break startup
        try:
            from agents.skills.registry import all_skills
            self.skills = all_skills(brain=self.brain, model=self.model)
        except Exception as exc:  # pragma: no cover
            log_json("WARN", "skills_load_failed", details={"error": str(exc)})
            self.skills = {}

        # Self-improvement loops (all optional — never block the main loop)
        self._improvement_loops: list = []

        # CASPA-W components — set via attach_caspa() after construction
        self.adaptive_pipeline = None   # AdaptivePipeline
        self.propagation_engine = None  # PropagationEngine
        self.context_graph = None       # ContextGraph
        self._circuit_breaker = CircuitBreaker(threshold=5, cooldown_s=60.0)
        self._ui_callbacks: list = []
        self.fot_arbiter = FoTArbiter(
            goal_queue=self.goal_queue,
            max_selected=self.fot_max_candidates_per_cycle,
            max_auto_queue=self.fot_max_auto_queue_per_cycle,
            goal_validator=self._validate_self_dev_goal,
        )

    @property
    def last_capability_plan(self) -> dict:
        return self.capabilities.last_plan

    @property
    def last_capability_goal_queue(self) -> dict:
        return self.capabilities.last_goal_queue

    @property
    def last_capability_provisioning(self) -> dict:
        return self.capabilities.last_provisioning

    @property
    def last_capability_status(self) -> dict:
        return self.capabilities.last_status

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
                except Exception as exc:
                    log_json("WARN", "ui_callback_error",
                             details={"method": method_name, "error": str(exc)})

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
        log_json("INFO", "orchestrator_loops_attached",
                 details={"count": len(self._improvement_loops),
                           "types": [type(l).__name__ for l in self._improvement_loops]})

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
        log_json("INFO", "orchestrator_caspa_attached", details={
            "adaptive_pipeline": adaptive_pipeline is not None,
            "propagation_engine": propagation_engine is not None,
            "context_graph": context_graph is not None,
        })

    def _validate_self_dev_goal(self, goal: str) -> tuple[bool, str | None]:
        goal = str(goal or "").strip()
        if not goal:
            return False, "empty_goal"
        if len(goal) > 500:
            return False, "goal_too_long"
        if any(token in goal for token in [";", "&&", "||", "`", "$("]):
            return False, "suspicious_goal_content"

        try:
            from core.development_weakness import build_development_context

            dev_context = build_development_context(self.project_root, goal=goal)
        except Exception:
            dev_context = {}
        if (
            isinstance(dev_context, dict)
            and dev_context.get("target_subsystem") == "recursive_self_improvement"
            and dev_context.get("overlap_classification") == "legacy_overlap_present"
            and any(path in goal for path in ("core/evolution_plan.py", "scripts/autonomous_rsi_run.py"))
        ):
            return False, "deprecated_rsi_path"
        return True, None

    def _queue_self_dev_goals(
        self,
        *,
        proposals: list[dict] | None = None,
        beads_follow_up_goals: list[str] | None = None,
        max_to_queue: int | None = None,
    ) -> tuple[list[str], list[dict]]:
        if self.goal_queue is None:
            return [], [{"goal": "", "reason": "goal_queue_unavailable"}]

        max_to_queue = max_to_queue or self.ralph_max_auto_queue_per_cycle
        existing_goals = {str(item) for item in list(getattr(self.goal_queue, "queue", []))}
        queued: list[str] = []
        blocked: list[dict] = []
        candidates: list[tuple[str, str, str | None]] = []

        for proposal in proposals or []:
            goal = str(proposal.get("recommended_goal") or "").strip()
            if proposal.get("queueable") and goal:
                candidates.append(("proposal", goal, proposal.get("proposal_id")))
            elif goal:
                blocked.append({"goal": goal, "reason": proposal.get("queue_block_reason") or "proposal_not_queueable"})

        for goal in beads_follow_up_goals or []:
            normalized_goal = str(goal).strip()
            if normalized_goal:
                candidates.append(("beads", normalized_goal, None))

        for source, goal, proposal_id in candidates:
            if len(queued) >= max_to_queue:
                blocked.append({"goal": goal, "reason": "auto_queue_limit_reached", "source": source, "proposal_id": proposal_id})
                continue
            if goal in existing_goals or goal in queued:
                blocked.append({"goal": goal, "reason": "duplicate_goal", "source": source, "proposal_id": proposal_id})
                continue
            ok, reason = self._validate_self_dev_goal(goal)
            if not ok:
                blocked.append({"goal": goal, "reason": reason or "invalid_goal", "source": source, "proposal_id": proposal_id})
                continue
            self.goal_queue.add(goal)
            queued.append(goal)
            existing_goals.add(goal)

        return queued, blocked

    def run_self_development(self, goal: str | None = None, mode: str | None = None) -> Dict[str, Any]:
        effective_mode = str(mode or self.ralph_mode or "propose")
        if not self.ralph_enabled:
            return {
                "status": "disabled",
                "goal": goal,
                "self_dev_mode": effective_mode,
                "proposal_count": 0,
                "proposals": [],
                "follow_up_goals": [],
                "auto_queued_goals": [],
                "queue_block_reasons": [],
            }

        improvement_service = getattr(self.evolution_loop, "improvement_service", None) if self.evolution_loop else None
        if improvement_service is None:
            return {
                "status": "unavailable",
                "goal": goal,
                "self_dev_mode": effective_mode,
                "proposal_count": 0,
                "proposals": [],
                "follow_up_goals": [],
                "auto_queued_goals": [],
                "queue_block_reasons": [{"goal": "", "reason": "recursive_improvement_service_unavailable"}],
            }

        recent_entries = []
        if self.memory_controller and self.memory_controller.persistent_store:
            try:
                recent_entries = self.memory_controller.persistent_store.read_log(limit=25)
            except Exception as exc:
                log_json("WARN", "self_dev_history_read_failed", details={"error": str(exc)})

        history = [
            improvement_service.normalize_cycle_entry(entry)
            for entry in recent_entries
            if isinstance(entry, dict)
        ]
        proposals = list(improvement_service.evaluate_candidates(history))[: self.ralph_max_proposals_per_cycle]
        source = "history"
        if goal:
            proposals.insert(
                0,
                improvement_service.create_goal_proposal(
                    goal,
                    summary="Manual Ralph-loop goal request queued through the canonical runtime path.",
                ),
            )
            source = "manual_goal"
        elif not proposals:
            try:
                from core.goal_generator import ContextualGoalGenerator

                generated_goal = ContextualGoalGenerator(
                    self.project_root,
                    brain=self.brain,
                    model=self.model,
                ).generate_impactful_goal()
            except Exception as exc:
                log_json("WARN", "self_dev_goal_generation_failed", details={"error": str(exc)})
                generated_goal = "evolve and improve the AURA system via recursive self-improvement"
            proposals = [
                improvement_service.create_goal_proposal(
                    generated_goal,
                    summary="Contextual Ralph-loop follow-up goal synthesized from the current repo state.",
                )
            ]
            source = "contextual_goal"

        proposals = proposals[: self.ralph_max_proposals_per_cycle]
        queued: list[str] = []
        blocked: list[dict] = []
        if effective_mode == "auto_queue":
            queued, blocked = self._queue_self_dev_goals(
                proposals=proposals,
                beads_follow_up_goals=[],
                max_to_queue=self.ralph_max_auto_queue_per_cycle,
            )

        return {
            "status": "ok",
            "goal": goal,
            "source": source,
            "self_dev_mode": effective_mode,
            "history_count": len(history),
            "proposal_count": len(proposals),
            "proposals": proposals,
            "follow_up_goals": [
                proposal.get("recommended_goal")
                for proposal in proposals
                if proposal.get("recommended_goal")
            ],
            "auto_queued_goals": queued,
            "queue_block_reasons": blocked,
        }

    def _load_latest_memory_report(self, bucket: str) -> Dict[str, Any] | None:
        store = getattr(self.memory_controller, "persistent_store", None)
        if store is None:
            return None
        try:
            items = store.query(bucket, limit=1)
        except Exception as exc:
            log_json("WARN", "memory_report_query_failed", details={"bucket": bucket, "error": str(exc)})
            return None
        if not items:
            return None
        latest = items[-1]
        return latest if isinstance(latest, dict) else None

    def _build_review_candidates(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        phase_outputs = entry.get("phase_outputs", {}) if isinstance(entry, dict) else {}
        critique = phase_outputs.get("critique", {}) if isinstance(phase_outputs, dict) else {}
        verification = phase_outputs.get("verification", {}) if isinstance(phase_outputs, dict) else {}
        candidates: List[Dict[str, Any]] = []

        issues = critique.get("issues", []) if isinstance(critique, dict) else []
        if issues:
            issue_text = "; ".join(str(item) for item in issues[:2] if str(item).strip())
            candidates.append({
                "candidate_id": f"critique:{entry.get('cycle_id', 'unknown')}",
                "source": "critique",
                "summary": issue_text or "Critique surfaced follow-up risks.",
                "recommended_goal": "Address critique findings from the latest implementation plan before broadening scope",
                "priority": "medium",
                "confidence": 0.7,
                "queueable": True,
                "requires_human_review": False,
                "beads_recheck_required": False,
                "evidence": ["phase:critique"],
            })

        failures = verification.get("failures", []) if isinstance(verification, dict) else []
        if failures:
            failure_text = "; ".join(str(item) for item in failures[:2] if str(item).strip())
            candidates.append({
                "candidate_id": f"review:{entry.get('cycle_id', 'unknown')}",
                "source": "review",
                "summary": failure_text or "Verification surfaced follow-up failures.",
                "recommended_goal": "Stabilize the latest verification failures with a focused regression fix",
                "priority": "high",
                "confidence": 0.8,
                "queueable": True,
                "requires_human_review": False,
                "beads_recheck_required": False,
                "evidence": ["phase:verify"],
            })

        return candidates

    def _build_beads_candidates(self, beads_state: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        if not isinstance(beads_state, dict):
            return []
        target_subsystem = beads_state.get("target_subsystem")
        return [
            {
                "candidate_id": f"beads:{idx}:{goal}",
                "source": "beads",
                "summary": beads_state.get("summary") or "BEADS requested follow-up work.",
                "recommended_goal": str(goal).strip(),
                "target_subsystem": target_subsystem,
                "priority": "high",
                "confidence": 0.85,
                "queueable": True,
                "requires_human_review": False,
                "beads_recheck_required": False,
                "evidence": ["beads_follow_up_goal"],
            }
            for idx, goal in enumerate(beads_state.get("follow_up_goals", []), start=1)
            if str(goal).strip()
        ]

    def _run_fot_arbiter(
        self,
        entry: Dict[str, Any],
        *,
        discovery_report: Dict[str, Any] | None = None,
        propagation_candidates: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if not self.fot_enabled:
            fot_payload = {
                "status": "disabled",
                "candidates_considered": 0,
                "candidates_selected": 0,
                "selected_candidates": [],
                "queue_delta": 0,
                "auto_queued_goals": [],
                "blocked_candidates": [],
                "sources": [],
                "requires_human_review": 0,
                "beads_rechecks": 0,
            }
            entry.setdefault("phase_outputs", {})["fot"] = fot_payload
            return fot_payload

        phase_outputs = entry.setdefault("phase_outputs", {})
        ralph_payload = phase_outputs.get("ralph", {}) if isinstance(phase_outputs, dict) else {}
        if not isinstance(ralph_payload, dict):
            ralph_payload = {}

        reflection_report = self._load_latest_memory_report("reflection_reports")
        candidates: List[Dict[str, Any]] = []
        candidates.extend(self._build_review_candidates(entry))
        candidates.extend(self._build_beads_candidates(entry.get("beads")))
        candidates.extend(
            candidate
            for candidate in list(ralph_payload.get("proposals", []))
            if isinstance(candidate, dict)
        )
        if isinstance(reflection_report, dict):
            candidates.extend(
                candidate
                for candidate in list(reflection_report.get("fot_candidates", []))
                if isinstance(candidate, dict)
            )
        if isinstance(discovery_report, dict):
            candidates.extend(
                candidate
                for candidate in list(discovery_report.get("fot_candidates", []))
                if isinstance(candidate, dict)
            )
        candidates.extend(candidate for candidate in (propagation_candidates or []) if isinstance(candidate, dict))

        auto_queue = bool(self.ralph_enabled and self.ralph_mode == "auto_queue" and not bool(entry.get("dry_run")))
        arbiter = self.fot_arbiter
        arbiter.goal_queue = self.goal_queue
        arbiter.max_selected = self.fot_max_candidates_per_cycle
        arbiter.max_auto_queue = self.fot_max_auto_queue_per_cycle
        result = arbiter.arbitrate(candidates, auto_queue=auto_queue, dry_run=bool(entry.get("dry_run")))
        fot_payload = {
            "status": "ok",
            "candidates_considered": int(result.get("candidates_considered", 0) or 0),
            "candidates_selected": int(result.get("candidates_selected", 0) or 0),
            "selected_candidates": list(result.get("selected_candidates", [])),
            "queue_delta": int(result.get("queue_delta", 0) or 0),
            "auto_queued_goals": list(result.get("auto_queued_goals", [])),
            "blocked_candidates": list(result.get("blocked_candidates", [])),
            "sources": list(result.get("sources", [])),
            "requires_human_review": int(result.get("requires_human_review", 0) or 0),
            "beads_rechecks": int(result.get("beads_rechecks", 0) or 0),
        }
        phase_outputs["fot"] = fot_payload

        if isinstance(ralph_payload, dict):
            ralph_selected_goals = [
                str(candidate.get("recommended_goal") or "").strip()
                for candidate in fot_payload["selected_candidates"]
                if candidate.get("source") == "recursive_improvement" and str(candidate.get("recommended_goal") or "").strip() in fot_payload["auto_queued_goals"]
            ]
            ralph_payload["auto_queued_goals"] = ralph_selected_goals
            ralph_payload["queue_block_reasons"] = [
                blocked
                for blocked in fot_payload["blocked_candidates"]
                if blocked.get("source") == "recursive_improvement"
            ]
            phase_outputs["ralph"] = ralph_payload

        store = getattr(self.memory_controller, "persistent_store", None)
        if store is not None:
            try:
                store.put("fot_reports", {
                    "cycle_id": entry.get("cycle_id"),
                    "goal": entry.get("goal"),
                    "timestamp": time.time(),
                    **fot_payload,
                })
            except Exception as exc:
                log_json("WARN", "fot_report_store_failed", details={"error": str(exc)})

        log_json(
            "INFO",
            "fot_arbiter_complete",
            details={
                "cycle_id": entry.get("cycle_id"),
                "considered": fot_payload["candidates_considered"],
                "selected": fot_payload["candidates_selected"],
                "queue_delta": fot_payload["queue_delta"],
                "sources": fot_payload["sources"],
            },
        )
        return fot_payload

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
        except Exception as exc:
            log_json("WARN", "retrieve_hints_failed", details={"error": str(exc)})
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
        ranked = heapq.nlargest(
            limit,
            enumerate(summaries),
            key=lambda iv: _score(iv[1], iv[0], total),
        )
        return [s for _, s in ranked]

    def _run_phase(self, name: str, input_data: Dict) -> Dict:
        """Dispatch a single pipeline phase to its registered agent.

        Looks up the agent by *name* in :attr:`agents` and calls
        ``agent.run(input_data)``.  If no agent is registered for *name* the
        phase is silently skipped.

        Args:
            name: The phase identifier (e.g. ``"plan"``, ``"act"``, ``"verify"``).
            input_data: Arbitrary dict passed directly to ``agent.run()``.

        Returns:
            The dict returned by ``agent.run()``, or an empty dict ``{}`` when
            no agent is registered for *name*.
        """
        agent = self.agents.get(name)
        if not agent:
            return {}
        return agent.run(input_data)

    def _goal_identifier(self, goal: str) -> str:
        return self._parse_bead_id(goal) or goal

    def _phase_result_label(self, phase: str, payload: Any) -> str:
        if phase == "verify":
            return (self._normalize_verification_result(payload).get("status") if isinstance(payload, dict) else None) or "fail"
        if phase == "sandbox" and isinstance(payload, dict):
            if payload.get("status") == "skip":
                return "skip"
            return "pass" if payload.get("passed", True) else "fail"
        if phase == "apply" and isinstance(payload, dict):
            return "fail" if payload.get("failed") else "pass"
        if isinstance(payload, dict):
            status = payload.get("status")
            if status in {"pass", "fail", "skip"}:
                return status
            if status == "error":
                return "fail"
            if status == "ok":
                return "pass"
        return "pass"

    def _emit_phase_telemetry(self, cycle_id: str, goal: str, phase: str, duration_ms: int, payload: Any) -> None:
        log_json(
            "INFO",
            "cycle_phase_telemetry",
            details={
                "cycle_id": cycle_id,
                "goal_id": self._goal_identifier(goal),
                "goal": goal,
                "phase": phase,
                "duration_ms": duration_ms,
                "result": self._phase_result_label(phase, payload),
            },
        )

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

    def _apply_change_set(
        self, change_set: Dict, dry_run: bool
    ) -> Dict[str, List]:
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
                    self.project_root, file_path, old_code, new_code,
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

        Inspects the ``"failures"`` list and ``"logs"`` string in *verification*
        for well-known signal words to determine how the orchestrator should
        respond.

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
        failure_items = verification.get("failures", [])
        failure_text = " ".join(str(f) for f in failure_items)
        extra_fields = [
            verification.get("logs", ""),
            verification.get("summary", ""),
            verification.get("stderr", ""),
            verification.get("error", ""),
            verification.get("details", ""),
        ]
        combined = " ".join([failure_text, *(str(item) for item in extra_fields if item)]).lower()

        structural_signals = [
            "architecture",
            "circular",
            "api_breaking",
            "api breaking",
            "breaking_change",
            "breaking change",
            "design",
            "interface",
            "contract",
            "schema mismatch",
            "incompatible interface",
        ]
        external_signals = [
            "dependency",
            "network",
            "dns",
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "service unavailable",
            "temporary failure in name resolution",
            "environment variable",
            "missing api key",
            "api key not set",
            "permission denied",
            "permissionerror",
            "read-only file system",
            "no space left",
            "permission",
            "no module",
            "no module named",
            "modulenotfounderror",
            "import error",
            "importerror",
            "command not found",
            "executable file not found",
        ]

        if any(s in combined for s in structural_signals):
            return "plan"
        if any(s in combined for s in external_signals):
            return "skip"
        return "act"  # default: code-level fix is worth retrying

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
            sandbox_result = self._run_phase("sandbox", {
                "act": act,
                "dry_run": dry_run,
                "project_root": str(self.project_root),
            }) or {}
            duration_ms = int((time.time() - t0_sandbox) * 1000)
            self._notify_ui("on_phase_complete", "sandbox", duration_ms)
            self._emit_phase_telemetry(phase_outputs.get("_cycle_id", "unknown"), goal, "sandbox", duration_ms, sandbox_result)

            phase_outputs["sandbox"] = sandbox_result
            sandbox_passed = sandbox_result.get("passed", True)
            if sandbox_passed or dry_run:
                break

            stderr_hint = (
                (sandbox_result.get("details") or {}).get("stderr", "")
                or sandbox_result.get("summary", "sandbox_failed")
            )
            log_json("WARN", "sandbox_pre_apply_failed",
                     details={"try": _sandbox_try + 1,
                              "summary": sandbox_result.get("summary", "")})

            if _sandbox_try < MAX_SANDBOX_RETRIES - 1:
                phase_outputs["retry_count"] = phase_outputs.get("retry_count", 0) + 1
                task_bundle["fix_hints"] = [stderr_hint]
                act = self._run_phase("act", {
                    "task": goal,
                    "task_bundle": task_bundle,
                    "dry_run": dry_run,
                    "project_root": str(self.project_root),
                    "fix_hints": [stderr_hint],
                })
                act_attempts_used += 1
                phase_outputs["change_set"] = act
            else:
                log_json("WARN", "sandbox_max_retries_exceeded",
                         details={"max": MAX_SANDBOX_RETRIES,
                                  "continuing_with_best_attempt": True})

        if not sandbox_passed and not dry_run:
            task_bundle["fix_hints"] = [
                (sandbox_result.get("details") or {}).get("stderr", "")
                or sandbox_result.get("summary", "sandbox_failed")
            ]

        return act, sandbox_passed, act_attempts_used

    def _execute_act_verify_attempt(self, goal: str, plan: Dict, task_bundle: Dict, cycle_id: str, phase_outputs: Dict, dry_run: bool):
        """Execute one attempt of act -> sandbox -> apply -> verify."""
        self._notify_ui("on_phase_start", "act")
        t0_act = time.time()
        
        if USE_CODEX_PARALLEL:
            log_json("INFO", "orchestrator_using_parallel_codex_manager")
            act = self.codex_manager.decompose_and_run_parallel(goal, task_bundle)
        else:
            act = self._run_phase("act", {
                "task": goal, "task_bundle": task_bundle, "dry_run": dry_run,
                "project_root": str(self.project_root), "fix_hints": task_bundle.get("fix_hints", []),
            })

        duration_ms = int((time.time() - t0_act) * 1000)
        self._notify_ui("on_phase_complete", "act", duration_ms)
        self._emit_phase_telemetry(cycle_id, goal, "act", duration_ms, act)

        if validate_phase_output("change_set", act) and self.debugger:
            debug_hint = self.debugger.diagnose(error=f"CoderAgent produced invalid change_set", context={"goal": goal, "plan": plan, "task_bundle": task_bundle})
            act = self._run_phase("act", {"task": goal, "task_bundle": task_bundle, "dry_run": dry_run, "project_root": str(self.project_root), "debug_hint": debug_hint})
        
        phase_outputs["change_set"] = act
        act, _passed, extra_uses = self._run_sandbox_loop(goal, act, task_bundle, dry_run, phase_outputs)

        self._notify_ui("on_phase_start", "apply")
        t0_apply = time.time()
        apply_result = self._apply_change_set(act, dry_run=dry_run)
        duration_ms = int((time.time() - t0_apply) * 1000)
        self._notify_ui("on_phase_complete", "apply", duration_ms)
        self._emit_phase_telemetry(cycle_id, goal, "apply", duration_ms, apply_result)
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
        duration_ms = int((time.time() - t0_verify) * 1000)
        self._notify_ui("on_phase_complete", "verify", duration_ms, success=(verification.get("status") in ("pass", "skip")))
        self._emit_phase_telemetry(cycle_id, goal, "verify", duration_ms, verification)
        
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
            if route == "plan" and plan_attempt < max_plan_retries:
                phase_outputs["retry_count"] = phase_outputs.get("retry_count", 0) + 1
                phase_outputs["_failure_context"] = {"failures": verification.get("failures", []), "logs": verification.get("logs", ""), "route": "plan"}
                replan_needed = True
                break
            elif route == "skip":
                break
            task_bundle["fix_hints"] = verification.get("failures", [])

        return verification, replan_needed, None

    def _execute_plan_critique_synthesize(self, goal: str, context: Dict, skill_context: Dict, pipeline_cfg: Any, phase_outputs: Dict) -> Tuple[Dict, Dict]:
        """Section 3-5: PLAN -> CRITIQUE -> SYNTHESIZE."""
        beads_decision = phase_outputs.get("beads_gate", {})
        self._notify_ui("on_phase_start", "plan")
        t0 = time.time()
        plan = self._run_phase("plan", {
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
        })
        duration_ms = int((time.time() - t0) * 1000)
        self._notify_ui("on_phase_complete", "plan", duration_ms)
        self._emit_phase_telemetry(phase_outputs.get("_cycle_id", "unknown"), goal, "plan", duration_ms, plan)
        phase_outputs["plan"] = plan

        self._notify_ui("on_phase_start", "critique")
        t0 = time.time()
        critique = self._run_phase("critique", {
            "task": goal,
            "plan": plan.get("steps", []),
        })
        duration_ms = int((time.time() - t0) * 1000)
        self._notify_ui("on_phase_complete", "critique", duration_ms)
        self._emit_phase_telemetry(phase_outputs.get("_cycle_id", "unknown"), goal, "critique", duration_ms, critique)
        phase_outputs["critique"] = critique

        self._notify_ui("on_phase_start", "synthesize")
        t0 = time.time()
        task_bundle = self._run_phase("synthesize", {
            "goal": goal,
            "plan": plan,
            "critique": critique,
            "beads_decision": beads_decision,
        })
        duration_ms = int((time.time() - t0) * 1000)
        self._notify_ui("on_phase_complete", "synthesize", duration_ms)
        self._emit_phase_telemetry(phase_outputs.get("_cycle_id", "unknown"), goal, "synthesize", duration_ms, task_bundle)
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
                goal=goal, plan=plan, task_bundle=task_bundle, pipeline_cfg=pipeline_cfg,
                cycle_id=cycle_id, phase_outputs=phase_outputs, dry_run=dry_run,
                plan_attempt=plan_attempt, max_plan_retries=max_plan_retries, skill_context=skill_context
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
                goal, goal_type,
                consecutive_fails=self._circuit_breaker.consecutive_fails,
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
        """Section 0.1: CAPABILITY MANAGEMENT (delegates to CapabilityCoordinator)."""
        self.capabilities.handle(
            goal,
            pipeline_cfg,
            phase_outputs,
            skills=self.skills,
            goal_queue=self.goal_queue,
            dry_run=dry_run,
        )

    def _run_ingest_phase(self, goal: str, cycle_id: str, phase_outputs: Dict) -> Dict:
        """Section 1: INGEST."""
        self._notify_ui("on_phase_start", "ingest")
        t0 = time.time()
        working_memory = self.memory_controller.retrieve(MemoryTier.WORKING)
        session_memory = self.memory_controller.retrieve(MemoryTier.SESSION)
        context = self._run_phase("ingest", {"goal": goal, "project_root": str(self.project_root), "hints": self._retrieve_hints(goal), "working_memory": working_memory, "session_memory": session_memory})
        duration_ms = int((time.time() - t0) * 1000)
        self._notify_ui("on_phase_complete", "ingest", duration_ms)
        self._emit_phase_telemetry(cycle_id, goal, "ingest", duration_ms, context)
        if "bundle" in context:
            self._notify_ui("on_context_assembled", context["bundle"])
        
        errors = validate_phase_output("context", context)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "context", "errors": errors})
        phase_outputs["context"] = context
        return context

    def _dispatch_skills(self, goal_type: str, pipeline_cfg: Any, phase_outputs: Dict) -> Dict:
        """Section 2: SKILL DISPATCH."""
        self._notify_ui("on_phase_start", "skill_dispatch")
        t0 = time.monotonic()
        skill_context: Dict = {}
        if self.skills and pipeline_cfg.skill_set:
            active_skills = {k: self.skills[k] for k in pipeline_cfg.skill_set if k in self.skills}
            skill_context = dispatch_skills(goal_type, active_skills, str(self.project_root))
        phase_outputs["skill_context"] = skill_context
        duration_ms = int((time.monotonic() - t0) * 1000)
        self._notify_ui("on_phase_complete", "skill_dispatch", duration_ms)
        self._emit_phase_telemetry(phase_outputs.get("_cycle_id", "unknown"), self.current_goal or "", "skill_dispatch", duration_ms, skill_context)
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
            "status": (
                decision.get("status")
                if decision
                else ("capability_unavailable" if result.get("error") == "capability_unavailable" else ("error" if not result.get("ok") else None))
            ),
            "decision_id": decision.get("decision_id"),
            "summary": decision.get("summary"),
            "required_constraints": list(decision.get("required_constraints", [])) if isinstance(decision, dict) else [],
            "required_skills": list(decision.get("required_skills", [])) if isinstance(decision, dict) else [],
            "required_tests": list(decision.get("required_tests", [])) if isinstance(decision, dict) else [],
            "follow_up_goals": list(decision.get("follow_up_goals", [])) if isinstance(decision, dict) else [],
            "target_subsystem": decision.get("target_subsystem") if isinstance(decision, dict) else None,
            "canonical_path": decision.get("canonical_path") if isinstance(decision, dict) else None,
            "overlap_classification": decision.get("overlap_classification") if isinstance(decision, dict) else None,
            "validation_status": decision.get("validation_status") if isinstance(decision, dict) else None,
            "required_remediation": list(decision.get("required_remediation", [])) if isinstance(decision, dict) else [],
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
        sanitized_outputs = dict(phase_outputs)
        sanitized_outputs.pop("_cycle_id", None)
        sanitized_outputs.pop("_failure_context", None)
        entry = {
            "cycle_id": cycle_id,
            "goal": goal,
            "goal_type": goal_type,
            "phase_outputs": sanitized_outputs,
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
        duration_ms = int((time.time() - t0) * 1000)
        self._notify_ui("on_phase_complete", "reflect", duration_ms)
        self._emit_phase_telemetry(cycle_id, self.current_goal or "", "reflect", duration_ms, reflection)
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
            self._circuit_breaker.record_success()
        elif verify_status == "fail":
            self._circuit_breaker.record_failure()

        # Phase 8: measure()
        self._notify_ui("on_phase_start", "measure")
        t0_measure = time.time()
        changed_files = phase_outputs.get("apply_result", {}).get("applied", [])
        quality = run_quality_snapshot(self.project_root, changed_files=changed_files)
        phase_outputs["quality"] = quality
        duration_ms = int((time.time() - t0_measure) * 1000)
        self._notify_ui("on_phase_complete", "measure", duration_ms)
        self._emit_phase_telemetry(cycle_id, goal, "measure", duration_ms, quality)

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

        if self.brain:
            try:
                self.brain.set(f"outcome:{cycle_id}", outcome.to_json())
                self.brain.remember(f"Cycle completed: {goal} -> {'SUCCESS' if passed else 'FAILED'}")
            except Exception as exc:
                log_json("WARN", "brain_outcome_storage_failed", details={"error": str(exc)})

        duration_ms = int((time.time() - t0_learn) * 1000)
        self._notify_ui("on_phase_complete", "learn", duration_ms)
        self._emit_phase_telemetry(cycle_id, goal, "learn", duration_ms, summary)

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
        discovery_report: Dict[str, Any] | None = None
        if self.discovery_loop:
            log_json("INFO", "orchestrator_triggering_discovery")
            try:
                discovery_report = self.discovery_loop.on_cycle_complete(entry, queue=False)
            except TypeError:
                discovery_report = self.discovery_loop.on_cycle_complete(entry)
        self._notify_ui("on_phase_complete", "discover", 0)
        self._emit_phase_telemetry(
            cycle_id,
            goal,
            "discover",
            0,
            {
                "status": "ok" if self.discovery_loop else "skip",
                "candidate_count": len((discovery_report or {}).get("fot_candidates", [])) if isinstance(discovery_report, dict) else 0,
            },
        )

        # Phase 11: evolve()
        self._notify_ui("on_phase_start", "evolve")
        if self.evolution_loop:
            log_json("INFO", "orchestrator_triggering_evolution")
            self.evolution_loop.on_cycle_complete(entry)
        self._notify_ui("on_phase_complete", "evolve", 0)
        self._emit_phase_telemetry(cycle_id, goal, "evolve", 0, {"status": "ok" if self.evolution_loop else "skip"})

        propagation_candidates: List[Dict[str, Any]] = []
        if self.propagation_engine is not None:
            try:
                propagation_candidates = list(self.propagation_engine.on_cycle_complete(entry, queue=False))
            except TypeError:
                collector = getattr(self.propagation_engine, "collect_candidates", None)
                if callable(collector):
                    propagation_candidates = list(collector(entry))
                else:
                    self.propagation_engine.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "propagation_engine_error", details={"error": str(exc)})

        self._run_fot_arbiter(
            entry,
            discovery_report=discovery_report,
            propagation_candidates=propagation_candidates,
        )

        summary = self._refresh_cycle_summary(entry)
        self._notify_ui("on_cycle_complete", summary)
        self.current_goal = None
        self.active_cycle_summary = None
        if self.memory_controller.persistent_store:
            self.memory_controller.persistent_store.append_log(entry)
            self.memory_controller.store(
                MemoryTier.PROJECT,
                json.dumps(summary),
                metadata={"type": "cycle_summary", "goal": goal, "cycle_id": cycle_id}
            )
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
            beads_skill.run({
                "cmd": "update",
                "id": bead_id,
                "args": ["--status", "in_progress"]
            })

    def _close_bead(self, bead_id: str, reason: str):
        """Close a bead using BeadsSkill."""
        beads_skill = self._get_beads_skill()
        if beads_skill is not None:
            log_json("INFO", "orchestrator_closing_bead", details={"bead_id": bead_id})
            beads_skill.run({
                "cmd": "close",
                "id": bead_id,
                "args": ["--reason", reason]
            })

    def run_cycle(self, goal: str, dry_run: bool = False) -> Dict:
        """Execute a single complete plan-act-verify cycle for *goal*."""
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        started_at = time.time()

        if self._circuit_breaker.is_open():
            log_json("WARN", "circuit_breaker_open", details=self._circuit_breaker.as_dict())
            return self._build_early_stop_entry(
                cycle_id=cycle_id,
                goal=goal,
                goal_type=classify_goal(goal),
                phase_outputs={"retry_count": 0, "dry_run": dry_run},
                started_at=started_at,
                stop_reason="CIRCUIT_BREAKER_OPEN",
            )

        phase_outputs = {"retry_count": 0, "dry_run": dry_run, "_cycle_id": cycle_id}
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
            goal=goal, context=context, skill_context=skill_context,
            pipeline_cfg=pipeline_cfg, cycle_id=cycle_id,
            phase_outputs=phase_outputs, dry_run=dry_run,
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

        phase_outputs.pop("_failure_context", None)
        phase_outputs.pop("_cycle_id", None)
        return self._record_cycle_outcome(cycle_id, goal, goal_type, phase_outputs, started_at)

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

    def run_loop(self, goal: str, max_cycles: int = 5, dry_run: bool = False) -> Dict:
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
        
        cycle_count = 0
        while cycle_count < max_cycles:
            cycle_count += 1
            entry = self.run_cycle(goal, dry_run=dry_run)
            history.append(entry)
            
            if entry.get("stop_reason"):
                stop_reason = entry["stop_reason"]
                break
                
            stop_reason = self.policy.evaluate(history, entry.get("phase_outputs", {}).get("verification", {}), started_at=started_at)
            
            if stop_reason:
                if stop_reason != "PASS" and sys.stdin.isatty() and os.environ.get("AURA_AUTO_APPROVE", "").strip() != "1":
                    # Dynamic Loop Steering
                    print(f"\n[AURA Orchestrator] Loop is halting with reason: {stop_reason}")
                    try:
                        hint = input("Provide a hint to continue, or press Enter to halt: ").strip()
                        if hint:
                            log_json("INFO", "user_provided_hint", details={"hint": hint})
                            goal = f"{goal}\n\n[USER HINT]: {hint}"
                            stop_reason = ""
                            max_cycles += 1 # grant one more cycle
                            continue
                    except EOFError:
                        pass
                
                entry["stop_reason"] = stop_reason
                self._refresh_cycle_summary(entry)
                break
                
        if not stop_reason and history:
            if sys.stdin.isatty() and os.environ.get("AURA_AUTO_APPROVE", "").strip() != "1":
                print(f"\n[AURA Orchestrator] Reached MAX_CYCLES ({max_cycles}).")
                try:
                    hint = input("Provide a hint to continue, or press Enter to halt: ").strip()
                    if hint:
                        log_json("INFO", "user_provided_hint_max_cycles", details={"hint": hint})
                        goal = f"{goal}\n\n[USER HINT]: {hint}"
                        # Execute one more cycle recursively or just loop again
                        max_cycles += 1
                        entry = self.run_cycle(goal, dry_run=dry_run)
                        history.append(entry)
                        stop_reason = self.policy.evaluate(history, entry.get("phase_outputs", {}).get("verification", {}), started_at=started_at)
                except EOFError:
                    pass
                    
            if not stop_reason:
                stop_reason = "MAX_CYCLES"
            history[-1]["stop_reason"] = stop_reason
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
            "cycles_used": len(history),
        }
