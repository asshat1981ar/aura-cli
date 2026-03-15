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
import heapq
import os
import time
import uuid
from unittest.mock import Mock
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import core.file_tools as file_tools_module
from core.logging_utils import log_json
from core.exceptions import PhaseExecutionError, PhaseValidationError, ValidationError, PreflightError
from core.capability_manager import (
    analyze_capability_needs,
    provision_capability_actions,
    queue_follow_up_goals,
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
    apply_change_set,
    snapshot_file_state,
    restore_file_state,
)
from core.schema import validate_phase_output
from core.skill_dispatcher import classify_goal, dispatch_skills
from core.human_gate import HumanGate
from memory.controller import memory_controller, MemoryTier
from core.sandbox_loop import run_sandbox_loop


class BeadsSyncLoop:
    """Periodic sync loop for BEADS external goals."""

    EVERY_N = 3

    def __init__(self, beads_skill, project_root: Path | str | None = None):
        self.beads_skill = beads_skill
        self.project_root = Path(project_root) if project_root is not None else None
        self._n = 0

    def _skill_input(self, cmd: str, **extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"cmd": cmd, **extra}
        if self.project_root is not None:
            payload["project_root"] = str(self.project_root)
        return payload

    def on_cycle_complete(self, entry):
        if not self.beads_skill:
            return

        phase_outputs = entry.get("phase_outputs", {}) or {}
        if phase_outputs.get("dry_run"):
            return

        self._n += 1
        if self._n % self.EVERY_N != 0:
            return

        try:
            self.beads_skill.run(self._skill_input("sync"))
        except Exception as exc:  # pragma: no cover
            log_json("WARN", "beads_sync_failed", details={"error": str(exc)})


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

    _STRUCTURAL_FAILURE_SIGNALS = (
        "architecture",
        "circular",
        "api_breaking",
        "breaking_change",
        "design",
        "interface",
        "contract",
    )
    _EXTERNAL_FAILURE_SIGNALS = (
        "dependency",
        "network",
        "env",
        "environment",
        "permission",
        "not found",
        "no module",
        "import error",
    )

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
        beads_bridge=None,
        beads_enabled: bool = False,
        beads_required: bool = False,
        beads_skill=None,
        goal_queue=None,
        brain=None,
        model=None,
        planner=None,
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
        self.auto_add_capabilities = auto_add_capabilities
        self.auto_queue_missing_capabilities = auto_queue_missing_capabilities
        self.auto_provision_mcp = auto_provision_mcp
        self.auto_start_mcp_servers = auto_start_mcp_servers
        self.beads_bridge = beads_bridge
        self.beads_enabled = beads_enabled
        self.beads_required = beads_required
        self.beads_skill = beads_skill
        self.goal_queue = goal_queue
        self.brain = brain
        self.model = model
        self.planner = planner
        self.last_capability_plan: dict = {}
        self.last_capability_goal_queue: dict = {}
        self.last_capability_provisioning: dict = {}
        self.last_capability_status: dict = {}

        # Lazy-load skills so missing optional deps don't break startup
        try:
            from agents.skills.registry import all_skills
            self.skills = all_skills()
        except Exception as exc:  # pragma: no cover
            log_json("WARN", "skills_load_failed", details={"error": str(exc)})
            self.skills = {}

        if not self.beads_skill and self.skills:
            self.beads_skill = self.skills.get("beads_skill")

        # Self-improvement loops (all optional — never block the main loop)
        self._improvement_loops: list = []

        # CASPA-W components — set via attach_caspa() after construction
        self.adaptive_pipeline = None   # AdaptivePipeline
        self.propagation_engine = None  # PropagationEngine
        self.context_graph = None       # ContextGraph
        self._consecutive_fails: int = 0
        self._ui_callbacks: list = []
        self.current_goal: Optional[str] = None
        self.active_cycle_summary: Optional[dict] = None
        self.last_cycle_summary: Optional[dict] = None

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

    # ── BEADS helpers ──────────────────────────────────────────────────────────

    def _run_beads_gate(self, goal: str, goal_type: str, dry_run: bool, phase_outputs: Dict, corr_id: str):
        if not (self.beads_enabled and self.beads_bridge):
            return None

        beads_payload = {}
        try:
            beads_payload = self.beads_bridge.run({
                "goal": goal,
                "goal_type": goal_type,
                "runtime_mode": "dry_run" if dry_run else "run",
                "corr_id": corr_id,
            }) or {}
        except Exception as exc:
            log_json("WARN", "beads_bridge_failed", details={"error": str(exc)})
            if not self.beads_required:
                return None
            beads_payload = {"status": "error", "error": str(exc)}

        decision_payload = beads_payload.get("decision")
        if isinstance(decision_payload, dict):
            normalized_payload = dict(decision_payload)
            if beads_payload.get("error") is not None:
                normalized_payload.setdefault("error", beads_payload.get("error"))
            if beads_payload.get("stderr") is not None:
                normalized_payload.setdefault("stderr", beads_payload.get("stderr"))
            if beads_payload.get("duration_ms") is not None:
                normalized_payload.setdefault("duration_ms", beads_payload.get("duration_ms"))
            if beads_payload.get("schema_version") is not None:
                normalized_payload.setdefault("schema_version", beads_payload.get("schema_version"))
            beads_payload = normalized_payload

        phase_outputs["beads_gate"] = beads_payload
        status = beads_payload.get("status")
        if status == "block":
            return {"stop_reason": "BEADS_BLOCKED", "beads": beads_payload}
        if status == "revise":
            return {"stop_reason": "BEADS_REVISE", "beads": beads_payload}
        if status == "error" and self.beads_required:
            return {"stop_reason": "BEADS_UNAVAILABLE", "beads": beads_payload}
        return None

    def _build_beads_context(self, phase_outputs: Dict) -> Dict:
        beads_payload = phase_outputs.get("beads_gate", {})
        if not isinstance(beads_payload, dict):
            return {}

        def _string_list(key: str) -> list[str]:
            values = beads_payload.get(key, [])
            if not isinstance(values, list):
                return []
            normalized = []
            for item in values:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text or text in normalized:
                    continue
                normalized.append(text)
            return normalized

        return {
            "status": beads_payload.get("status"),
            "decision_id": beads_payload.get("decision_id"),
            "summary": beads_payload.get("summary"),
            "required_constraints": _string_list("required_constraints"),
            "required_skills": _string_list("required_skills"),
            "required_tests": _string_list("required_tests"),
            "follow_up_goals": _string_list("follow_up_goals"),
        }

    def _queue_beads_follow_up_goals(self, phase_outputs: Dict, dry_run: bool) -> None:
        beads_context = self._build_beads_context(phase_outputs)
        follow_up_goals = beads_context.get("follow_up_goals", [])
        if not follow_up_goals:
            return

        beads_goal_queue = queue_follow_up_goals(
            goal_queue=self.goal_queue,
            goals=follow_up_goals,
            enabled=True,
            dry_run=dry_run,
        )
        if beads_goal_queue.get("queued") or beads_goal_queue.get("skipped"):
            phase_outputs["beads_goal_queue"] = beads_goal_queue

    def _claim_bead(self, bead_id: str):
        beads_skill = (self.skills or {}).get("beads_skill") or self.beads_skill
        if not (self.beads_enabled and beads_skill):
            return
        try:
            beads_skill.run(
                {
                    "cmd": "update",
                    "project_root": str(self.project_root),
                    "id": bead_id,
                    "args": ["--status", "in_progress"],
                }
            )
        except Exception as exc:  # pragma: no cover
            log_json("WARN", "beads_claim_failed", details={"error": str(exc), "id": bead_id})

    def _close_bead(self, bead_id: str, reason: str):
        beads_skill = (self.skills or {}).get("beads_skill") or self.beads_skill
        if not (self.beads_enabled and beads_skill):
            return
        try:
            beads_skill.run(
                {
                    "cmd": "close",
                    "project_root": str(self.project_root),
                    "id": bead_id,
                    "args": ["--reason", reason],
                }
            )
        except Exception as exc:  # pragma: no cover
            log_json("WARN", "beads_close_failed", details={"error": str(exc), "id": bead_id})

    def poll_external_goals(self):
        beads_skill = (self.skills or {}).get("beads_skill") or self.beads_skill
        if not (self.beads_enabled and beads_skill):
            return []
        try:
            resp = beads_skill.run({"cmd": "ready", "project_root": str(self.project_root)}) or {}
        except Exception as exc:  # pragma: no cover
            log_json("WARN", "beads_poll_failed", details={"error": str(exc)})
            return []

        goals = []
        for item in resp.get("ready", []) or []:
            bead_id = item.get("id")
            title = item.get("title") or item.get("summary") or ""
            if bead_id:
                goals.append(f"bead:{bead_id}: {title}".strip())
        return goals

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
        if limit <= 0:
            return []
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
        top_hints: list[tuple[float, int, dict]] = []
        for rank, summary in enumerate(summaries):
            entry = (_score(summary, rank, total), -rank, summary)
            if len(top_hints) < limit:
                heapq.heappush(top_hints, entry)
                continue
            if entry > top_hints[0]:
                heapq.heapreplace(top_hints, entry)

        top_hints.sort(reverse=True)
        return [summary for _, _, summary in top_hints]

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
        corr_id = input_data.get("corr_id")
        try:
            return agent.run(input_data)
        except ValidationError as exc:
            log_json("WARN", "phase_validation_error", details={"phase": name, "error": str(exc)}, corr_id=corr_id, phase=name, outcome="fail", failure_reason="validation_error")
            raise PhaseValidationError(name, str(exc)) from exc
        except Exception as exc:
            log_json("ERROR", "phase_execution_error", details={"phase": name, "error": str(exc)}, corr_id=corr_id, phase=name, outcome="fail", failure_reason=type(exc).__name__)
            raise PhaseExecutionError(name, str(exc)) from exc

    def _snapshot_file_state(self, file_path: str) -> Dict:
        """Capture a restorable snapshot of a target file before mutation."""
        return snapshot_file_state(self.project_root, file_path)

    def _restore_applied_changes(self, snapshots: List[Dict]) -> None:
        """Restore only the files mutated by the current loop attempt.

        This avoids touching unrelated user changes elsewhere in the repo.
        """
        failed = restore_file_state(snapshots)
        
        failed_files = {f["file"] for f in failed}
        restored = [s["file"] for s in snapshots if s["file"] not in failed_files]

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
        apply_change_fn = file_tools_module.apply_change_with_explicit_overwrite_policy
        if isinstance(file_tools_module.apply_change_with_explicit_overwrite_policy, Mock):
            apply_change_fn = file_tools_module.apply_change_with_explicit_overwrite_policy
        if isinstance(apply_change_with_explicit_overwrite_policy, Mock):
            apply_change_fn = apply_change_with_explicit_overwrite_policy

        return apply_change_set(
            self.project_root,
            change_set,
            dry_run,
            apply_change_fn=apply_change_fn,
        )

    def _failure_signal_text(self, verification: Dict) -> str:
        failures = " ".join(str(f) for f in verification.get("failures", []))
        logs = str(verification.get("logs", ""))
        return (failures + " " + logs).lower()

    def _matched_failure_signal(
        self,
        combined: str,
        signals: Tuple[str, ...],
    ) -> Optional[str]:
        for signal in signals:
            if signal in combined:
                return signal
        return None

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
        combined = self._failure_signal_text(verification)

        if self._matched_failure_signal(combined, self._STRUCTURAL_FAILURE_SIGNALS):
            return "plan"
        if self._matched_failure_signal(combined, self._EXTERNAL_FAILURE_SIGNALS):
            return "skip"
        return "act"  # default: code-level fix is worth retrying

    def _explain_route_decision(self, verification: Dict, route: str) -> str:
        """Explain which failure signal triggered a routing decision."""
        combined = self._failure_signal_text(verification)

        if route == "plan":
            signal = self._matched_failure_signal(
                combined,
                self._STRUCTURAL_FAILURE_SIGNALS,
            )
            if signal:
                return f"structural: detected '{signal}' in failures/logs"
            return "structural: default re-plan strategy"

        if route == "skip":
            signal = self._matched_failure_signal(
                combined,
                self._EXTERNAL_FAILURE_SIGNALS,
            )
            if signal:
                return f"external: detected '{signal}' in failures/logs"
            return "external: environment issue detected"

        return "recoverable: code-level fix worth retrying"

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
        corr_id: str,
    ):
        """Run the sandbox pre-apply check, retrying up to MAX_SANDBOX_RETRIES.

        On each failure, injects stderr as a fix_hint and re-generates code.

        Returns:
            Tuple of (final_act_dict, sandbox_passed, act_attempt_delta).
        """
        def run_phase_wrapper(phase: str, context: Dict) -> Dict:
            return self._run_phase(phase, context) or {}

        return run_sandbox_loop(
            run_phase=run_phase_wrapper,
            notify_ui=self._notify_ui,
            project_root=str(self.project_root),
            goal=goal,
            act=act,
            task_bundle=task_bundle,
            dry_run=dry_run,
            phase_outputs=phase_outputs,
            corr_id=corr_id,
        )

    def _execute_act_verify_attempt(self, goal: str, plan: Dict, task_bundle: Dict, cycle_id: str, phase_outputs: Dict, dry_run: bool, corr_id: str):
        """Execute one attempt of act -> sandbox -> apply -> verify."""
        self._notify_ui("on_phase_start", "act")
        t0_act = time.time()
        act = self._run_phase("act", {
            "task": goal, "task_bundle": task_bundle, "dry_run": dry_run,
            "project_root": str(self.project_root), "fix_hints": task_bundle.get("fix_hints", []),
            "corr_id": corr_id,
        })
        self._notify_ui("on_phase_complete", "act", (time.time() - t0_act) * 1000)

        change_set_errors = validate_phase_output("change_set", act)
        if change_set_errors and self.debugger:
            debug_hint = self.debugger.diagnose(error=f"CoderAgent produced invalid change_set", context={"goal": goal, "plan": plan, "task_bundle": task_bundle})
            act = self._run_phase("act", {"task": goal, "task_bundle": task_bundle, "dry_run": dry_run, "project_root": str(self.project_root), "debug_hint": debug_hint, "corr_id": corr_id})
            change_set_errors = validate_phase_output("change_set", act)
        if change_set_errors and dry_run:
            act = {"changes": []}
            change_set_errors = []
        # Short-circuit known-bad change_set to avoid unhandled apply/verify failures
        if change_set_errors:
            raise PhaseValidationError("change_set", "invalid change_set payload")

        phase_outputs["change_set"] = act
        act, _passed, extra_uses = self._run_sandbox_loop(goal, act, task_bundle, dry_run, phase_outputs, corr_id=corr_id)

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
            verification = self._run_phase("verify", {"change_set": act, "dry_run": dry_run, "project_root": str(self.project_root), "tests": tests, "corr_id": corr_id})
        verification = self._normalize_verification_result(verification)
        self._notify_ui("on_phase_complete", "verify", (time.time() - t0_verify) * 1000, success=(verification.get("status") in ("pass", "skip")))
        
        phase_outputs["verification"] = verification
        return act, apply_result, verification, extra_uses

    def _run_act_loop(self, goal: str, plan: Dict, task_bundle: Dict, pipeline_cfg, cycle_id: str, phase_outputs: Dict, dry_run: bool, plan_attempt: int, max_plan_retries: int, skill_context: Dict, corr_id: Optional[str] = None):
        max_act_attempts = pipeline_cfg.max_act_attempts
        act_attempt = 0
        verification: Dict = {}
        replan_needed = False

        corr_id = corr_id or cycle_id

        while act_attempt < max_act_attempts:
            act_attempt += 1
            if act_attempt > 1:
                time.sleep(min(2 ** (act_attempt - 2), 16))

            act, apply_result, verification, extra_uses = self._execute_act_verify_attempt(goal, plan, task_bundle, cycle_id, phase_outputs, dry_run, corr_id)
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
            failure_context = phase_outputs.get("_failure_context")
            if not isinstance(failure_context, dict):
                failure_context = {}
                phase_outputs["_failure_context"] = failure_context
            failure_context["failures"] = verification.get("failures", [])
            failure_context["logs"] = verification.get("logs", "")
            failure_context["route"] = route
            failure_context["routing_reason"] = self._explain_route_decision(
                verification,
                route,
            )
            if route == "plan" and plan_attempt < max_plan_retries:
                replan_needed = True
                break
            elif route == "skip":
                break
            task_bundle["fix_hints"] = verification.get("failures", [])

        return verification, replan_needed, None

    def _execute_plan_critique_synthesize(self, goal: str, context: Dict, skill_context: Dict, pipeline_cfg: Any, phase_outputs: Dict) -> Tuple[Dict, Dict]:
        """Section 3-5: PLAN -> CRITIQUE -> SYNTHESIZE."""
        beads_context = self._build_beads_context(phase_outputs)
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
            "beads_context": beads_context,
        })
        self._notify_ui("on_phase_complete", "plan", (time.time() - t0) * 1000)
        phase_outputs["plan"] = plan

        self._notify_ui("on_phase_start", "critique")
        t0 = time.time()
        critique = self._run_phase("critique", {
            "task": goal,
            "plan": plan.get("steps", []),
        })
        self._notify_ui("on_phase_complete", "critique", (time.time() - t0) * 1000)
        phase_outputs["critique"] = critique

        self._notify_ui("on_phase_start", "synthesize")
        t0 = time.time()
        task_bundle = self._run_phase("synthesize", {
            "goal": goal,
            "plan": plan,
            "critique": critique,
            "beads_context": beads_context,
        })
        self._notify_ui("on_phase_complete", "synthesize", (time.time() - t0) * 1000)
        phase_outputs["task_bundle"] = task_bundle
        return plan, task_bundle

    def _run_plan_loop(self, goal: str, context: Dict, skill_context: Dict, pipeline_cfg: Any, cycle_id: str, phase_outputs: Dict, dry_run: bool, corr_id: str) -> Tuple[Dict, Optional[Dict]]:
        max_plan_retries = getattr(pipeline_cfg, "plan_retries", 3)
        plan_attempt = 0
        verification: Dict = {}

        while plan_attempt < max_plan_retries:
            plan_attempt += 1
            plan, task_bundle = self._execute_plan_critique_synthesize(goal, context, skill_context, pipeline_cfg, phase_outputs)

            verification, replan_needed, early_return = self._run_act_loop(
                goal=goal, plan=plan, task_bundle=task_bundle, pipeline_cfg=pipeline_cfg,
                cycle_id=cycle_id, phase_outputs=phase_outputs, dry_run=dry_run,
                plan_attempt=plan_attempt, max_plan_retries=max_plan_retries, skill_context=skill_context, corr_id=corr_id
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

    def _run_ingest_phase(self, goal: str, cycle_id: str, phase_outputs: Dict, corr_id: str) -> Dict:
        """Section 1: INGEST."""
        self._notify_ui("on_phase_start", "ingest")
        t0 = time.time()
        working_memory = self.memory_controller.retrieve(MemoryTier.WORKING, corr_id=corr_id)
        session_memory = self.memory_controller.retrieve(MemoryTier.SESSION, corr_id=corr_id)
        context = self._run_phase("ingest", {"goal": goal, "project_root": str(self.project_root), "hints": self._retrieve_hints(goal), "working_memory": working_memory, "session_memory": session_memory, "corr_id": corr_id})
        self._notify_ui("on_phase_complete", "ingest", (time.time() - t0) * 1000)
        if "bundle" in context:
            self._notify_ui("on_context_assembled", context["bundle"])
        
        errors = validate_phase_output("context", context)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "context", "errors": errors})
        phase_outputs["context"] = context
        return context

    def _dispatch_skills(self, goal_type: str, pipeline_cfg: Any, phase_outputs: Dict, corr_id: str) -> Dict:
        """Section 2: SKILL DISPATCH."""
        self._notify_ui("on_phase_start", "skill_dispatch")
        t0 = time.time()
        skill_context: Dict = {}
        if self.skills and pipeline_cfg.skill_set:
            active_skills = {k: self.skills[k] for k in pipeline_cfg.skill_set if k in self.skills}
            skill_context = dispatch_skills(goal_type, active_skills, str(self.project_root), corr_id=corr_id)
        phase_outputs["skill_context"] = skill_context
        self._notify_ui("on_phase_complete", "skill_dispatch", (time.monotonic() - t0) * 1000)
        return skill_context

    def _run_reflection_phase(self, verification: Dict, skill_context: Dict, goal_type: str, cycle_id: str, phase_outputs: Dict, corr_id: str) -> Dict:
        """Section 7: REFLECT."""
        reflection_input = {
            "verification": verification,
            "skill_context": skill_context,
            "goal_type": goal_type,
            "corr_id": corr_id,
        }
        self._notify_ui("on_phase_start", "reflect")
        t0 = time.time()
        reflection = self._run_phase("reflect", reflection_input)
        return self._finalize_reflection_result(
            reflection,
            cycle_id,
            phase_outputs,
            corr_id,
            elapsed_ms=(time.time() - t0) * 1000,
        )

    def _finalize_reflection_result(
        self,
        reflection: Dict,
        cycle_id: str,
        phase_outputs: Dict,
        corr_id: str,
        *,
        elapsed_ms: float,
    ) -> Dict:
        self._notify_ui("on_phase_complete", "reflect", elapsed_ms)
        errors = validate_phase_output("reflection", reflection)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "reflection", "errors": errors})
        phase_outputs["reflection"] = reflection
        if reflection.get("summary"):
            self.memory_controller.store(
                MemoryTier.SESSION,
                reflection["summary"],
                metadata={"cycle_id": cycle_id, "type": "reflection"},
                corr_id=corr_id,
            )
        return reflection

    def _execute_post_act_phases(
        self,
        verification: Dict,
        skill_context: Dict,
        goal_type: str,
        cycle_id: str,
        phase_outputs: Dict,
        corr_id: str,
    ) -> Dict:
        """Run quality snapshot and reflection work concurrently."""
        import concurrent.futures
        from core import quality_snapshot

        apply_result = phase_outputs.get("apply_result")
        if not isinstance(apply_result, dict):
            apply_result = {}
        changed_files = list(apply_result.get("applied", []))
        reflection_input = {
            "verification": verification,
            "skill_context": skill_context,
            "goal_type": goal_type,
            "corr_id": corr_id,
        }
        t0 = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="post-act",
        ) as pool:
            snapshot_future = pool.submit(
                quality_snapshot.run_quality_snapshot,
                self.project_root,
                changed_files=changed_files,
            )
            reflection_future = pool.submit(
                self._run_phase,
                "reflect",
                reflection_input,
            )

            quality_snapshot_result = snapshot_future.result()
            phase_outputs["quality_snapshot"] = quality_snapshot_result

            self._notify_ui("on_phase_start", "reflect")
            reflection = reflection_future.result()

        return self._finalize_reflection_result(
            reflection,
            cycle_id,
            phase_outputs,
            corr_id,
            elapsed_ms=(time.time() - t0) * 1000,
        )

    def _finalize_cycle_result(
        self,
        *,
        goal: str,
        cycle_id: str,
        goal_type: Optional[str],
        phase_outputs: Dict,
        stop_reason: Optional[str],
        started_at: float,
        beads: Optional[dict] = None,
    ) -> Dict:
        from core.operator_runtime import build_cycle_summary

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
        entry["cycle_summary"] = build_cycle_summary(entry)
        self.last_cycle_summary = entry["cycle_summary"]
        self.active_cycle_summary = None
        self.current_goal = None
        return entry

    def _record_cycle_outcome(self, goal: str, cycle_id: str, goal_type: str, phase_outputs: Dict, started_at: float):
        """Final persistence and loop notification."""
        verify_status = phase_outputs.get("verification", {}).get("status", "skip")
        passed = verify_status in ("pass", "skip")
        if passed:
            self._consecutive_fails = 0
        elif verify_status == "fail":
            self._consecutive_fails += 1

        if self.adaptive_pipeline:
            try:
                strategy = phase_outputs.get("pipeline_config", {}).get("intensity", "normal")
                self.adaptive_pipeline.record_outcome(goal_type, strategy, passed)
            except Exception as exc:
                log_json("WARN", "adaptive_pipeline_outcome_record_failed", details={"error": str(exc)})

        entry = self._finalize_cycle_result(
            goal=goal,
            cycle_id=cycle_id,
            goal_type=goal_type,
            phase_outputs=phase_outputs,
            stop_reason=None,
            started_at=started_at,
            beads=phase_outputs.get("beads_gate"),
        )
        if self.memory_controller.persistent_store:
            self.memory_controller.persistent_store.append_log(entry)

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

        if self.propagation_engine is not None:
            try:
                self.propagation_engine.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "propagation_engine_error", details={"error": str(exc)})
        return entry

    def _validate_goal(self, goal: str) -> None:
        if not isinstance(goal, str) or not goal.strip():
            raise PreflightError("Goal must be a non-empty string")

    def run_cycle(self, goal: str, dry_run: bool = False) -> Dict:
        """Execute a single complete plan-act-verify cycle for *goal*."""
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        corr_id = cycle_id
        phase_outputs = {}
        phase_outputs["dry_run"] = dry_run
        started_at = time.time()
        goal_type: Optional[str] = None
        self.current_goal = goal
        self.active_cycle_summary = {"cycle_id": cycle_id, "goal": goal, "state": "running"}
        self._notify_ui("on_cycle_start", goal)

        try:
            self._validate_goal(goal)

            goal_type = classify_goal(goal)
            pipeline_cfg = self._configure_pipeline(goal, goal_type, phase_outputs)
            self._handle_capabilities(goal, pipeline_cfg, phase_outputs, dry_run)

            context = self._run_ingest_phase(goal, cycle_id, phase_outputs, corr_id)
            if self.strict_schema and validate_phase_output("context", context):
                 return self._finalize_cycle_result(
                     goal=goal,
                     cycle_id=cycle_id,
                     goal_type=goal_type,
                     phase_outputs=phase_outputs,
                     stop_reason="INVALID_OUTPUT",
                     started_at=started_at,
                 )

            skill_context = self._dispatch_skills(goal_type, pipeline_cfg, phase_outputs, corr_id)

            beads_gate = self._run_beads_gate(goal, goal_type, dry_run, phase_outputs, corr_id)
            self._queue_beads_follow_up_goals(phase_outputs, dry_run)
            if beads_gate is not None:
                return self._finalize_cycle_result(
                    goal=goal,
                    cycle_id=cycle_id,
                    goal_type=goal_type,
                    phase_outputs=phase_outputs,
                    stop_reason=beads_gate.get("stop_reason"),
                    started_at=started_at,
                    beads=beads_gate.get("beads"),
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
                phase_outputs=phase_outputs, dry_run=dry_run, corr_id=corr_id,
            )
            if early_return is not None:
                return early_return

            reflection = self._execute_post_act_phases(
                verification,
                skill_context,
                goal_type,
                cycle_id,
                phase_outputs,
                corr_id,
            )
            if self.strict_schema and validate_phase_output("reflection", reflection):
                return self._finalize_cycle_result(
                    goal=goal,
                    cycle_id=cycle_id,
                    goal_type=goal_type,
                    phase_outputs=phase_outputs,
                    stop_reason="INVALID_OUTPUT",
                    started_at=started_at,
                )

            if self.brain and hasattr(self.brain, "set"):
                try:
                    self.brain.set(f"outcome:{cycle_id}", phase_outputs)
                except Exception:
                    pass

            phase_outputs.pop("_failure_context", None)
            return self._record_cycle_outcome(goal, cycle_id, goal_type, phase_outputs, started_at)
        except ValidationError as exc:
            log_json("WARN", "goal_validation_failed", details={"error": str(exc)}, corr_id=corr_id, phase="ingest", outcome="fail", failure_reason="validation_error")
            return self._finalize_cycle_result(
                goal=goal,
                cycle_id=cycle_id,
                goal_type=goal_type,
                phase_outputs=phase_outputs,
                stop_reason="VALIDATION_ERROR",
                started_at=started_at,
            )
        except PhaseExecutionError as exc:
            log_json("ERROR", "cycle_phase_execution_failed", details={"error": str(exc)}, corr_id=corr_id, phase=getattr(exc, "phase", "unknown"), outcome="fail", failure_reason="phase_execution_error")
            return self._finalize_cycle_result(
                goal=goal,
                cycle_id=cycle_id,
                goal_type=goal_type,
                phase_outputs=phase_outputs,
                stop_reason="PHASE_ERROR",
                started_at=started_at,
            )

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
        for _ in range(max_cycles):
            entry = self.run_cycle(goal, dry_run=dry_run)
            history.append(entry)
            if entry.get("stop_reason"):
                stop_reason = entry["stop_reason"]
                break
            stop_reason = self.policy.evaluate(history, entry.get("phase_outputs", {}).get("verification", {}), started_at=started_at)
            if stop_reason:
                entry["stop_reason"] = stop_reason
                if isinstance(entry.get("cycle_summary"), dict):
                    entry["cycle_summary"]["stop_reason"] = stop_reason
                    self.last_cycle_summary = entry["cycle_summary"]
                break
        return {
            "goal": goal,
            "stop_reason": stop_reason or "MAX_CYCLES",
            "history": history,
        }
