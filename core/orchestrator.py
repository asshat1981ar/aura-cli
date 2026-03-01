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
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.logging_utils import log_json
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
from memory.controller import memory_controller, MemoryTier

MAX_SANDBOX_RETRIES = 3


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
        self.policy = policy or Policy.from_config({})
        self.project_root = Path(project_root or ".")
        self.strict_schema = strict_schema
        self.debugger = debugger  # Optional DebuggerAgent for auto-recovery
        self.human_gate = HumanGate()
        self.auto_add_capabilities = auto_add_capabilities
        self.auto_queue_missing_capabilities = auto_queue_missing_capabilities
        self.auto_provision_mcp = auto_provision_mcp
        self.auto_start_mcp_servers = auto_start_mcp_servers
        self.goal_queue = goal_queue
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

        # Self-improvement loops (all optional — never block the main loop)
        self._improvement_loops: list = []

        # CASPA-W components — set via attach_caspa() after construction
        self.adaptive_pipeline = None   # AdaptivePipeline
        self.propagation_engine = None  # PropagationEngine
        self.context_graph = None       # ContextGraph
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
        failures = " ".join(str(f) for f in verification.get("failures", []))
        logs = str(verification.get("logs", ""))
        combined = (failures + " " + logs).lower()

        structural_signals = [
            "architecture", "circular", "api_breaking", "breaking_change",
            "design", "interface", "contract",
        ]
        external_signals = [
            "dependency", "network", "env", "environment", "permission",
            "not found", "no module", "import error",
        ]

        if any(s in combined for s in structural_signals):
            return "plan"
        if any(s in combined for s in external_signals):
            return "skip"
        return "act"  # default: code-level fix is worth retrying

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
            self._notify_ui("on_phase_complete", "sandbox", (time.time() - t0_sandbox) * 1000)

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

    def _run_act_loop(
        self,
        goal: str,
        plan: Dict,
        task_bundle: Dict,
        pipeline_cfg,
        cycle_id: str,
        phase_outputs: Dict,
        dry_run: bool,
        plan_attempt: int,
        max_plan_retries: int,
        skill_context: Dict,
    ):
        """Execute the act → sandbox → apply → verify loop with retries and backoff.

        On each failed verify, delegates to :meth:`_route_failure` to decide
        whether to retry the act, escalate to re-planning, or skip.

        Returns:
            Tuple of (verification_dict, replan_needed, early_return_dict_or_None).
            ``early_return_dict_or_None`` is set when strict_schema aborts early.
        """
        max_act_attempts = pipeline_cfg.max_act_attempts
        act_attempt = 0
        verification: Dict = {}
        replan_needed = False

        while act_attempt < max_act_attempts:
            act_attempt += 1

            # Exponential backoff between retries (not on first attempt)
            if act_attempt > 1:
                backoff = min(2 ** (act_attempt - 2), 16)
                log_json("INFO", "act_retry_backoff",
                         details={"attempt": act_attempt, "backoff_s": backoff})
                time.sleep(backoff)

            self._notify_ui("on_phase_start", "act")
            t0_act = time.time()
            act = self._run_phase("act", {
                "task": goal,
                "task_bundle": task_bundle,
                "dry_run": dry_run,
                "project_root": str(self.project_root),
                "fix_hints": task_bundle.get("fix_hints", []),
            })
            self._notify_ui("on_phase_complete", "act", (time.time() - t0_act) * 1000)

            errors = validate_phase_output("change_set", act)
            if errors:
                log_json("ERROR", "phase_schema_invalid",
                         details={"phase": "change_set", "errors": errors})
                if self.debugger is not None:
                    log_json("INFO", "orchestrator_invoking_debugger_on_act_failure")
                    debug_hint = self.debugger.diagnose(
                        error=f"CoderAgent produced invalid change_set: {errors}",
                        context={"goal": goal, "plan": plan,
                                 "task_bundle": task_bundle},
                    )
                    act = self._run_phase("act", {
                        "task": goal,
                        "task_bundle": task_bundle,
                        "dry_run": dry_run,
                        "project_root": str(self.project_root),
                        "debug_hint": debug_hint,
                    })
                    errors = validate_phase_output("change_set", act)
                    if errors:
                        log_json("WARN", "orchestrator_act_retry_still_invalid")
                if errors and self.strict_schema:
                    return verification, replan_needed, {
                        "cycle_id": cycle_id,
                        "phase_outputs": phase_outputs,
                        "stop_reason": "INVALID_OUTPUT",
                    }
            phase_outputs["change_set"] = act

            # ── SANDBOX ───────────────────────────────────────────────────────
            act, _sandbox_passed, extra_act_uses = self._run_sandbox_loop(
                goal, act, task_bundle, dry_run, phase_outputs
            )
            act_attempt += extra_act_uses

            # ── APPLY ─────────────────────────────────────────────────────────
            self._notify_ui("on_phase_start", "apply")
            t0_apply = time.time()
            apply_result = self._apply_change_set(act, dry_run=dry_run)
            self._notify_ui("on_phase_complete", "apply", (time.time() - t0_apply) * 1000)
            phase_outputs["apply_result"] = apply_result

            nothing_applied = (
                not apply_result["applied"] and bool(apply_result["failed"])
            )
            if nothing_applied:
                verification = {
                    "status": "fail",
                    "failures": [f["error"] for f in apply_result["failed"]],
                    "logs": "apply_change_failed — no files written",
                }
            else:
                tests = []
                if isinstance(task_bundle, dict):
                    tasks = task_bundle.get("tasks", [])
                    if tasks:
                        tests = tasks[0].get("tests", []) or []

                self._notify_ui("on_phase_start", "verify")
                t0_verify = time.time()
                verification = self._run_phase("verify", {
                    "change_set": act,
                    "dry_run": dry_run,
                    "project_root": str(self.project_root),
                    "tests": tests,
                })
                self._notify_ui(
                    "on_phase_complete", "verify",
                    (time.time() - t0_verify) * 1000,
                    success=(verification.get("status") in ("pass", "skip")),
                )
                if not verification:
                    verification = {"status": "skip", "failures": [], "logs": ""}

            errors = validate_phase_output("verification", verification)
            if errors:
                log_json("ERROR", "phase_schema_invalid",
                         details={"phase": "verification", "errors": errors})
                if self.strict_schema:
                    return verification, replan_needed, {
                        "cycle_id": cycle_id,
                        "phase_outputs": phase_outputs,
                        "stop_reason": "INVALID_OUTPUT",
                    }
            phase_outputs["verification"] = verification

            verify_passed = verification.get("status") in ("pass", "skip")
            if verify_passed:
                # ── HUMAN GATE ────────────────────────────────────────────────
                blocked, gate_reason = self.human_gate.should_block(
                    verification, skill_context
                )
                if blocked:
                    approved = self.human_gate.request_approval(
                        gate_reason,
                        {"cycle_id": cycle_id, "goal": goal,
                         "changes": len((act or {}).get("changes", []))},
                    )
                    if not approved:
                        log_json("WARN", "human_gate_apply_skipped",
                                 details={"reason": gate_reason})
                        phase_outputs["human_gate"] = {
                            "blocked": True, "reason": gate_reason, "approved": False
                        }
                        break
                    phase_outputs["human_gate"] = {
                        "blocked": True, "reason": gate_reason, "approved": True
                    }
                break  # success — exit act loop

            # ── Failure routing ───────────────────────────────────────────────
            # On verify fail, restore only the files mutated by this loop attempt.
            if apply_result.get("applied") and not dry_run:
                self._restore_applied_changes(apply_result.get("snapshots", []))

            route = self._route_failure(verification)
            log_json("INFO", "verification_failed_routing",
                     details={"route": route, "act_attempt": act_attempt,
                              "plan_attempt": plan_attempt})

            if route == "plan" and plan_attempt < max_plan_retries:
                phase_outputs["_failure_context"] = {
                    "failures": verification.get("failures", []),
                    "logs": verification.get("logs", ""),
                    "route": "plan",
                }
                replan_needed = True
                break
            elif route == "skip":
                log_json("WARN", "verification_failure_external_skip")
                break
            else:
                task_bundle["fix_hints"] = verification.get("failures", [])
                log_json("INFO", "act_retry",
                         details={"attempt": act_attempt,
                                  "hints": task_bundle["fix_hints"]})

        return verification, replan_needed, None

    def _run_plan_loop(
        self,
        goal: str,
        context: Dict,
        skill_context: Dict,
        pipeline_cfg,
        cycle_id: str,
        phase_outputs: Dict,
        dry_run: bool,
    ):
        """Execute the plan → critique → synthesize → act loop with retries.

        Returns:
            Tuple of (verification_dict, early_return_dict_or_None).
            ``early_return_dict_or_None`` is non-None when strict_schema aborts.
        """
        max_plan_retries = pipeline_cfg.plan_retries
        plan_attempt = 0
        verification: Dict = {}

        while plan_attempt < max_plan_retries:
            plan_attempt += 1
            failure_context = phase_outputs.get("_failure_context", {})

            # ── PLAN ──────────────────────────────────────────────────────────
            self._notify_ui("on_phase_start", "plan")
            t0_plan = time.time()
            plan = self._run_phase("plan", {
                "goal": goal,
                "memory_snapshot": context.get("memory_summary", ""),
                "similar_past_problems": context.get("hints_summary", ""),
                "known_weaknesses": "",
                "skill_context": skill_context,
                "failure_context": failure_context,
                "extra_context": pipeline_cfg.extra_plan_ctx,
            })
            self._notify_ui("on_phase_complete", "plan", (time.time() - t0_plan) * 1000)

            errors = validate_phase_output("plan", plan)
            if errors:
                log_json("ERROR", "phase_schema_invalid",
                         details={"phase": "plan", "errors": errors})
                if self.strict_schema:
                    return verification, {
                        "cycle_id": cycle_id,
                        "phase_outputs": phase_outputs,
                        "stop_reason": "INVALID_OUTPUT",
                    }
            phase_outputs["plan"] = plan

            # ── CRITIQUE ──────────────────────────────────────────────────────
            self._notify_ui("on_phase_start", "critique")
            t0_crit = time.time()
            critique = self._run_phase("critique", {
                "task": goal,
                "plan": plan.get("steps", []),
            })
            self._notify_ui("on_phase_complete", "critique", (time.time() - t0_crit) * 1000)

            errors = validate_phase_output("critique", critique)
            if errors:
                log_json("ERROR", "phase_schema_invalid",
                         details={"phase": "critique", "errors": errors})
                if self.strict_schema:
                    return verification, {
                        "cycle_id": cycle_id,
                        "phase_outputs": phase_outputs,
                        "stop_reason": "INVALID_OUTPUT",
                    }
            phase_outputs["critique"] = critique

            # ── SYNTHESIZE ────────────────────────────────────────────────────
            self._notify_ui("on_phase_start", "synthesize")
            t0_synth = time.time()
            task_bundle = self._run_phase("synthesize", {
                "goal": goal,
                "plan": plan,
                "critique": critique,
            })
            self._notify_ui("on_phase_complete", "synthesize", (time.time() - t0_synth) * 1000)

            errors = validate_phase_output("task_bundle", task_bundle)
            if errors:
                log_json("ERROR", "phase_schema_invalid",
                         details={"phase": "task_bundle", "errors": errors})
                if self.strict_schema:
                    return verification, {
                        "cycle_id": cycle_id,
                        "phase_outputs": phase_outputs,
                        "stop_reason": "INVALID_OUTPUT",
                    }
            phase_outputs["task_bundle"] = task_bundle

            # ── ACT LOOP ──────────────────────────────────────────────────────
            verification, replan_needed, early_return = self._run_act_loop(
                goal=goal,
                plan=plan,
                task_bundle=task_bundle,
                pipeline_cfg=pipeline_cfg,
                cycle_id=cycle_id,
                phase_outputs=phase_outputs,
                dry_run=dry_run,
                plan_attempt=plan_attempt,
                max_plan_retries=max_plan_retries,
                skill_context=skill_context,
            )
            if early_return is not None:
                return verification, early_return

            if not replan_needed:
                break  # done — exit plan retry loop

        return verification, None

    def run_cycle(self, goal: str, dry_run: bool = False) -> Dict:
        """Execute a single complete plan-act-verify cycle for *goal*."""
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        phase_outputs = {}
        self._notify_ui("on_cycle_start", goal)

        # ── 0. ADAPTIVE PIPELINE CONFIG ──────────────────────────────────────
        goal_type = classify_goal(goal)
        if self.adaptive_pipeline:
            pipeline_cfg = self.adaptive_pipeline.configure(
                goal, goal_type,
                consecutive_fails=self._consecutive_fails,
                past_failures=list(
                    phase_outputs.get("_failure_context", {}).get("failures", [])
                ),
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

        capability_plan = {
            "matched_capabilities": [],
            "recommended_skills": [],
            "missing_skills": [],
            "mcp_tools": [],
            "provisioning_actions": [],
        }
        capability_goal_queue = {"attempted": False, "queued": [], "skipped": [], "queue_strategy": None}
        capability_provisioning = {"attempted": False, "results": []}

        if self.auto_add_capabilities:
            capability_plan = analyze_capability_needs(
                goal,
                available_skills=self.skills.keys(),
                active_skills=pipeline_cfg.skill_set,
            )
            if capability_plan["recommended_skills"]:
                pipeline_cfg.skill_set = list(
                    dict.fromkeys(list(pipeline_cfg.skill_set) + list(capability_plan["recommended_skills"]))
                )
                phase_outputs["pipeline_config"]["skills"] = pipeline_cfg.skill_set
            phase_outputs["capability_plan"] = capability_plan
            capability_goal_queue = queue_missing_capability_goals(
                goal_queue=self.goal_queue,
                missing_skills=capability_plan["missing_skills"],
                goal=goal,
                enabled=self.auto_queue_missing_capabilities,
                dry_run=dry_run,
            )
            if capability_goal_queue["queued"] or capability_goal_queue["skipped"]:
                phase_outputs["capability_goal_queue"] = capability_goal_queue
            if capability_plan["provisioning_actions"]:
                capability_provisioning = provision_capability_actions(
                    project_root=self.project_root,
                    provisioning_actions=capability_plan["provisioning_actions"],
                    auto_provision=self.auto_provision_mcp,
                    start_servers=self.auto_start_mcp_servers,
                    dry_run=dry_run,
                )
                phase_outputs["capability_provisioning"] = capability_provisioning

        self.last_capability_plan = capability_plan
        self.last_capability_goal_queue = capability_goal_queue
        self.last_capability_provisioning = capability_provisioning
        self.last_capability_status = record_capability_status(
            project_root=self.project_root,
            goal=goal,
            capability_plan=capability_plan,
            capability_goal_queue=capability_goal_queue,
            capability_provisioning=capability_provisioning,
            goal_queue=self.goal_queue,
        )

        # ── 1. INGEST ────────────────────────────────────────────────────────
        self._notify_ui("on_phase_start", "ingest")
        t0_ingest = time.time()
        
        # Retrieve context from memory tiers
        working_memory = self.memory_controller.retrieve(MemoryTier.WORKING)
        session_memory = self.memory_controller.retrieve(MemoryTier.SESSION)
        
        context = self._run_phase("ingest", {
            "goal": goal,
            "project_root": str(self.project_root),
            "hints": self._retrieve_hints(goal),
            "working_memory": working_memory,
            "session_memory": session_memory,
        })
        self._notify_ui("on_phase_complete", "ingest", (time.time() - t0_ingest) * 1000)

        if "bundle" in context:
            self._notify_ui("on_context_assembled", context["bundle"])

        errors = validate_phase_output("context", context)
        if errors:
            log_json("ERROR", "phase_schema_invalid",
                     details={"phase": "context", "errors": errors})
            if self.strict_schema:
                return {
                    "cycle_id": cycle_id,
                    "phase_outputs": {"context": context},
                    "stop_reason": "INVALID_OUTPUT",
                }
        phase_outputs["context"] = context

        # ── 2. SKILL DISPATCH (adaptive skill set) ───────────────────────────
        self._notify_ui("on_phase_start", "skill_dispatch")
        t0_skills = time.time()
        skill_context: Dict = {}
        if self.skills and pipeline_cfg.skill_set:
            active_skills = {
                k: self.skills[k] for k in pipeline_cfg.skill_set if k in self.skills
            }
            skill_context = dispatch_skills(goal_type, active_skills, str(self.project_root))
        phase_outputs["skill_context"] = skill_context
        self._notify_ui("on_phase_complete", "skill_dispatch",
                        (time.time() - t0_skills) * 1000)

        # ── 3-8. PLAN → CRITIQUE → SYNTHESIZE → ACT → VERIFY (with retries) ─
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

        # ── 7. REFLECT ───────────────────────────────────────────────────────
        self._notify_ui("on_phase_start", "reflect")
        t0_reflect = time.time()
        reflection = self._run_phase("reflect", {
            "verification": verification,
            "skill_context": skill_context,
            "goal_type": goal_type,
        })
        self._notify_ui("on_phase_complete", "reflect", (time.time() - t0_reflect) * 1000)
        errors = validate_phase_output("reflection", reflection)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "reflection", "errors": errors})
            if self.strict_schema:
                return {"cycle_id": cycle_id, "phase_outputs": phase_outputs, "stop_reason": "INVALID_OUTPUT"}
        phase_outputs["reflection"] = reflection
        
        # R2: Unified Control Plane - Store reflection in SESSION memory for cross-cycle context
        if reflection.get("summary"):
            self.memory_controller.store(
                MemoryTier.SESSION, 
                reflection["summary"], 
                metadata={"cycle_id": cycle_id, "type": "reflection"}
            )

        # Remove internal scratchpad key before persisting
        phase_outputs.pop("_failure_context", None)

        # Track consecutive failures for AdaptivePipeline intensity tuning
        verify_status = phase_outputs.get("verification", {}).get("status", "skip")
        passed = verify_status in ("pass", "skip")
        if passed:
            self._consecutive_fails = 0
        elif verify_status == "fail":
            self._consecutive_fails += 1

        if self.adaptive_pipeline:
            try:
                # Record strategy success/failure for learning
                strategy = phase_outputs.get("pipeline_config", {}).get("intensity", "normal")
                self.adaptive_pipeline.record_outcome(goal_type, strategy, passed)
            except Exception as exc:
                log_json("WARN", "adaptive_pipeline_outcome_record_failed", details={"error": str(exc)})

        entry = {
            "cycle_id": cycle_id,
            "goal_type": goal_type,
            "phase_outputs": phase_outputs,
            "stop_reason": None,
        }
        if self.memory_controller.persistent_store:
            self.memory_controller.persistent_store.append_log(entry)

        # ── CASPA-W: Update context graph ───────────────────────────────────
        if self.context_graph is not None:
            try:
                self.context_graph.update_from_cycle(entry)
            except Exception as exc:
                log_json("WARN", "context_graph_update_failed",
                         details={"error": str(exc)})

        # Fire all registered self-improvement loops (errors are swallowed)
        for loop in self._improvement_loops:
            try:
                loop.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "improvement_loop_error",
                         details={"loop": type(loop).__name__, "error": str(exc)})

        # ── CASPA-W: Propagation engine fires after all loops ────────────────
        if self.propagation_engine is not None:
            try:
                self.propagation_engine.on_cycle_complete(entry)
            except Exception as exc:
                log_json("WARN", "propagation_engine_error",
                         details={"error": str(exc)})

        return entry

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
                break
        return {
            "goal": goal,
            "stop_reason": stop_reason or "MAX_CYCLES",
            "history": history,
        }
