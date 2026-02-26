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
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from core.logging_utils import log_json
from core.policy import Policy
from core.file_tools import _safe_apply_change, OldCodeNotFoundError
from core.schema import validate_phase_output
from core.skill_dispatcher import classify_goal, dispatch_skills
from memory.store import MemoryStore


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
        memory_store: Persistent log store for cycle history and summaries.
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

    def __init__(self, agents: Dict[str, object], memory_store: MemoryStore, policy: Policy = None, project_root: Path = None, strict_schema: bool = False, debugger=None):
        """Initialise the orchestrator with its agents and supporting services.

        Args:
            agents: Dict mapping phase names (``"ingest"``, ``"plan"``, ``"act"``,
                etc.) to agent instances that implement a ``run(input_data)`` method.
            memory_store: Persistent store for cycle logs and summaries.
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
        self.memory_store = memory_store
        self.policy = policy or Policy.from_config({})
        self.project_root = Path(project_root or ".")
        self.strict_schema = strict_schema
        self.debugger = debugger  # Optional DebuggerAgent for auto-recovery

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
            :attr:`memory_store`, ordered from most to least relevant.
            Returns ``[]`` when the memory store is unavailable or empty.
        """
        if not self.memory_store:
            return []
        try:
            summaries = self.memory_store.query("cycle_summaries", limit=200)
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
        """
        changes: List[Dict] = []
        if isinstance(change_set, dict):
            if all(k in change_set for k in ["file_path", "old_code", "new_code"]):
                changes.append(change_set)
            elif "changes" in change_set and isinstance(change_set["changes"], list):
                changes.extend(change_set["changes"])

        applied: List[str] = []
        failed: List[Dict] = []

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
                _safe_apply_change(
                    self.project_root, file_path, old_code, new_code,
                    overwrite_file=overwrite_file,
                )
                applied.append(file_path)
            except OldCodeNotFoundError as exc:
                log_json("ERROR", "old_code_not_found", details={"error": str(exc), "file": file_path})
                failed.append({"file": file_path, "error": str(exc)})
            except Exception as exc:
                log_json("ERROR", "apply_change_failed", details={"error": str(exc), "file": file_path})
                failed.append({"file": file_path, "error": str(exc)})

        return {"applied": applied, "failed": failed}

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

    def run_cycle(self, goal: str, dry_run: bool = False) -> Dict:
        """Execute a single complete plan-act-verify cycle for *goal*.

        Runs all pipeline phases (ingest → skill dispatch → plan → critique →
        synthesize → act loop → sandbox → apply → verify → reflect) in order,
        with adaptive retries.  The cycle result is appended to
        :attr:`memory_store` and returned.

        Args:
            goal: Natural-language description of the coding task to complete.
            dry_run: When ``True``, no files are written to disk.  All other
                phases run normally so that the pipeline output can be inspected
                without side-effects.

        Returns:
            A dict with the following keys:

            * ``"cycle_id"`` (str) — unique hex identifier for this cycle.
            * ``"goal_type"`` (str) — classified goal category from
              :func:`~core.skill_dispatcher.classify_goal`.
            * ``"phase_outputs"`` (dict) — keyed by phase name, each value is
              the raw dict returned by that phase's agent.
            * ``"stop_reason"`` (str | None) — ``None`` on a normal completion,
              or ``"INVALID_OUTPUT"`` if :attr:`strict_schema` is ``True`` and a
              phase violated its schema contract.
        """
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        phase_outputs = {}

        # ── 0. ADAPTIVE PIPELINE CONFIG ──────────────────────────────────────
        # Ask AdaptivePipeline for a context-aware configuration.
        # Falls back to static defaults if CASPA-W is not attached.
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

        # ── 1. INGEST ────────────────────────────────────────────────────────
        context = self._run_phase("ingest", {
            "goal": goal,
            "project_root": str(self.project_root),
            "hints": self._retrieve_hints(goal),
        })
        errors = validate_phase_output("context", context)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "context", "errors": errors})
            if self.strict_schema:
                return {"cycle_id": cycle_id, "phase_outputs": {"context": context}, "stop_reason": "INVALID_OUTPUT"}
        phase_outputs["context"] = context

        # ── 2. SKILL DISPATCH (adaptive skill set) ───────────────────────────
        skill_context: Dict = {}
        if self.skills and pipeline_cfg.skill_set:
            active_skills = {k: self.skills[k] for k in pipeline_cfg.skill_set if k in self.skills}
            skill_context = dispatch_skills(goal_type, active_skills, str(self.project_root))
        phase_outputs["skill_context"] = skill_context

        # ── 3. PLAN ──────────────────────────────────────────────────────────
        max_plan_retries = pipeline_cfg.plan_retries
        plan_attempt = 0
        plan = {}
        critique = {}
        task_bundle = {}

        while plan_attempt < max_plan_retries:
            plan_attempt += 1
            failure_context = phase_outputs.get("_failure_context", {})

            plan = self._run_phase("plan", {
                "goal": goal,
                "memory_snapshot": context.get("memory_summary", ""),
                "similar_past_problems": context.get("hints_summary", ""),
                "known_weaknesses": "",
                "skill_context": skill_context,
                "failure_context": failure_context,
                "extra_context": pipeline_cfg.extra_plan_ctx,
            })
            errors = validate_phase_output("plan", plan)
            if errors:
                log_json("ERROR", "phase_schema_invalid", details={"phase": "plan", "errors": errors})
                if self.strict_schema:
                    return {"cycle_id": cycle_id, "phase_outputs": phase_outputs, "stop_reason": "INVALID_OUTPUT"}
            phase_outputs["plan"] = plan

            # ── 4. CRITIQUE ──────────────────────────────────────────────────
            critique = self._run_phase("critique", {
                "task": goal,
                "plan": plan.get("steps", []),
            })
            errors = validate_phase_output("critique", critique)
            if errors:
                log_json("ERROR", "phase_schema_invalid", details={"phase": "critique", "errors": errors})
                if self.strict_schema:
                    return {"cycle_id": cycle_id, "phase_outputs": phase_outputs, "stop_reason": "INVALID_OUTPUT"}
            phase_outputs["critique"] = critique

            # ── 5. SYNTHESIZE ────────────────────────────────────────────────
            task_bundle = self._run_phase("synthesize", {
                "goal": goal,
                "plan": plan,
                "critique": critique,
            })
            errors = validate_phase_output("task_bundle", task_bundle)
            if errors:
                log_json("ERROR", "phase_schema_invalid", details={"phase": "task_bundle", "errors": errors})
                if self.strict_schema:
                    return {"cycle_id": cycle_id, "phase_outputs": phase_outputs, "stop_reason": "INVALID_OUTPUT"}
            phase_outputs["task_bundle"] = task_bundle

            # ── 6. ACT LOOP ──────────────────────────────────────────────────
            # Retry code generation up to max_act_attempts times.
            # On verification failure, route_failure() decides whether to
            # retry act (recoverable) or break out and re-plan (structural).
            max_act_attempts = pipeline_cfg.max_act_attempts
            act_attempt = 0
            verification: Dict = {}
            replan_needed = False

            while act_attempt < max_act_attempts:
                act_attempt += 1

                act = self._run_phase("act", {
                    "task": goal,
                    "task_bundle": task_bundle,
                    "dry_run": dry_run,
                    "project_root": str(self.project_root),
                    "fix_hints": task_bundle.get("fix_hints", []),
                })
                errors = validate_phase_output("change_set", act)
                if errors:
                    log_json("ERROR", "phase_schema_invalid", details={"phase": "change_set", "errors": errors})
                    if self.debugger is not None:
                        log_json("INFO", "orchestrator_invoking_debugger_on_act_failure")
                        debug_hint = self.debugger.diagnose(
                            error=f"CoderAgent produced invalid change_set: {errors}",
                            context={"goal": goal, "plan": plan, "task_bundle": task_bundle},
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
                        return {"cycle_id": cycle_id, "phase_outputs": phase_outputs, "stop_reason": "INVALID_OUTPUT"}
                phase_outputs["change_set"] = act

                # ── 6a. SANDBOX (pre-apply code execution check) ─────────────
                # Run the generated snippet in isolation before touching the fs.
                # Failures here trigger the same retry logic as verify failures.
                sandbox_result = self._run_phase("sandbox", {
                    "act": act,
                    "dry_run": dry_run,
                    "project_root": str(self.project_root),
                })
                phase_outputs["sandbox"] = sandbox_result or {}
                sandbox_passed = (sandbox_result or {}).get("passed", True)
                if not sandbox_passed and not dry_run:
                    log_json("WARN", "sandbox_pre_apply_failed",
                             details={"summary": (sandbox_result or {}).get("summary", "")})
                    # Treat as recoverable code error — same as verify fail
                    task_bundle["fix_hints"] = [
                        (sandbox_result or {}).get("summary", "sandbox_failed")
                    ]
                    act_attempt += 1
                    continue

                # Apply changes — partial failures are tolerated
                apply_result = self._apply_change_set(act, dry_run=dry_run)
                phase_outputs["apply_result"] = apply_result

                nothing_applied = (
                    not apply_result["applied"] and bool(apply_result["failed"])
                )
                if nothing_applied:
                    # Treat complete apply failure as a recoverable code error
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

                    verification = self._run_phase("verify", {
                        "change_set": act,
                        "dry_run": dry_run,
                        "project_root": str(self.project_root),
                        "tests": tests,
                    })
                    if not verification:
                        verification = {"status": "skip", "failures": [], "logs": ""}

                errors = validate_phase_output("verification", verification)
                if errors:
                    log_json("ERROR", "phase_schema_invalid", details={"phase": "verification", "errors": errors})
                    if self.strict_schema:
                        return {"cycle_id": cycle_id, "phase_outputs": phase_outputs, "stop_reason": "INVALID_OUTPUT"}
                phase_outputs["verification"] = verification

                verify_passed = verification.get("status") in ("pass", "skip")
                if verify_passed:
                    break  # success — exit act loop

                # Failure routing
                route = self._route_failure(verification)
                log_json("INFO", "verification_failed_routing",
                         details={"route": route, "act_attempt": act_attempt,
                                  "plan_attempt": plan_attempt})

                if route == "plan" and plan_attempt < max_plan_retries:
                    # Escalate: break act loop, re-run planning with failure ctx
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
                    # Recoverable — inject failure hints and retry act
                    task_bundle["fix_hints"] = verification.get("failures", [])
                    log_json("INFO", "act_retry",
                             details={"attempt": act_attempt, "hints": task_bundle["fix_hints"]})

            if not replan_needed:
                break  # done — exit plan retry loop

        # ── 7. REFLECT ───────────────────────────────────────────────────────
        reflection = self._run_phase("reflect", {
            "verification": verification,
            "skill_context": skill_context,
            "goal_type": goal_type,
        })
        errors = validate_phase_output("reflection", reflection)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "reflection", "errors": errors})
            if self.strict_schema:
                return {"cycle_id": cycle_id, "phase_outputs": phase_outputs, "stop_reason": "INVALID_OUTPUT"}
        phase_outputs["reflection"] = reflection

        # Remove internal scratchpad key before persisting
        phase_outputs.pop("_failure_context", None)

        # Track consecutive failures for AdaptivePipeline intensity tuning
        verify_status = phase_outputs.get("verification", {}).get("status", "skip")
        if verify_status == "pass":
            self._consecutive_fails = 0
        elif verify_status == "fail":
            self._consecutive_fails += 1

        entry = {
            "cycle_id": cycle_id,
            "goal_type": goal_type,
            "phase_outputs": phase_outputs,
            "stop_reason": None,
        }
        self.memory_store.append_log(entry)

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
