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
    def __init__(self, agents: Dict[str, object], memory_store: MemoryStore, policy: Policy = None, project_root: Path = None, strict_schema: bool = False, debugger=None):
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
        """Register improvement loops to be called after each cycle."""
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
        """Attach CASPA-W components for contextually adaptive self-propagation."""
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
        agent = self.agents.get(name)
        if not agent:
            return {}
        return agent.run(input_data)

    def _apply_change_set(
        self, change_set: Dict, dry_run: bool
    ) -> Dict[str, List]:
        """Apply each change independently.  Returns ``{"applied": [...], "failed": [...]}``.

        Unlike the old all-or-nothing approach, a failure on one file is
        recorded and the loop continues so that other files are still applied.
        Callers check ``result["failed"]`` to decide whether verification
        should run.
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
        """Classify a verification failure and return the re-entry point.

        Returns:
            ``"act"``      — recoverable code-level error; retry generation.
            ``"plan"``     — structural/design error; re-plan from scratch.
            ``"skip"``     — external / environment issue; cannot self-fix.
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
