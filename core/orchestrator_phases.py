"""Phase execution methods for the AURA orchestration pipeline."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

from core.logging_utils import log_json
from core.schema import RoutingDecision
from core.phase_result import PhaseResult
from memory.controller import MemoryTier

MAX_SANDBOX_RETRIES = 3


class PhasesMixin:
    """Mixin providing plan/critique/synthesize/act/verify phase methods."""

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
                route=RoutingDecision.ACT.value,
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
        except (AttributeError, KeyError):
            pass
        return "act"

    def _execute_act_verify_attempt(self, goal: str, plan: Dict, task_bundle: Dict, cycle_id: str, phase_outputs: Dict, dry_run: bool):
        """Execute one attempt of act -> sandbox -> apply -> verify.

        When ``n_best_candidates`` > 1 in config, generates multiple code
        variants and uses critic tournament to select the best one.
        """
        import sys

        _orch = sys.modules["core.orchestrator"]

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
        except (AttributeError, TypeError):
            pass

        # Pipeline quality gate: inject critique from P3 Pipeline Coordinator.
        # P3 pre-fetches Dev Suite review and injects it via WebhookGoalRequest.metadata
        # into the cycle_context before this phase runs.  No blocking HTTP call here.
        task_bundle = self._enrich_act_context(task_bundle)

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

        if _orch.validate_phase_output("change_set", act) and self.debugger:
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

            if route == RoutingDecision.PLAN and plan_attempt < max_plan_retries:
                phase_outputs["retry_count"] = phase_outputs.get("retry_count", 0) + 1
                phase_outputs["_failure_context"] = {
                    "failures": verification.get("failures", []),
                    "logs": verification.get("logs", ""),
                    "route": RoutingDecision.PLAN.value,
                    "suggestion": analysis_suggestion,
                    "failure_investigation": failure_investigation,
                    "root_cause_analysis": root_cause_analysis,
                    "remediation_plan": remediation_plan,
                    "investigation": investigation,
                }
                replan_needed = True
                break
            elif route == RoutingDecision.SKIP:
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

            # Fire n8n feedback after each plan attempt so P4/P5 see every iteration
            passed = verification.get("status") in ("pass", "skip")
            try:
                self._notify_n8n_feedback(goal, cycle_id, passed, phase_outputs)
            except Exception:
                pass

            if early_return:
                return verification, early_return
            if not replan_needed:
                break

        return verification, None

    def _run_ingest_phase(self, goal: str, cycle_id: str, phase_outputs: Dict) -> Dict:
        """Section 1: INGEST."""
        import sys

        _orch = sys.modules["core.orchestrator"]

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

        errors = _orch.validate_phase_output("context", context)
        if errors:
            log_json("ERROR", "phase_schema_invalid", details={"phase": "context", "errors": errors})
        phase_outputs["context"] = context
        return context

    def _dispatch_skills(self, goal_type: str, pipeline_cfg: Any, phase_outputs: Dict) -> Dict:
        """Section 2: SKILL DISPATCH."""
        import sys

        _orch = sys.modules["core.orchestrator"]

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
                except (TypeError, KeyError):
                    pass

            skill_context = _orch.dispatch_skills(goal_type, active_skills, str(self.project_root))
        phase_outputs["skill_context"] = skill_context
        self._notify_ui("on_phase_complete", "skill_dispatch", (time.monotonic() - t0) * 1000)
        return skill_context

    def _run_mcp_discovery_phase(self, phase_outputs: Dict) -> Dict:
        """Section 1.5: MCP DISCOVERY."""
        self._notify_ui("on_phase_start", "mcp_discovery")
        t0 = time.time()
        mcp_discovery_output = self._run_phase("mcp_discovery", {"project_root": str(self.project_root)})
        self._notify_ui("on_phase_complete", "mcp_discovery", (time.time() - t0) * 1000)
        phase_outputs["mcp_discovery"] = mcp_discovery_output
        return mcp_discovery_output
