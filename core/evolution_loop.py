import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from core.logging_utils import log_json  # Import the new logging utility
from core.file_tools import _aura_safe_loads  # Import _aura_safe_loads
from core.capability_manager import analyze_capability_needs
from core.experiment_tracker import ExperimentTracker, MetricsCollector
from core.evolution_prompts import EVOLUTION_HYPOTHESIS_PROMPT, EVOLUTION_TASK_DECOMPOSITION_PROMPT, EVOLUTION_MUTATION_PROMPT


@dataclass(frozen=True)
class InnovationProposal:
    proposal_id: str
    title: str
    category: str
    goal: str
    rationale: str
    evidence: List[str]
    smallest_surface: str
    expected_value: str
    risk_level: str
    verification_cost: str
    recommended_action: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "category": self.category,
            "goal": self.goal,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "smallest_surface": self.smallest_surface,
            "expected_value": self.expected_value,
            "risk_level": self.risk_level,
            "verification_cost": self.verification_cost,
            "recommended_action": self.recommended_action,
        }


class EvolutionLoop:
    """
    Orchestrates an advanced evolutionary loop for autonomous development,
    involving multiple agents (Planner, Coder, Critic, Mutator) to
    hypothesize, decompose, implement, critique, and mutate code,
    driving continuous self-improvement of the AURA system.
    """

    def __init__(
        self,
        planner,
        coder,
        critic,
        brain,
        vector_store,
        git_tools,
        mutator,
        improvement_service=None,
        goal_queue=None,
        orchestrator=None,
        project_root=None,
        skills=None,
        auto_execute_queued: bool = True,
        innovation_goal_limit: int = 2,
    ):
        """
        Initializes the EvolutionLoop with instances of various agents and core tools.

        Args:
            planner: An instance of the PlannerAgent.
            coder: An instance of the CoderAgent.
            critic: An instance of the CriticAgent.
            brain: An instance of the system's memory (Brain).
            vector_store: An instance of the VectorStore for semantic search.
            git_tools: An instance of GitTools for repository operations.
            mutator: An instance of the MutatorAgent for applying changes.
            improvement_service: Optional service for evaluating meta-improvements.
        """
        self.planner = planner
        self.coder = coder
        self.critic = critic
        self.brain = brain
        self.vector = vector_store
        self.git = git_tools
        self.mutator = mutator
        self.improvement_service = improvement_service
        self.goal_queue = goal_queue
        self.orchestrator = orchestrator
        self.project_root = Path(project_root or getattr(git_tools, "repo_path", ".") or ".")
        self.skills = skills or {}
        self.auto_execute_queued = auto_execute_queued
        self.innovation_goal_limit = max(1, int(innovation_goal_limit))
        self._cycle_count = 0
        self.TRIGGER_EVERY_N = 20

        # Experiment tracking (Karpathy-style measure→keep/discard)
        memory_dir = Path(__file__).parent.parent / "memory"
        metrics_collector = MetricsCollector(memory_dir)
        self.experiment_tracker = ExperimentTracker(
            memory_dir / "experiments.jsonl",
            metrics_collector,
        )

    def _available_skill_names(self) -> List[str]:
        skill_names = list(self.skills.keys())
        if skill_names:
            return skill_names
        try:
            from agents.skills.registry import all_skills

            return list(all_skills(brain=self.brain, model=getattr(self.planner, "model", None)).keys())
        except (OSError, IOError, ValueError):
            return []

    def _safe_skill_run(self, skill_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        skill = self.skills.get(skill_name)
        if skill is None:
            try:
                from agents.skills.registry import all_skills

                skill = all_skills(brain=self.brain, model=getattr(self.planner, "model", None)).get(skill_name)
            except (OSError, IOError, ValueError):
                skill = None
        if skill is None:
            return {"error": f"skill_unavailable:{skill_name}"}
        try:
            result = skill.run(inputs)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            return {"error": str(exc)}

    def _load_mcp_summary(self) -> Dict[str, Any]:
        config_path = self.project_root / ".mcp.json"
        if not config_path.exists():
            return {"config_path": str(config_path), "server_count": 0, "servers": []}
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"config_path": str(config_path), "error": str(exc), "server_count": 0, "servers": []}
        servers = sorted((payload.get("mcpServers") or {}).keys())
        return {"config_path": str(config_path), "server_count": len(servers), "servers": servers}

    def _architecture_explorer(self) -> Dict[str, Any]:
        project_root = str(self.project_root)
        structural = self._safe_skill_run("structural_analyzer", {"project_root": project_root, "report_coverage": True})
        debt = self._safe_skill_run("tech_debt_quantifier", {"project_root": project_root})
        clones = self._safe_skill_run("code_clone_detector", {"project_root": project_root})
        return {
            "role": "architecture_explorer",
            "structural": structural,
            "tech_debt": debt,
            "clones": clones,
        }

    def _capability_researcher(self, goal: str) -> Dict[str, Any]:
        capability_plan = analyze_capability_needs(
            goal,
            available_skills=self._available_skill_names(),
            active_skills=(),
        )
        return {
            "role": "capability_researcher",
            "capability_plan": capability_plan,
            "mcp": self._load_mcp_summary(),
            "skill_count": len(self._available_skill_names()),
        }

    def _verification_reviewer(self, proposals: List[InnovationProposal], architecture: Dict[str, Any]) -> Dict[str, Any]:
        hotspots = architecture.get("structural", {}).get("hotspots", []) if isinstance(architecture, dict) else []
        reviewed = []
        hotspot_files = {item.get("file") for item in hotspots if isinstance(item, dict)}
        for proposal in proposals:
            residual_risk = "medium"
            if proposal.category in {"skill", "mcp"}:
                residual_risk = "low"
            if any(file_name and file_name in proposal.smallest_surface for file_name in hotspot_files):
                residual_risk = "high"
            reviewed.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "verification_target": proposal.goal,
                    "residual_risk": residual_risk,
                    "recommended_test_scope": proposal.verification_cost,
                }
            )
        return {"role": "verification_reviewer", "reviews": reviewed}

    def _summarize_subagents(self, *reports: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "role": report.get("role"),
                "status": "ok" if "error" not in report else "error",
            }
            for report in reports
        ]

    def _build_innovation_proposals(self, goal: str, architecture: Dict[str, Any], capability: Dict[str, Any]) -> List[InnovationProposal]:
        capability_plan = capability.get("capability_plan", {})
        mcp_summary = capability.get("mcp", {})
        proposals: List[InnovationProposal] = []

        for skill_name in capability_plan.get("missing_skills", []):
            proposals.append(
                InnovationProposal(
                    proposal_id=f"skill:{skill_name}",
                    title=f"Add AURA skill '{skill_name}'",
                    category="skill",
                    goal=f"Add AURA skill '{skill_name}' to support goals related to: {goal}",
                    rationale=f"The goal maps to a missing capability '{skill_name}' that is not currently installed.",
                    evidence=[f"Missing skill from capability analysis: {skill_name}"],
                    smallest_surface="agents/skills + registry + targeted tests",
                    expected_value="high",
                    risk_level="medium",
                    verification_cost="targeted unit tests plus registry smoke test",
                    recommended_action="queue",
                )
            )

        for skill_name in capability_plan.get("recommended_skills", []):
            proposals.append(
                InnovationProposal(
                    proposal_id=f"enable:{skill_name}",
                    title=f"Deepen integration for '{skill_name}'",
                    category="capability",
                    goal=f"Improve AURA's use of the '{skill_name}' capability for goals related to: {goal}",
                    rationale=f"The capability matcher already recommends '{skill_name}', but the innovation workflow should convert that into a concrete integration improvement.",
                    evidence=[f"Recommended skill from capability analysis: {skill_name}"],
                    smallest_surface="capability manager or orchestrator routing",
                    expected_value="medium",
                    risk_level="low",
                    verification_cost="targeted capability and orchestrator tests",
                    recommended_action="queue",
                )
            )

        for action in capability_plan.get("provisioning_actions", []):
            action_name = action.get("action", "unknown_action")
            proposals.append(
                InnovationProposal(
                    proposal_id=f"mcp:{action_name}",
                    title=f"Operationalize provisioning action '{action_name}'",
                    category="mcp",
                    goal=f"Improve MCP provisioning flow for {action_name} in support of: {goal}",
                    rationale=action.get("reason", "Capability analysis identified an MCP provisioning opportunity."),
                    evidence=[f"Provisioning action: {action_name}", f"MCP servers configured: {mcp_summary.get('server_count', 0)}"],
                    smallest_surface="capability provisioning and documentation",
                    expected_value="high",
                    risk_level="medium",
                    verification_cost="targeted provisioning and CLI workflow tests",
                    recommended_action="queue",
                )
            )

        structural = architecture.get("structural", {})
        hotspots = structural.get("hotspots", []) if isinstance(structural, dict) else []
        for hotspot in hotspots[:2]:
            file_name = hotspot.get("file", "<unknown>")
            proposals.append(
                InnovationProposal(
                    proposal_id=f"hotspot:{file_name}",
                    title=f"Reduce structural hotspot in {file_name}",
                    category="orchestration",
                    goal=f"Refactor structural hotspot '{file_name}' to improve long-term agent capability and maintainability",
                    rationale="High-centrality, high-complexity files reduce safe automation and raise the cost of future capability work.",
                    evidence=[f"Hotspot file: {file_name}", f"Risk level: {hotspot.get('risk_level', 'unknown')}"],
                    smallest_surface=file_name,
                    expected_value="medium",
                    risk_level=str(hotspot.get("risk_level", "medium")).lower(),
                    verification_cost="focused unit tests on the touched subsystem",
                    recommended_action="queue",
                )
            )

        debt = architecture.get("tech_debt", {})
        if isinstance(debt, dict) and debt.get("debt_score", 100) < 80:
            proposals.append(
                InnovationProposal(
                    proposal_id="verification:debt",
                    title="Reduce concentrated technical debt in core automation paths",
                    category="verification",
                    goal="Improve tests and maintainability around concentrated technical-debt hotspots",
                    rationale="Debt hotspots degrade confidence in autonomous refactors and future skill generation.",
                    evidence=[debt.get("summary", "Technical debt summary unavailable.")],
                    smallest_surface="top debt items only",
                    expected_value="medium",
                    risk_level="medium",
                    verification_cost="targeted regression tests in debt-heavy modules",
                    recommended_action="queue",
                )
            )

        clones = architecture.get("clones", {})
        if isinstance(clones, dict) and clones.get("clone_count", 0) > 0:
            proposals.append(
                InnovationProposal(
                    proposal_id="refactor:clones",
                    title="Consolidate duplicated code paths found by clone analysis",
                    category="developer-surface",
                    goal="Reduce duplicate code in the current codebase to improve safe reuse and future agent edits",
                    rationale="Duplicate code expands the change surface for future autonomous implementation.",
                    evidence=list(clones.get("consolidation_suggestions", [])[:2]) or [f"Clone count: {clones.get('clone_count', 0)}"],
                    smallest_surface="top duplicate pair only",
                    expected_value="medium",
                    risk_level="medium",
                    verification_cost="narrow unit tests across both duplicated call paths",
                    recommended_action="queue",
                )
            )

        return proposals

    def _select_proposals(self, proposals: List[InnovationProposal], *, focus: str, proposal_limit: int) -> List[InnovationProposal]:
        focus_priority = {
            "capability": {
                "skill": 0,
                "mcp": 1,
                "capability": 2,
                "verification": 3,
                "orchestration": 4,
                "developer-surface": 5,
            },
            "quality": {
                "verification": 0,
                "orchestration": 1,
                "capability": 2,
                "skill": 3,
                "mcp": 4,
                "developer-surface": 5,
            },
            "throughput": {
                "developer-surface": 0,
                "capability": 1,
                "skill": 2,
                "mcp": 3,
                "verification": 4,
                "orchestration": 5,
            },
            "research": {
                "capability": 0,
                "verification": 1,
                "mcp": 2,
                "skill": 3,
                "orchestration": 4,
                "developer-surface": 5,
            },
        }
        category_priority = focus_priority.get(focus, focus_priority["capability"])
        risk_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        ranked = sorted(
            proposals,
            key=lambda item: (
                category_priority.get(item.category, 99),
                risk_priority.get(item.risk_level, 99),
                item.title,
            ),
        )
        return ranked[: max(1, int(proposal_limit))]

    def _queue_selected_goals(self, selected: List[InnovationProposal], *, dry_run: bool) -> Dict[str, Any]:
        if not selected:
            return {"attempted": False, "queued": [], "skipped": [], "queue_strategy": None}
        goals = [proposal.goal for proposal in selected]
        if dry_run:
            return {
                "attempted": False,
                "queued": [],
                "skipped": [{"goal": goal, "reason": "dry_run"} for goal in goals],
                "queue_strategy": None,
            }
        if self.goal_queue is None:
            return {
                "attempted": False,
                "queued": [],
                "skipped": [{"goal": goal, "reason": "goal_queue_unavailable"} for goal in goals],
                "queue_strategy": None,
            }

        existing = set(getattr(self.goal_queue, "queue", []) or [])
        new_goals = [goal for goal in goals if goal not in existing]
        skipped = [{"goal": goal, "reason": "already_queued"} for goal in goals if goal in existing]
        if not new_goals:
            return {"attempted": True, "queued": [], "skipped": skipped, "queue_strategy": None}
        if hasattr(self.goal_queue, "prepend_batch"):
            self.goal_queue.prepend_batch(new_goals)
            strategy = "prepend"
        elif hasattr(self.goal_queue, "batch_add"):
            self.goal_queue.batch_add(new_goals)
            strategy = "append"
        else:
            for goal in new_goals:
                self.goal_queue.add(goal)
            strategy = "append"
        return {
            "attempted": True,
            "queued": new_goals,
            "skipped": skipped,
            "queue_strategy": strategy,
        }

    def _execute_selected_goals(self, goals: List[str], *, dry_run: bool, execution_limit: int) -> Dict[str, Any]:
        if not goals:
            return {"attempted": False, "executed": []}
        if dry_run:
            return {
                "attempted": False,
                "executed": [{"goal": goal, "status": "planned"} for goal in goals],
            }
        if self.orchestrator is None:
            return {
                "attempted": False,
                "executed": [{"goal": goal, "status": "orchestrator_unavailable"} for goal in goals],
            }
        executed = []
        for goal in goals[: max(1, int(execution_limit))]:
            result = self.orchestrator.run_loop(goal, max_cycles=1, dry_run=False)
            executed.append(
                {
                    "goal": goal,
                    "status": result.get("stop_reason", "unknown"),
                    "history_length": len(result.get("history", [])),
                }
            )
        return {"attempted": True, "executed": executed}

    def _run_innovation_workflow(
        self,
        goal: str,
        *,
        execute_queued: bool,
        dry_run: bool,
        proposal_limit: int,
        focus: str,
    ) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            architecture_future = pool.submit(self._architecture_explorer)
            capability_future = pool.submit(self._capability_researcher, goal)
            architecture = architecture_future.result()
            capability = capability_future.result()

        proposals = self._build_innovation_proposals(goal, architecture, capability)
        selected = self._select_proposals(proposals, focus=focus, proposal_limit=proposal_limit)
        verification = self._verification_reviewer(selected, architecture)
        queue_result = self._queue_selected_goals(selected, dry_run=dry_run)
        execution_result = self._execute_selected_goals(queue_result.get("queued", []), dry_run=dry_run, execution_limit=proposal_limit) if execute_queued else {"attempted": False, "executed": []}

        return {
            "workflow": "innovation",
            "focus": focus,
            "subagents": self._summarize_subagents(architecture, capability, verification),
            "analysis": {
                "architecture": architecture,
                "capability": capability,
                "verification": verification,
            },
            "proposals": [proposal.as_dict() for proposal in proposals],
            "selected_proposals": [proposal.as_dict() for proposal in selected],
            "queue": queue_result,
            "implementation": execution_result,
        }

    def _get_mutation_response(self, prompt: str, memory_snapshot: str, similar_past_problems: str, known_weaknesses: str) -> str:
        """Request a raw mutation plan instead of forcing planner list parsing."""
        planner_respond = getattr(self.planner, "_respond", None)
        if callable(planner_respond):
            return planner_respond(prompt)

        planner_model = getattr(self.planner, "model", None)
        responder = getattr(planner_model, "respond_for_role", None)
        if callable(responder):
            return responder("analysis", prompt)
        if planner_model is not None and callable(getattr(planner_model, "respond", None)):
            return planner_model.respond(prompt)

        # Last-resort compatibility path for non-standard planner stubs.
        fallback = self.planner.plan(
            goal=prompt,
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses,
        )
        return "\n".join(fallback) if isinstance(fallback, list) else str(fallback)

    def _mutation_plan_to_dsl(self, mutation_plan: dict) -> str:
        """Convert a structured mutation plan into the MutatorAgent DSL."""
        mutations = mutation_plan.get("mutations", [])
        blocks: list[str] = []
        for mutation in mutations:
            if not isinstance(mutation, dict):
                continue
            file_path = str(mutation.get("file_path") or "").strip()
            new_content = mutation.get("new_content")
            if not file_path or new_content is None:
                continue
            old_content = mutation.get("old_content")
            if old_content is not None:
                block = "\n".join(
                    [
                        f"REPLACE_IN_FILE {file_path}",
                        "---OLD_CONTENT_START---",
                        str(old_content),
                        "---OLD_CONTENT_END---",
                        "---NEW_CONTENT_START---",
                        str(new_content),
                        "---NEW_CONTENT_END---",
                    ]
                )
            else:
                block = "\n".join(
                    [
                        f"ADD_FILE {file_path}",
                        str(new_content),
                    ]
                )
            blocks.append(block)
        return "\n".join(blocks)

    def _normalize_mutation_plan(self, raw_response: str) -> tuple[dict | None, str]:
        """Return a parsed mutation plan and the corresponding mutator DSL."""
        parsed = _aura_safe_loads(raw_response, "evolution_mutation_plan")
        if not isinstance(parsed, dict):
            raise ValueError("Mutation plan must be a JSON object.")
        mutations = parsed.get("mutations")
        if not isinstance(mutations, list):
            raise ValueError("Mutation plan must include a `mutations` list.")
        mutation_dsl = self._mutation_plan_to_dsl(parsed)
        if not mutation_dsl.strip():
            raise ValueError("Mutation plan did not contain any applicable file mutations.")
        return parsed, mutation_dsl

    def on_cycle_complete(self, entry: dict) -> None:
        """Trigger evolution every N cycles, or immediately for structural hotspot signals."""
        self._cycle_count += 1

        goal = entry.get("goal", "evolve and improve the AURA system")
        is_hotspot = "structural_hotspot" in str(entry.get("phase_outputs", {}).get("skill_context", {})) or "refactor hotspot" in goal.lower()

        # If we have an improvement service, evaluate recent history
        if self.improvement_service and self.brain:
            recent_history = self.improvement_service.observe_cycle(entry)
            proposals = self.improvement_service.evaluate_candidates(recent_history)
            for p in proposals:
                self.improvement_service.log_proposal(p)
                # In the future, we could auto-enqueue these proposals as goals

        if self._cycle_count % self.TRIGGER_EVERY_N == 0 or is_hotspot:
            if is_hotspot:
                log_json("INFO", "evolution_loop_triggered_by_hotspot_signal", details={"goal": goal})
            self.run(goal, execute_queued=False)

    def _hypothesize(self, goal: str, memory_snapshot: str, similar_past_problems: str, known_weaknesses: str) -> str:
        """Phase 1: Generate a hypothesis for the evolutionary goal."""
        hypothesis = self.planner.plan(
            goal=EVOLUTION_HYPOTHESIS_PROMPT.replace("{goal}", goal).replace("{memory_snapshot}", memory_snapshot).replace("{similar_past_problems}", similar_past_problems).replace("{known_weaknesses}", known_weaknesses),
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses,
        )
        return "\n".join(hypothesis) if isinstance(hypothesis, list) else str(hypothesis)

    def _decompose_tasks(self, hypothesis_str: str, memory_snapshot: str, similar_past_problems: str, known_weaknesses: str):
        """Phase 2: Decompose hypothesis into atomic tasks."""
        task_list = self.planner.plan(
            goal=EVOLUTION_TASK_DECOMPOSITION_PROMPT.replace("{hypothesis}", hypothesis_str).replace("{memory_snapshot}", memory_snapshot),
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses,
        )
        return task_list

    def _implement_and_critique(self, goal: str, task_list) -> tuple:
        """Phase 3-4: Implement tasks and critique the result."""
        task_str = "\n".join(task_list)
        implementation = self.coder.implement(task_str)
        evaluation = self.critic.critique_code(
            task=goal,
            code=implementation,
            requirements="Evaluate the implementation against the goal and provide a score based on correctness, efficiency, and adherence to the task.",
        )
        self.brain.analyze_critique_for_weaknesses(evaluation)
        return implementation, evaluation

    def _propose_and_validate_mutation(self, evaluation_str: str, memory_snapshot: str, similar_past_problems: str, known_weaknesses: str) -> tuple:
        """Phase 5-6: Propose mutation and validate it."""
        mutation_prompt = EVOLUTION_MUTATION_PROMPT.replace("{evaluation}", evaluation_str).replace("{memory_snapshot}", memory_snapshot)
        raw_mutation_response = self._get_mutation_response(
            mutation_prompt,
            memory_snapshot,
            similar_past_problems,
            known_weaknesses,
        )
        mutation_plan = None
        mutation_str = raw_mutation_response
        try:
            mutation_plan, mutation_str = self._normalize_mutation_plan(raw_mutation_response)
        except Exception as e:
            log_json("ERROR", "evolution_mutation_plan_invalid", details={"error": str(e), "raw_response_snippet": raw_mutation_response[:200]})

        validation_target = json.dumps(mutation_plan) if mutation_plan is not None else mutation_str
        raw_validation_result = self.critic.validate_mutation(validation_target)

        decision, confidence_score = self._parse_validation_result(raw_validation_result)

        if decision == "APPROVED" and confidence_score >= 0.7:
            self.mutator.apply_mutation(mutation_str)
            log_json("INFO", "mutation_approved_and_applied", details={"confidence": confidence_score})
        else:
            log_json("INFO", "mutation_rejected", details={"decision": decision, "confidence": confidence_score, "reason": "Programmatic decision"})

        return mutation_plan, mutation_str

    def _parse_validation_result(self, raw_validation_result: str) -> tuple:
        """Parse the mutation validation result from the critic."""
        decision = "REJECTED"
        confidence_score = 0.0
        impact_assessment = "N/A"
        reasoning = "Could not parse validation response or LLM failed to provide valid structure."

        try:
            parsed_validation = _aura_safe_loads(raw_validation_result, "evolution_mutation_validation")
            if isinstance(parsed_validation, dict):
                decision = parsed_validation.get("decision", decision)
                raw_confidence = parsed_validation.get("confidence_score", confidence_score)
                try:
                    confidence_score = float(raw_confidence)
                except (ValueError, TypeError):
                    confidence_score = 0.0
                impact_assessment = parsed_validation.get("impact_assessment", impact_assessment)
                reasoning = parsed_validation.get("reasoning", reasoning)
            else:
                log_json("ERROR", "evolution_mutation_validation_invalid_format", details={"raw_response_snippet": raw_validation_result[:200]})
        except json.JSONDecodeError as e:
            log_json("ERROR", "evolution_mutation_validation_json_decode_error", details={"error": str(e), "raw_response_snippet": raw_validation_result[:200]})
        except Exception as e:
            log_json("ERROR", "evolution_mutation_validation_unexpected_error", details={"error": str(e), "raw_response_snippet": raw_validation_result[:200]})

        log_json("INFO", "mutation_validation_result", details={"decision": decision, "confidence": confidence_score, "impact": impact_assessment, "reasoning_snippet": reasoning[:100]})
        return decision, confidence_score

    def _commit_and_track_experiment(self, goal: str, experiment_id: str, t0_experiment: float, hypothesis_str: str, baseline_metrics) -> dict:
        """Phase 7: Commit changes and evaluate experiment outcome."""
        self.git.commit_all(f"AURA evolutionary update: {goal}")

        experiment_result = self.experiment_tracker.finish_experiment(
            experiment_id=experiment_id,
            hypothesis=hypothesis_str[:200],
            change_description=f"Evolution: {goal[:100]}",
            metrics_before=baseline_metrics,
            cycle_number=self._cycle_count,
            duration=time.time() - t0_experiment,
        )
        if not experiment_result.kept:
            log_json("WARN", "evolution_experiment_discarded", details={"id": experiment_id, "reason": experiment_result.reason})
            try:
                self.git.run(["git", "revert", "--no-commit", "HEAD"])
                self.git.commit_all(f"Revert evolution (experiment {experiment_id} regressed)")
                log_json("INFO", "evolution_reverted", details={"id": experiment_id})
            except Exception as revert_err:
                log_json("WARN", "evolution_revert_failed", details={"error": str(revert_err)})

        return {
            "id": experiment_id,
            "kept": experiment_result.kept,
            "net_improvement": experiment_result.net_improvement,
        }

    def _persist_memories(self, goal: str, hypothesis, evaluation, mutation_str: str) -> None:
        """Store evolution artifacts in brain and vector memory."""
        self.brain.remember(goal)
        self.brain.remember(hypothesis)
        self.brain.remember(evaluation)
        self.brain.remember(mutation_str)

        if self.vector:
            self.vector.add(goal)
            self.vector.add(mutation_str)

    def run(
        self,
        goal,
        *,
        execute_queued: bool | None = None,
        dry_run: bool = False,
        proposal_limit: int | None = None,
        focus: str = "capability",
    ):
        """
        Executes a single cycle of the evolutionary development loop. This involves:
        1. Hypothesizing a solution or improvement.
        2. Decomposing the hypothesis into atomic tasks.
        3. Implementing the tasks via the CoderAgent.
        4. Critiquing the implementation.
        5. Proposing and integrating mutations to the AURA system itself.
        6. Committing the changes.

        Args:
            goal (str): The current overarching goal for this evolutionary cycle.

        Returns:
            dict: A dictionary containing the results of various phases of the loop,
                  such as hypothesis, tasks, implementation, evaluation, and mutation.
        """
        execute_queued = self.auto_execute_queued if execute_queued is None else execute_queued
        proposal_limit = max(1, int(proposal_limit or self.innovation_goal_limit))

        if self.goal_queue is not None or self.orchestrator is not None:
            innovation_result = self._run_innovation_workflow(
                goal,
                execute_queued=execute_queued,
                dry_run=dry_run,
                proposal_limit=proposal_limit,
                focus=focus,
            )
            if innovation_result["selected_proposals"]:
                log_json(
                    "INFO",
                    "innovation_workflow_complete",
                    details={
                        "goal": goal,
                        "selected": len(innovation_result["selected_proposals"]),
                        "queued": len(innovation_result["queue"].get("queued", [])),
                        "executed": len(innovation_result["implementation"].get("executed", [])),
                    },
                )
                return {
                    "status": "ok",
                    "goal": goal,
                    "innovation_report": innovation_result,
                    "queued_goals": innovation_result["queue"].get("queued", []),
                    "implementation_results": innovation_result["implementation"].get("executed", []),
                    "meta": {
                        "mode": "queue_then_implement" if execute_queued else "queue_only",
                        "focus": focus,
                        "proposal_limit": proposal_limit,
                        "proposal_count": len(innovation_result["proposals"]),
                        "selected_count": len(innovation_result["selected_proposals"]),
                    },
                }

        # ── Experiment tracking: capture baseline ──
        experiment_id = f"evo_{uuid.uuid4().hex[:8]}"
        t0_experiment = time.time()
        baseline_metrics = self.experiment_tracker.start_experiment(
            experiment_id,
            f"Evolution: {goal[:100]}",
        )

        # Gather system state for PlannerAgent
        memory_snapshot = "\n".join(self.brain.recall_with_budget(max_tokens=3000))
        similar_past_problems = "\n".join(self.vector.search(goal)) if self.vector else ""
        known_weaknesses = "\n".join(self.brain.recall_weaknesses())

        log_json("INFO", "evolution_loop_start", details={"goal": goal})

        # 1. Hypothesize
        hypothesis_str = self._hypothesize(goal, memory_snapshot, similar_past_problems, known_weaknesses)

        # 2. Decompose
        task_list = self._decompose_tasks(hypothesis_str, memory_snapshot, similar_past_problems, known_weaknesses)

        # 3-4. Implement and Critique
        implementation, evaluation = self._implement_and_critique(goal, task_list)

        # 5-6. Propose and validate mutation
        mutation_plan, mutation_str = self._propose_and_validate_mutation(
            str(evaluation), memory_snapshot, similar_past_problems, known_weaknesses,
        )

        # Persist memories
        self._persist_memories(goal, hypothesis_str, evaluation, mutation_str)

        # 7. Commit and track experiment
        experiment_info = self._commit_and_track_experiment(
            goal, experiment_id, t0_experiment, hypothesis_str, baseline_metrics,
        )

        return {
            "hypothesis": hypothesis_str,
            "tasks": task_list,
            "implementation": implementation,
            "evaluation": evaluation,
            "mutation": mutation_plan if mutation_plan is not None else mutation_str,
            "experiment": experiment_info,
        }
