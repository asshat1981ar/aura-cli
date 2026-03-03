import json
import dataclasses
from core.logging_utils import log_json # Import the new logging utility
from core.file_tools import _aura_safe_loads # Import _aura_safe_loads
from core.evolution_prompts import (
    EVOLUTION_HYPOTHESIS_PROMPT,
    EVOLUTION_TASK_DECOMPOSITION_PROMPT,
    EVOLUTION_MUTATION_PROMPT
)

class EvolutionLoop:
    """
    Orchestrates an advanced evolutionary loop for autonomous development,
    involving multiple agents (Planner, Coder, Critic, Mutator) to
    hypothesize, decompose, implement, critique, and mutate code,
    driving continuous self-improvement of the AURA system.
    """

    def __init__(self, planner, coder, critic, brain, vector_store, git_tools, mutator, improvement_service=None):
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
        self._cycle_count = 0
        self.TRIGGER_EVERY_N = 20

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
                block = "\n".join([
                    f"REPLACE_IN_FILE {file_path}",
                    "---OLD_CONTENT_START---",
                    str(old_content),
                    "---OLD_CONTENT_END---",
                    "---NEW_CONTENT_START---",
                    str(new_content),
                    "---NEW_CONTENT_END---",
                ])
            else:
                block = "\n".join([
                    f"ADD_FILE {file_path}",
                    str(new_content),
                ])
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
            self.run(goal)

    def run(self, goal):
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
        # Gather system state for PlannerAgent
        memory_snapshot = "\n".join(self.brain.recall_with_budget(max_tokens=3000))
        similar_past_problems = "\n".join(self.vector.search(goal)) if self.vector else ""
        known_weaknesses = "\n".join(self.brain.recall_weaknesses())

        log_json("INFO", "evolution_loop_start", details={"goal": goal})

        # 1. Hypothesize
        hypothesis = self.planner.plan(
            goal=EVOLUTION_HYPOTHESIS_PROMPT.replace("{goal}", goal)
                                            .replace("{memory_snapshot}", memory_snapshot)
                                            .replace("{similar_past_problems}", similar_past_problems)
                                            .replace("{known_weaknesses}", known_weaknesses),
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses
        )
        # Hypothesis can be a list or a string depending on model output, ensure string
        hypothesis_str = "\n".join(hypothesis) if isinstance(hypothesis, list) else str(hypothesis)

        # 2. Decompose
        task_list = self.planner.plan(
            goal=EVOLUTION_TASK_DECOMPOSITION_PROMPT.replace("{hypothesis}", hypothesis_str)
                                                    .replace("{memory_snapshot}", memory_snapshot),
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses
        )
        task_str = "\n".join(task_list)

        # 3. Execute
        implementation = self.coder.implement(task_str)

        # 4. Critique
        evaluation = self.critic.critique_code(
            task=goal,
            code=implementation,
            requirements="Evaluate the implementation against the goal and provide a score based on correctness, efficiency, and adherence to the task."
        )

        # Detect weaknesses from evaluation using Brain's method
        self.brain.analyze_critique_for_weaknesses(evaluation)

        evaluation_str = str(evaluation)

        # 5. Mutate (Self-Improvement)
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

        # 6. Integrate mutation
        validation_target = json.dumps(mutation_plan) if mutation_plan is not None else mutation_str
        raw_validation_result = self.critic.validate_mutation(validation_target)
        
        # Initialize with safe defaults
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

        # Programmatic decision to apply mutation
        if decision == "APPROVED" and confidence_score >= 0.7: # Example threshold
            self.mutator.apply_mutation(mutation_str)
            log_json("INFO", "mutation_approved_and_applied", details={"confidence": confidence_score})
        else:
            log_json("INFO", "mutation_rejected", details={"decision": decision, "confidence": confidence_score, "reason": "Programmatic decision"})

        self.brain.remember(goal)
        self.brain.remember(hypothesis)
        self.brain.remember(evaluation)
        self.brain.remember(mutation_str)

        if self.vector:
            self.vector.add(goal)
            self.vector.add(mutation_str)

        # 7. Commit evolution
        self.git.commit_all(f"AURA evolutionary update: {goal}")

        return {
            "hypothesis": hypothesis,
            "tasks": task_list,
            "implementation": implementation,
            "evaluation": evaluation,
            "mutation": mutation_plan if mutation_plan is not None else mutation_str
        }
