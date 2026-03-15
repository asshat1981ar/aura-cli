import json

from core.file_tools import _aura_safe_loads
from core.logging_utils import log_json


class EvolutionLoop:
    """
    Orchestrates the recursive improvement workflow used by AURA.
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
    ):
        self.planner = planner
        self.coder = coder
        self.critic = critic
        self.brain = brain
        self.vector = vector_store
        self.git = git_tools
        self.mutator = mutator
        self.improvement_service = improvement_service

    def _planner_respond(self, prompt: str):
        responder = getattr(self.planner, "_respond", None)
        if callable(responder):
            return responder(prompt)
        return self.planner.plan(prompt, "", "", "")

    def _build_mutation_prompt(
        self,
        evaluation: str,
        memory_snapshot: str,
        similar_past_problems: str,
        known_weaknesses: str,
    ) -> str:
        return (
            "Based on this critique:\n"
            f"{evaluation}\n\n"
            "Produce a structured JSON mutation plan with a top-level "
            "\"mutations\" array. Each mutation should include type, file_path, "
            "and new_content, plus old_code when replacing existing content.\n\n"
            f"Memory Snapshot:\n{memory_snapshot}\n\n"
            f"Similar Past Problems:\n{similar_past_problems}\n\n"
            f"Known Weaknesses:\n{known_weaknesses}"
        )

    def _structured_mutation_to_commands(self, mutations) -> str:
        commands = []
        for mutation in mutations:
            if not isinstance(mutation, dict):
                continue
            mutation_type = mutation.get("type")
            file_path = mutation.get("file_path")
            new_content = mutation.get("new_content")
            if not isinstance(file_path, str) or not isinstance(new_content, str):
                continue

            old_code = mutation.get("old_code", "")
            if mutation_type in {"file_change", "add_file"} and not old_code:
                commands.append(f"ADD_FILE {file_path}\n{new_content}")
                continue

            commands.append(
                "\n".join(
                    [
                        f"REPLACE_IN_FILE {file_path}",
                        "---OLD_CONTENT_START---",
                        str(old_code),
                        "---OLD_CONTENT_END---",
                        "---NEW_CONTENT_START---",
                        new_content,
                        "---NEW_CONTENT_END---",
                    ]
                )
            )
        return "\n".join(commands)

    def _normalize_mutation_plan(self, raw_response: str):
        try:
            parsed = _aura_safe_loads(raw_response, "evolution_mutation_plan")
        except json.JSONDecodeError:
            return raw_response, raw_response

        if isinstance(parsed, dict) and isinstance(parsed.get("mutations"), list):
            return parsed, self._structured_mutation_to_commands(parsed["mutations"])

        return parsed, raw_response

    def on_cycle_complete(self, cycle_entry):
        if self.improvement_service is None:
            return []

        cycle_history = self.improvement_service.observe_cycle(cycle_entry)
        proposals = self.improvement_service.evaluate_candidates(cycle_history)
        for proposal in proposals:
            self.improvement_service.log_proposal(proposal)
        return proposals

    def run(self, goal, context=None):
        memory_snapshot = "\n".join(self.brain.recall_with_budget(max_tokens=3000))
        similar_past_problems = "\n".join(self.vector.search(goal))
        known_weaknesses = "\n".join(self.brain.recall_weaknesses())

        hypothesis = self.planner.plan(
            goal=f"Analyze this goal: {goal}\nIdentify:\n- Required capabilities\n- Missing architecture\n- Possible improvements to AURA itself",
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses,
        )

        task_list = self.planner.plan(
            goal=f"Break this into atomic executable tasks:\n{hypothesis}",
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses,
        )
        task_str = "\n".join(task_list)

        implementation = self.coder.implement(task_str)

        evaluation = self.critic.critique_code(
            task=goal,
            code=implementation,
            requirements="Evaluate the implementation against the goal and provide a score based on correctness, efficiency, and adherence to the task.",
        )

        analyze_critique = getattr(self.brain, "analyze_critique_for_weaknesses", None)
        if callable(analyze_critique):
            analyze_critique(evaluation)

        mutation_raw_response = self._planner_respond(
            self._build_mutation_prompt(
                evaluation,
                memory_snapshot,
                similar_past_problems,
                known_weaknesses,
            )
        )
        mutation_result, mutation_commands = self._normalize_mutation_plan(mutation_raw_response)
        validation_payload = mutation_raw_response if isinstance(mutation_result, dict) else mutation_commands

        try:
            raw_validation_result = self.critic.validate_mutation(validation_payload)
        except Exception as e:
            log_json("ERROR", "mutation_validation_call_failed", details={"error": str(e)})
            raw_validation_result = ""

        decision = "REJECTED"
        confidence_score = 0.0
        impact_assessment = "N/A"
        reasoning = "Could not parse validation response or LLM failed to provide valid structure."

        try:
            parsed_validation = _aura_safe_loads(raw_validation_result, "evolution_mutation_validation")
            if isinstance(parsed_validation, dict):
                decision = parsed_validation.get("decision", decision)
                confidence_score = parsed_validation.get("confidence_score", confidence_score)
                impact_assessment = parsed_validation.get("impact_assessment", impact_assessment)
                reasoning = parsed_validation.get("reasoning", reasoning)
            else:
                log_json(
                    "ERROR",
                    "evolution_mutation_validation_invalid_format",
                    details={"raw_response_snippet": raw_validation_result[:200]},
                )
        except json.JSONDecodeError as e:
            log_json(
                "ERROR",
                "evolution_mutation_validation_json_decode_error",
                details={"error": str(e), "raw_response_snippet": raw_validation_result[:200]},
            )
        except Exception as e:
            log_json(
                "ERROR",
                "evolution_mutation_validation_unexpected_error",
                details={"error": str(e), "raw_response_snippet": raw_validation_result[:200]},
            )

        log_json(
            "INFO",
            "mutation_validation_result",
            details={
                "decision": decision,
                "confidence": confidence_score,
                "impact": impact_assessment,
                "reasoning_snippet": reasoning[:100],
            },
        )

        mutation_applied = decision == "APPROVED" and confidence_score >= 0.7

        if mutation_applied:
            self.mutator.apply_mutation(mutation_commands)
            log_json("INFO", "mutation_approved_and_applied", details={"confidence": confidence_score})
            self.git.commit_all(f"AURA evolutionary update: {goal}")
        else:
            log_json(
                "INFO",
                "mutation_rejected",
                details={"decision": decision, "confidence": confidence_score, "reason": "Programmatic decision"},
            )

        self.brain.remember(goal)
        self.brain.remember(hypothesis)
        self.brain.remember(evaluation)
        self.brain.remember(mutation_raw_response)

        self.vector.add(goal)
        self.vector.add(mutation_commands)

        return {
            "hypothesis": hypothesis,
            "tasks": task_list,
            "implementation": implementation,
            "evaluation": evaluation,
            "mutation": mutation_result,
            "validation": {
                "decision": decision,
                "confidence_score": confidence_score,
                "impact_assessment": impact_assessment,
                "reasoning": reasoning,
            },
            "mutation_applied": mutation_applied,
        }
