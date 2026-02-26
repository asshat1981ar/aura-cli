import json
from core.logging_utils import log_json # Import the new logging utility
from core.file_tools import _aura_safe_loads # Import _aura_safe_loads

class EvolutionLoop:
    """
    Orchestrates an advanced evolutionary loop for autonomous development,
    involving multiple agents (Planner, Coder, Critic, Mutator) to
    hypothesize, decompose, implement, critique, and mutate code,
    driving continuous self-improvement of the AURA system.
    """

    def __init__(self, planner, coder, critic, brain, vector_store, git_tools, mutator):
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
        """
        self.planner = planner
        self.coder = coder
        self.critic = critic
        self.brain = brain
        self.vector = vector_store
        self.git = git_tools
        self.mutator = mutator

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
        memory_snapshot = "\n".join(self.brain.recall_all())
        similar_past_problems = "\n".join(self.vector.search(goal))
        known_weaknesses = "\n".join(self.brain.recall_weaknesses())

        # 1. Hypothesize
        hypothesis = self.planner.plan(
            goal=f"Analyze this goal: {goal}\nIdentify:\n- Required capabilities\n- Missing architecture\n- Possible improvements to AURA itself",
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses
        )

        # 2. Decompose
        task_list = self.planner.plan(
            goal=f"Break this into atomic executable tasks:\n{hypothesis}",
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

        # 5. Mutate (Self-Improvement)
        mutation_list = self.planner.plan(
            goal=f"Based on this critique:\n{evaluation}\n\nPropose modifications to:\n- Agents\n- Memory\n- Model routing\n- Tooling",
            memory_snapshot=memory_snapshot,
            similar_past_problems=similar_past_problems,
            known_weaknesses=known_weaknesses
        )
        mutation_str = "\n".join(mutation_list)

        # 6. Integrate mutation
        raw_validation_result = self.critic.validate_mutation(mutation_str)
        
        # Initialize with safe defaults
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

        self.vector.add(goal)
        self.vector.add(mutation_str)

        # 7. Commit evolution
        self.git.commit_all(f"AURA evolutionary update: {goal}")

        return {
            "hypothesis": hypothesis,
            "tasks": task_list,
            "implementation": implementation,
            "evaluation": evaluation,
            "mutation": mutation_list
        }
