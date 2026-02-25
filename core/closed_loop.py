from core.prompts import BOOTSTRAP_PROMPT_CLOSED_LOOP

class ClosedDevelopmentLoop:
    """
    Implements a basic closed-loop autonomous development workflow.
    It orchestrates LLM interactions to define, plan, implement, test,
    critique, improve, and version code changes.
    """

    def __init__(self, model, brain, git_tools):
        """
        Initializes the ClosedDevelopmentLoop with model, brain, and Git tools.

        Args:
            model: An instance of the ModelAdapter for LLM interactions.
            brain: An instance of the system's memory (Brain).
            git_tools: An instance of GitTools for repository operations.
        """
        self.model = model
        self.brain = brain
        self.git = git_tools

    def snapshot(self):
        """
        Captures a snapshot of the current system state, primarily memory entries.

        Returns:
            str: A string representation of the system snapshot.
        """
        memory_count = len(self.brain.recall_all())
        return f"Memory entries: {memory_count}"

    def run(self, goal):
        """
        Executes a single iteration of the Closed Development Loop.
        1. Captures system state.
        2. Prompts the LLM with the goal and state.
        3. Remembers the goal and LLM response.
        4. Commits all changes to Git.

        Args:
            goal (str): The current goal for this iteration.

        Returns:
            str: The raw response from the LLM.
        """

        state = self.snapshot()

        prompt = self._bootstrap_prompt_template.format(GOAL=goal, STATE=f"Current System State:\\n{state}")

        response = self.model.respond(prompt)

        self.brain.remember(goal)
        self.brain.remember(response)

        self.git.commit_all(f"Closed loop update: {goal}")

        return response
