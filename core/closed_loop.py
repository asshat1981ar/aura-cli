
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from core.prompts import BOOTSTRAP_PROMPT_CLOSED_LOOP, SELF_DIRECTED_PROMPT


class ClosedDevelopmentLoop:
    """
    Implements a basic closed-loop autonomous development workflow.
    It orchestrates LLM interactions to define, plan, implement, test,
    critique, improve, and version code changes.
    """

    def __init__(
        self,
        model,
        brain,
        git_tools,
        *,
        prompt_template: Optional[str] = None,
        self_directed: bool = False,
    ):
        """
        Initializes the ClosedDevelopmentLoop with model, brain, and Git tools.

        Args:
            model: An instance of the ModelAdapter for LLM interactions.
            brain: An instance of the system's memory (Brain).
            git_tools: An instance of GitTools for repository operations.
            prompt_template: Optional custom prompt template override.
            self_directed: When True, use the self-directed development prompt template.
        """
        self.model = model
        self.brain = brain
        self.git = git_tools
        if prompt_template is not None:
            self._bootstrap_prompt_template = prompt_template
        elif self_directed:
            self._bootstrap_prompt_template = SELF_DIRECTED_PROMPT
        else:
            self._bootstrap_prompt_template = BOOTSTRAP_PROMPT_CLOSED_LOOP

    def snapshot(self):
        """
        Captures a snapshot of the current system state, primarily memory entries.

        Returns:
            str: A string representation of the system snapshot.
        """
        memory_count = self.brain.count_memories()
        return f"Memory entries: {memory_count}"

    def run(self, goal: str, *, context: Optional[Mapping[str, Any]] = None, prompt_template: Optional[str] = None):
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

        template = prompt_template or self._bootstrap_prompt_template
        prompt = self._format_prompt(
            template,
            {
                "GOAL": goal,
                "goal": goal,
                "STATE": f"Current System State:\\n{state}",
                "state": state,
                "project_summary": (context or {}).get("project_summary", state),
                "git_status": (context or {}).get("git_status", ""),
                "signals": (context or {}).get("signals", ""),
                "constraints": (context or {}).get("constraints", ""),
                "resources": (context or {}).get("resources", ""),
            },
        )

        response = self.model.respond(prompt)

        self.brain.remember(goal)
        self.brain.remember(response)

        self.git.commit_all(f"Closed loop update: {goal}")

        return response

    @staticmethod
    def _format_prompt(template: str, values: Dict[str, Any]) -> str:
        class _Safe(dict):
            def __missing__(self, key):
                return ""

        return template.format_map(_Safe(**values))
