"""Agent for generating focused tests from code or a task description."""

from __future__ import annotations

from typing import Any

__test__ = False


class TestGeneratorAgent:
    __test__ = False

    def __init__(self, brain: Any, model: Any) -> None:
        self.brain = brain
        self.model = model

    def generate(self, *, goal: str = "", code: str = "", context: str = "") -> str:
        prompt = "\n".join(
            [
                "You generate focused regression tests.",
                f"Goal: {goal}".strip(),
                "Code:",
                code,
                "Context:",
                context,
                "Return only the test code.",
            ]
        ).strip()
        response = self.model.respond_for_role("quality", prompt) if hasattr(self.model, "respond_for_role") else self.model.respond(prompt)
        if getattr(self, "brain", None) is not None:
            self.brain.remember(f"Generated tests for goal: {goal or 'ad-hoc'}")
        return response
