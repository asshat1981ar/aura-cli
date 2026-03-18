"""Agent for proposing and drafting scoped refactors."""

from __future__ import annotations

from typing import Any


class RefactorAgent:
    def __init__(self, brain: Any, model: Any) -> None:
        self.brain = brain
        self.model = model

    def suggest(self, *, goal: str, code: str = "", constraints: list[str] | None = None) -> str:
        prompt = "\n".join(
            [
                "You are a careful refactoring agent.",
                f"Goal: {goal}",
                f"Constraints: {', '.join(constraints or []) or 'preserve behavior'}",
                "Code:",
                code,
                "Return a concrete refactor plan or patch-ready guidance.",
            ]
        )
        response = self.model.respond_for_role("analysis", prompt) if hasattr(self.model, "respond_for_role") else self.model.respond(prompt)
        if getattr(self, "brain", None) is not None:
            self.brain.remember(f"RefactorAgent reviewed: {goal}")
        return response
