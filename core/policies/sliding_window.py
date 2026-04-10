from __future__ import annotations

from core.policies.base import PolicyBase


class SlidingWindowPolicy(PolicyBase):
    def __init__(self, max_cycles: int = 5) -> None:
        self.max_cycles = max_cycles

    def evaluate(self, history: list[dict], verification: dict, started_at: float | None = None) -> str:
        if verification.get("status") == "pass":
            return "PASS"
        if len(history) >= self.max_cycles:
            return "MAX_CYCLES"
        return ""
