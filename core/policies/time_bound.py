from __future__ import annotations

import time

from core.policies.base import PolicyBase


class TimeBoundPolicy(PolicyBase):
    def __init__(self, max_seconds: int = 120) -> None:
        self.max_seconds = max_seconds

    def evaluate(self, history: list[dict], verification: dict, started_at: float | None = None) -> str:
        if verification.get("status") == "pass":
            return "PASS"
        if started_at is None:
            return ""
        if time.time() - started_at >= self.max_seconds:
            return "TIME_LIMIT"
        return ""
