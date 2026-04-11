from __future__ import annotations


class PolicyBase:
    def evaluate(self, history: list[dict], verification: dict, started_at: float | None = None) -> str:
        raise NotImplementedError
