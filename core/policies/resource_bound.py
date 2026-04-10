"""Resource-bound convergence policy: stops loop when estimated token usage exceeds budget."""
from __future__ import annotations

from core.policies.base import PolicyBase
from core.logging_utils import log_json


class ResourceBoundPolicy(PolicyBase):
    """
    Stops the loop when estimated total token usage exceeds `max_tokens`.
    Estimation: ~4 chars per token across all log text in history.
    """
    CHARS_PER_TOKEN: int = 4

    def __init__(self, max_tokens: int = 50000) -> None:
        self.max_tokens = max_tokens

    def evaluate(self, history: list[dict], verification: dict, started_at: float | None = None) -> str:
        if verification and verification.get("status") == "pass":
            return "PASS"
        total_chars = sum(
            len(str(entry.get("phase_outputs", {})))
            for entry in history
        )
        estimated_tokens = total_chars // self.CHARS_PER_TOKEN
        if estimated_tokens >= self.max_tokens:
            log_json("WARN", "resource_bound_policy_limit_reached",
                     details={"estimated_tokens": estimated_tokens, "max_tokens": self.max_tokens})
            return "TOKEN_BUDGET_EXCEEDED"
        return ""
