"""Cost tracker and cap enforcement for AURA model calls.

Usage::

    from core.cost_tracker import CostTracker, CostCapExceededError

    tracker = CostTracker()          # reads AURA_COST_CAP_USD from env
    tracker.record(model, tokens)    # raises if cap exceeded
"""

from __future__ import annotations

import os
from typing import Final

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Approximate prices per 1 000 000 tokens (input+output blended estimate).
# These are intentionally conservative approximations; update as pricing changes.
# ---------------------------------------------------------------------------
MODEL_PRICES: dict[str, float] = {
    "google/gemini-2.0-flash-exp:free": 0.0,
    "google/gemini-2.5-pro": 3.5,
    "openai/text-embedding-3-small": 0.02,
    # Fallback for unknown models: treat as moderately expensive
    "_default": 1.0,
}

_WARN_FRACTION: Final[float] = 0.80  # log warning at 80 % of cap


class CostCapExceededError(Exception):
    """Raised when the cumulative model call cost exceeds the configured cap."""

    def __init__(self, current_usd: float, cap_usd: float) -> None:
        super().__init__(f"Cost cap exceeded: ${current_usd:.6f} >= cap ${cap_usd:.6f}")
        self.current_usd = current_usd
        self.cap_usd = cap_usd


class CostTracker:
    """Track estimated LLM spend and enforce an optional cost cap.

    The cap is read once at construction time from the ``AURA_COST_CAP_USD``
    environment variable.  If the variable is absent or empty the cap is
    disabled (``None``).

    Args:
        cap_usd: Override cap in USD.  ``None`` disables cap enforcement.
            When omitted the value of ``AURA_COST_CAP_USD`` env var is used.
    """

    _SENTINEL: object = object()  # sentinel for default argument detection

    def __init__(self, cap_usd: object = _SENTINEL) -> None:
        if cap_usd is CostTracker._SENTINEL:
            env_val = os.environ.get("AURA_COST_CAP_USD", "").strip()
            self.cap_usd: float | None = float(env_val) if env_val else None
        else:
            self.cap_usd = float(cap_usd) if cap_usd is not None else None  # type: ignore[arg-type]

        self.total_usd: float = 0.0

    def record(self, model: str, tokens: int) -> None:
        """Record a model call and raise if the cost cap is breached.

        Args:
            model: Model identifier (key in :data:`MODEL_PRICES`).
            tokens: Estimated total token count for the call.

        Raises:
            CostCapExceededError: If adding this call would breach the cap.
        """
        price_per_million = MODEL_PRICES.get(model, MODEL_PRICES["_default"])
        call_cost = (tokens / 1_000_000) * price_per_million
        self.total_usd += call_cost

        if self.cap_usd is None:
            return  # no cap configured

        fraction = self.total_usd / self.cap_usd
        if fraction >= 1.0:
            log_json(
                "ERROR",
                "cost_cap_exceeded",
                details={"current_usd": self.total_usd, "cap_usd": self.cap_usd},
            )
            raise CostCapExceededError(self.total_usd, self.cap_usd)

        if fraction >= _WARN_FRACTION:
            log_json(
                "WARNING",
                "cost_cap_approaching",
                details={"current_usd": self.total_usd, "cap_usd": self.cap_usd},
            )

    def reset(self) -> None:
        """Reset the running total (useful between test cases)."""
        self.total_usd = 0.0
