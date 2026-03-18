"""Circuit breaker for the orchestration pipeline.

Tracks consecutive failures and opens the circuit when a threshold is
reached, preventing further cycles until a cooldown period elapses.
"""
import time


class CircuitBreaker:
    """Simple consecutive-failure circuit breaker with time-based cooldown."""

    def __init__(self, threshold: int = 5, cooldown_s: float = 60.0):
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._consecutive_fails = 0
        self._opened_at: float | None = None

    def record_success(self) -> None:
        self._consecutive_fails = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_fails += 1
        if self._consecutive_fails >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._cooldown_s:
            # Cooldown elapsed — half-open: allow one attempt
            self._opened_at = None
            self._consecutive_fails = 0
            return False
        return True

    @property
    def consecutive_fails(self) -> int:
        return self._consecutive_fails

    def as_dict(self) -> dict:
        return {
            "consecutive_fails": self._consecutive_fails,
            "threshold": self._threshold,
            "is_open": self.is_open(),
            "cooldown_s": self._cooldown_s,
        }
