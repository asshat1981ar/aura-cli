"""Three-state circuit breaker for LLM provider calls.

States:
  CLOSED    — normal operation; requests are forwarded to the underlying fn.
  OPEN      — circuit is tripped; all requests fail fast with
              :class:`CircuitBreakerOpenError`.
  HALF_OPEN — trial state after recovery_timeout; one request is allowed
              through as a probe.  Success → CLOSED, failure → OPEN.
"""

from __future__ import annotations

import time
from typing import Any, Callable


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted while the circuit breaker is OPEN."""

    def __init__(self, name: str = "circuit_breaker") -> None:
        super().__init__(f"Circuit breaker '{name}' is OPEN — calls rejected.")
        self.name = name


class CircuitBreaker:
    """Three-state circuit breaker: CLOSED → OPEN → HALF_OPEN.

    Args:
        failure_threshold: Number of consecutive failures required to open the
            circuit.  Defaults to ``5``.
        recovery_timeout: Seconds to wait in OPEN state before transitioning
            to HALF_OPEN for a probe call.  Defaults to ``60``.
        name: Optional label used in log messages and error text.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "circuit_breaker",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self.state: str = "CLOSED"
        self.failure_count: int = 0
        self.last_failure_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* through the circuit breaker.

        Args:
            fn: Callable to invoke.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            Whatever *fn* returns on success.

        Raises:
            CircuitBreakerOpenError: If the circuit is currently OPEN and the
                recovery timeout has not yet elapsed.
            Exception: Any exception raised by *fn* is propagated after
                recording the failure.
        """
        if self.is_open():
            raise CircuitBreakerOpenError(self.name)

        # HALF_OPEN: one probe allowed — transition is implicit via is_open()
        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def record_success(self) -> None:
        """Reset failure bookkeeping and close the circuit."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Increment failure count and open the circuit if threshold is met."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def is_open(self) -> bool:
        """Return *True* when the circuit is OPEN and should reject calls.

        If the circuit is OPEN but the *recovery_timeout* has elapsed,
        transition to HALF_OPEN (allowing a single probe call) and return
        *False*.
        """
        if self.state == "CLOSED":
            return False

        if self.state == "OPEN":
            if (
                self.last_failure_time is not None
                and (time.monotonic() - self.last_failure_time) >= self.recovery_timeout
            ):
                self.state = "HALF_OPEN"
                return False
            return True

        # HALF_OPEN — let probe through
        return False
