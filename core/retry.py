"""Retry logic with exponential backoff for AURA CLI.

Provides robust retry mechanisms for transient failures with:
- Exponential backoff with jitter
- Configurable retry policies
- Circuit breaker integration
- Async and sync support

This module complements core/agent_sdk/resilience.py with higher-level
retry utilities specifically for CLI operations.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
)

from core.exceptions import (
    MCPConnectionError,
    MCPRetryExhaustedError,
    MCPTimeoutError,
    RateLimitError,
    CircuitBreakerOpenError,
)

# Import from agent_sdk for circuit breaker integration
try:
    from core.agent_sdk.resilience import (
        RetryConfig as AgentSDKRetryConfig,
        retry_with_backoff as agent_sdk_retry,
    )

    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Retry Policies and Configuration
# =============================================================================


class RetryStrategy(Enum):
    """Retry strategy types."""

    EXPONENTIAL_BACKOFF = auto()
    LINEAR = auto()
    FIXED = auto()
    IMMEDIATE = auto()


@dataclass
class RetryPolicy:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delays
        jitter_factor: Maximum jitter as fraction of delay (0.0-1.0)
        strategy: Retry strategy type
        retryable_exceptions: Tuple of exception types to retry on
        retryable_codes: Set of error codes to retry on
        on_retry: Optional callback called on each retry attempt
        on_exhausted: Optional callback called when retries exhausted
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.5
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: tuple = (Exception,)
    retryable_codes: Set[str] = field(default_factory=set)
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_exhausted: Optional[Callable[[int, Exception], None]] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.jitter_factor < 0 or self.jitter_factor > 1:
            raise ValueError("jitter_factor must be between 0 and 1")


# Predefined retry policies
class RetryPolicies:
    """Predefined retry policies for common scenarios."""

    DEFAULT = RetryPolicy()

    AGGRESSIVE = RetryPolicy(
        max_retries=5,
        base_delay=0.5,
        max_delay=30.0,
        exponential_base=1.5,
    )

    CONSERVATIVE = RetryPolicy(
        max_retries=3,
        base_delay=2.0,
        max_delay=120.0,
        exponential_base=2.0,
    )

    NO_RETRY = RetryPolicy(
        max_retries=0,
    )

    NETWORK = RetryPolicy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            MCPConnectionError,
            MCPTimeoutError,
            RateLimitError,
        ),
        retryable_codes={"AURA-300", "AURA-301", "AURA-302", "AURA-308"},
    )

    MCP = RetryPolicy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=(
            MCPConnectionError,
            MCPTimeoutError,
            MCPRetryExhaustedError,
        ),
        retryable_codes={"AURA-301", "AURA-302", "AURA-306"},
    )

    FILESYSTEM = RetryPolicy(
        max_retries=2,
        base_delay=0.5,
        max_delay=10.0,
        retryable_exceptions=(
            OSError,
            IOError,
            PermissionError,
        ),
        retryable_codes={"AURA-401", "AURA-406"},
    )


# =============================================================================
# Retry State Tracking
# =============================================================================


@dataclass
class RetryState:
    """Tracks the state of retry attempts.

    Attributes:
        attempt: Current attempt number (0-indexed)
        max_attempts: Maximum number of attempts
        last_exception: The last exception that occurred
        cumulative_delay: Total time spent waiting between retries
        start_time: Timestamp when retry sequence started
    """

    attempt: int = 0
    max_attempts: int = 0
    last_exception: Optional[Exception] = None
    cumulative_delay: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def is_first_attempt(self) -> bool:
        """Check if this is the first attempt."""
        return self.attempt == 0

    @property
    def is_last_attempt(self) -> bool:
        """Check if this is the last attempt."""
        return self.attempt >= self.max_attempts - 1

    @property
    def elapsed_time(self) -> float:
        """Get total elapsed time since start."""
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "last_exception_type": type(self.last_exception).__name__ if self.last_exception else None,
            "last_exception_message": str(self.last_exception) if self.last_exception else None,
            "cumulative_delay": self.cumulative_delay,
            "elapsed_time": self.elapsed_time,
        }


# =============================================================================
# Retry Result
# =============================================================================


@dataclass
class RetryResult(Generic[T]):
    """Result of a retry operation.

    Attributes:
        success: Whether the operation succeeded
        result: The result value (if success=True)
        exception: The final exception (if success=False)
        state: The retry state at completion
        attempts: Number of attempts made
    """

    success: bool
    result: Optional[T] = None
    exception: Optional[Exception] = None
    state: Optional[RetryState] = None
    attempts: int = 0

    @property
    def failed(self) -> bool:
        """Check if the operation failed."""
        return not self.success

    def unwrap(self) -> T:
        """Get the result or raise the exception.

        Returns:
            The result value

        Raises:
            The final exception if the operation failed
        """
        if self.success:
            return self.result  # type: ignore
        raise self.exception if self.exception else RuntimeError("Unknown error")

    def unwrap_or(self, default: T) -> T:
        """Get the result or return a default value.

        Args:
            default: Default value to return on failure

        Returns:
            The result or default
        """
        return self.result if self.success else default

    def unwrap_or_else(self, f: Callable[[Exception], T]) -> T:
        """Get the result or compute from exception.

        Args:
            f: Function to compute result from exception

        Returns:
            The result or computed value
        """
        return self.result if self.success else f(self.exception)  # type: ignore


# =============================================================================
# Core Retry Functions
# =============================================================================


async def with_retry(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args,
    policy: Optional[RetryPolicy] = None,
    retry_state: Optional[RetryState] = None,
    circuit_breaker: Optional[Any] = None,
    **kwargs,
) -> RetryResult[T]:
    """Execute an async function with retry logic.

    Args:
        fn: Async function to execute
        *args: Positional arguments to pass to fn
        policy: Retry policy (uses default if None)
        retry_state: Optional retry state to track (creates new if None)
        circuit_breaker: Optional circuit breaker to use
        **kwargs: Keyword arguments to pass to fn

    Returns:
        RetryResult containing success/failure and result or exception

    Example:
        >>> async def fetch_data():
        ...     return await http_client.get("/data")
        ...
        >>> result = await with_retry(fetch_data, policy=RetryPolicies.NETWORK)
        >>> if result.success:
        ...     print(result.result)
        >>> else:
        ...     print(f"Failed after {result.attempts} attempts")
    """
    policy = policy or RetryPolicies.DEFAULT
    state = retry_state or RetryState(max_attempts=policy.max_retries + 1)
    state.max_attempts = policy.max_retries + 1

    last_exception: Optional[Exception] = None

    for attempt in range(policy.max_retries + 1):
        state.attempt = attempt

        # Check circuit breaker
        if circuit_breaker and hasattr(circuit_breaker, "state"):
            if circuit_breaker.state.name == "OPEN":
                error = CircuitBreakerOpenError(circuit_name=getattr(circuit_breaker, "name", "unknown"))
                return RetryResult(
                    success=False,
                    exception=error,
                    state=state,
                    attempts=attempt + 1,
                )

        try:
            result = await fn(*args, **kwargs)

            # Success!
            return RetryResult(
                success=True,
                result=result,
                state=state,
                attempts=attempt + 1,
            )

        except Exception as e:
            last_exception = e
            state.last_exception = e

            # Check if we should retry this exception
            if not _should_retry(e, policy):
                logger.debug(f"Exception {type(e).__name__} is not retryable, failing fast")
                break

            # Check if this was the last attempt
            if attempt >= policy.max_retries:
                logger.debug(f"Retry exhausted after {attempt + 1} attempts")
                break

            # Calculate delay
            delay = _calculate_delay(attempt, policy)
            state.cumulative_delay += delay

            # Call on_retry callback
            if policy.on_retry:
                try:
                    policy.on_retry(attempt, e, delay)
                except Exception as callback_error:
                    logger.warning(f"on_retry callback failed: {callback_error}")

            # Log retry attempt
            logger.warning(f"Attempt {attempt + 1}/{policy.max_retries + 1} failed: {e}. Retrying in {delay:.2f}s...")

            # Wait before retry
            await asyncio.sleep(delay)

    # All retries exhausted (or non-retryable exception)
    if policy.on_exhausted:
        try:
            policy.on_exhausted(state.attempt + 1, last_exception)
        except Exception as callback_error:
            logger.warning(f"on_exhausted callback failed: {callback_error}")

    return RetryResult(
        success=False,
        exception=last_exception,
        state=state,
        attempts=state.attempt + 1,
    )


def with_retry_sync(
    fn: Callable[..., T],
    *args,
    policy: Optional[RetryPolicy] = None,
    retry_state: Optional[RetryState] = None,
    **kwargs,
) -> RetryResult[T]:
    """Execute a sync function with retry logic.

    Args:
        fn: Sync function to execute
        *args: Positional arguments to pass to fn
        policy: Retry policy (uses default if None)
        retry_state: Optional retry state to track
        **kwargs: Keyword arguments to pass to fn

    Returns:
        RetryResult containing success/failure and result or exception
    """
    policy = policy or RetryPolicies.DEFAULT
    state = retry_state or RetryState(max_attempts=policy.max_retries + 1)
    state.max_attempts = policy.max_retries + 1

    last_exception: Optional[Exception] = None

    for attempt in range(policy.max_retries + 1):
        state.attempt = attempt

        try:
            result = fn(*args, **kwargs)

            return RetryResult(
                success=True,
                result=result,
                state=state,
                attempts=attempt + 1,
            )

        except Exception as e:
            last_exception = e
            state.last_exception = e

            if not _should_retry(e, policy):
                break

            if attempt >= policy.max_retries:
                break

            delay = _calculate_delay(attempt, policy)
            state.cumulative_delay += delay

            if policy.on_retry:
                try:
                    policy.on_retry(attempt, e, delay)
                except Exception:
                    pass

            logger.warning(f"Attempt {attempt + 1}/{policy.max_retries + 1} failed: {e}. Retrying in {delay:.2f}s...")

            time.sleep(delay)

    if policy.on_exhausted:
        try:
            policy.on_exhausted(state.attempt + 1, last_exception)
        except Exception:
            pass

    return RetryResult(
        success=False,
        exception=last_exception,
        state=state,
        attempts=state.attempt + 1,
    )


async def with_retry_or_raise(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args,
    policy: Optional[RetryPolicy] = None,
    **kwargs,
) -> T:
    """Execute with retry and raise on failure.

    This is a convenience wrapper that unwraps the result or raises.

    Args:
        fn: Async function to execute
        *args: Positional arguments
        policy: Retry policy
        **kwargs: Keyword arguments

    Returns:
        Function result

    Raises:
        The last exception if all retries fail
    """
    result = await with_retry(fn, *args, policy=policy, **kwargs)
    return result.unwrap()


def with_retry_sync_or_raise(
    fn: Callable[..., T],
    *args,
    policy: Optional[RetryPolicy] = None,
    **kwargs,
) -> T:
    """Execute sync function with retry and raise on failure.

    Args:
        fn: Sync function to execute
        *args: Positional arguments
        policy: Retry policy
        **kwargs: Keyword arguments

    Returns:
        Function result

    Raises:
        The last exception if all retries fail
    """
    result = with_retry_sync(fn, *args, policy=policy, **kwargs)
    return result.unwrap()


# =============================================================================
# Helper Functions
# =============================================================================


def _should_retry(exception: Exception, policy: RetryPolicy) -> bool:
    """Check if an exception should be retried.

    Args:
        exception: The exception that occurred
        policy: Retry policy

    Returns:
        True if should retry, False otherwise
    """
    # Check exception type
    if not isinstance(exception, policy.retryable_exceptions):
        return False

    # Check error code if it's an AURA error
    if hasattr(exception, "code"):
        code = getattr(exception, "code", "")
        if policy.retryable_codes and code not in policy.retryable_codes:
            return False

    return True


def _calculate_delay(attempt: int, policy: RetryPolicy) -> float:
    """Calculate delay for a retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        policy: Retry policy

    Returns:
        Delay in seconds
    """
    if policy.strategy == RetryStrategy.IMMEDIATE:
        return 0.0

    if policy.strategy == RetryStrategy.FIXED:
        delay = policy.base_delay
    elif policy.strategy == RetryStrategy.LINEAR:
        delay = policy.base_delay * (attempt + 1)
    else:  # EXPONENTIAL_BACKOFF
        delay = policy.base_delay * (policy.exponential_base**attempt)

    # Apply max delay cap
    delay = min(delay, policy.max_delay)

    # Apply jitter
    if policy.jitter:
        jitter_amount = delay * policy.jitter_factor * random.random()
        delay += jitter_amount

    return delay


# =============================================================================
# Decorators
# =============================================================================


def retry(
    policy: Optional[RetryPolicy] = None,
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    retryable_exceptions: Optional[tuple] = None,
):
    """Decorator to add retry logic to async functions.

    Args:
        policy: Retry policy to use
        max_retries: Override max retries from policy
        base_delay: Override base delay from policy
        retryable_exceptions: Override retryable exceptions

    Returns:
        Decorator function

    Example:
        >>> @retry(policy=RetryPolicies.NETWORK)
        ... async def fetch_data(url: str) -> dict:
        ...     return await http_get(url)
    """

    def decorator(fn: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, RetryResult[T]]]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> RetryResult[T]:
            # Build effective policy
            effective_policy = policy or RetryPolicies.DEFAULT

            # Apply overrides
            policy_kwargs = {}
            if max_retries is not None:
                policy_kwargs["max_retries"] = max_retries
            if base_delay is not None:
                policy_kwargs["base_delay"] = base_delay
            if retryable_exceptions is not None:
                policy_kwargs["retryable_exceptions"] = retryable_exceptions

            if policy_kwargs:
                effective_policy = RetryPolicy(**{**effective_policy.__dict__, **policy_kwargs})

            return await with_retry(fn, *args, policy=effective_policy, **kwargs)

        return wrapper

    return decorator


def retry_sync(
    policy: Optional[RetryPolicy] = None,
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    retryable_exceptions: Optional[tuple] = None,
):
    """Decorator to add retry logic to sync functions.

    Args:
        policy: Retry policy to use
        max_retries: Override max retries from policy
        base_delay: Override base delay from policy
        retryable_exceptions: Override retryable exceptions

    Returns:
        Decorator function
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., RetryResult[T]]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> RetryResult[T]:
            effective_policy = policy or RetryPolicies.DEFAULT

            policy_kwargs = {}
            if max_retries is not None:
                policy_kwargs["max_retries"] = max_retries
            if base_delay is not None:
                policy_kwargs["base_delay"] = base_delay
            if retryable_exceptions is not None:
                policy_kwargs["retryable_exceptions"] = retryable_exceptions

            if policy_kwargs:
                effective_policy = RetryPolicy(**{**effective_policy.__dict__, **policy_kwargs})

            return with_retry_sync(fn, *args, policy=effective_policy, **kwargs)

        return wrapper

    return decorator


def retry_or_raise(
    policy: Optional[RetryPolicy] = None,
    **kwargs,
):
    """Decorator to add retry logic that raises on failure.

    Args:
        policy: Retry policy to use
        **kwargs: Additional arguments passed to retry()

    Returns:
        Decorator function
    """

    def decorator(fn: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(fn)
        async def wrapper(*args, **fn_kwargs) -> T:
            @retry(policy=policy, **kwargs)
            async def _wrapper(*a, **kw):
                return await fn(*a, **kw)

            result = await _wrapper(*args, **fn_kwargs)
            return result.unwrap()

        return wrapper

    return decorator


# =============================================================================
# Context Manager
# =============================================================================


class RetryContext:
    """Context manager for retry operations.

    Example:
        >>> async with RetryContext(RetryPolicies.NETWORK) as retry_ctx:
        ...     result = await retry_ctx.run(fetch_data, url)
        ...     if result.success:
        ...         print(result.result)
    """

    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicies.DEFAULT
        self.state = RetryState(max_attempts=self.policy.max_retries + 1)
        self.results: List[RetryResult] = []

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def run(
        self,
        fn: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> RetryResult[T]:
        """Run a function with retry within this context."""
        result = await with_retry(fn, *args, policy=self.policy, retry_state=self.state, **kwargs)
        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all retry operations in this context."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful

        return {
            "total_operations": total,
            "successful": successful,
            "failed": failed,
            "total_attempts": sum(r.attempts for r in self.results),
            "cumulative_delay": self.state.cumulative_delay,
        }


# =============================================================================
# Agent SDK Integration
# =============================================================================


async def with_agent_sdk_retry(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> T:
    """Use agent_sdk retry if available, otherwise use local implementation.

    This provides a bridge to the core.agent_sdk.resilience module.

    Args:
        fn: Function to execute
        *args: Positional arguments
        max_attempts: Maximum retry attempts
        base_delay: Base delay for backoff
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    if AGENT_SDK_AVAILABLE:
        config = AgentSDKRetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
        return await agent_sdk_retry(fn, config, *args, **kwargs)
    else:
        policy = RetryPolicy(
            max_retries=max_attempts - 1,
            base_delay=base_delay,
        )
        result = await with_retry(fn, *args, policy=policy, **kwargs)
        return result.unwrap()


# =============================================================================
# Utility Functions
# =============================================================================


def is_transient_error(error: Exception) -> bool:
    """Check if an error is likely transient and worth retrying.

    Args:
        error: Exception to check

    Returns:
        True if error appears transient
    """
    transient_types = (
        ConnectionError,
        TimeoutError,
        MCPConnectionError,
        MCPTimeoutError,
        RateLimitError,
    )

    if isinstance(error, transient_types):
        return True

    # Check error message for transient indicators
    error_msg = str(error).lower()
    transient_indicators = [
        "temporary",
        "transient",
        "timeout",
        "connection",
        "unavailable",
        "rate limit",
        "too many requests",
        "try again",
        "retry",
    ]

    return any(indicator in error_msg for indicator in transient_indicators)


def get_retry_recommendation(error: Exception) -> str:
    """Get a retry recommendation for an error.

    Args:
        error: The error that occurred

    Returns:
        Human-readable recommendation
    """
    if isinstance(error, RateLimitError):
        retry_after = getattr(error, "retry_after", None)
        if retry_after:
            return f"Wait {retry_after}s before retrying (rate limited)"
        return "Wait a moment before retrying (rate limited)"

    if isinstance(error, (MCPConnectionError, ConnectionError)):
        return "Check network connection and retry"

    if isinstance(error, (MCPTimeoutError, TimeoutError)):
        return "The service is slow; consider retrying with a longer timeout"

    if is_transient_error(error):
        return "This error may be transient; retrying may succeed"

    return "This error may require intervention before retrying"


# =============================================================================
# Backward Compatibility
# =============================================================================

# Legacy function names for compatibility
async_retry = with_retry
sync_retry = with_retry_sync

__all__ = [
    # Core functions
    "with_retry",
    "with_retry_sync",
    "with_retry_or_raise",
    "with_retry_sync_or_raise",
    # Configuration
    "RetryPolicy",
    "RetryPolicies",
    "RetryStrategy",
    "RetryState",
    "RetryResult",
    "RetryContext",
    # Decorators
    "retry",
    "retry_sync",
    "retry_or_raise",
    # Utilities
    "is_transient_error",
    "get_retry_recommendation",
    "with_agent_sdk_retry",
    # Legacy aliases
    "async_retry",
    "sync_retry",
]
