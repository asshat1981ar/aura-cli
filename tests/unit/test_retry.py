"""Unit tests for retry logic and resilience patterns.

Tests cover:
- Exponential backoff
- Max retry limits
- Retryable exception filtering
- Circuit breaker patterns
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from core.agent_sdk.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    RetryConfig,
    retry_with_backoff,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=30.0,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        mock_fn = AsyncMock(return_value="success")
        config = RetryConfig(max_attempts=3, base_delay=0.01)

        result = await retry_with_backoff(mock_fn, config)

        assert result == "success"
        assert mock_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Test function succeeds after retry."""
        mock_fn = AsyncMock(side_effect=[Exception("fail"), "success"])
        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        result = await retry_with_backoff(mock_fn, config)

        assert result == "success"
        assert mock_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_exhaust_all_retries(self):
        """Test function exhausts all retries and raises."""
        mock_fn = AsyncMock(side_effect=Exception("always fails"))
        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        with pytest.raises(Exception, match="always fails"):
            await retry_with_backoff(mock_fn, config)

        assert mock_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_respects_max_attempts(self):
        """Test that max_attempts is respected exactly."""
        mock_fn = AsyncMock(side_effect=Exception("fail"))
        config = RetryConfig(max_attempts=5, base_delay=0.01, jitter=False)

        with pytest.raises(Exception):
            await retry_with_backoff(mock_fn, config)

        assert mock_fn.call_count == 5

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self):
        """Test non-retryable exceptions are not retried."""
        class CustomError(Exception):
            pass

        mock_fn = AsyncMock(side_effect=CustomError("custom error"))
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),  # CustomError not included
        )

        with pytest.raises(CustomError):
            await retry_with_backoff(mock_fn, config)

        assert mock_fn.call_count == 1  # Not retried

    @pytest.mark.asyncio
    async def test_retryable_exception_is_retried(self):
        """Test retryable exceptions are retried."""
        mock_fn = AsyncMock(side_effect=[ValueError("fail"), "success"])
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            jitter=False,
            retryable_exceptions=(ValueError,),
        )

        result = await retry_with_backoff(mock_fn, config)

        assert result == "success"
        assert mock_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_function_wrapping(self):
        """Test that sync functions are wrapped correctly."""
        def sync_fn():
            return "sync result"

        config = RetryConfig(max_attempts=2, base_delay=0.01)

        result = await retry_with_backoff(sync_fn, config)

        assert result == "sync result"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Test circuit starts in CLOSED state."""
        cb = CircuitBreaker("test")

        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful function call."""
        cb = CircuitBreaker("test")
        mock_fn = AsyncMock(return_value="success")

        result = await cb.call(mock_fn)

        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_count_increments(self):
        """Test failure count increments on failure."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))
        mock_fn = AsyncMock(side_effect=Exception("fail"))

        with pytest.raises(Exception):
            await cb.call(mock_fn)

        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2))
        mock_fn = AsyncMock(side_effect=Exception("fail"))

        # First failure
        with pytest.raises(Exception):
            await cb.call(mock_fn)

        # Second failure - should open circuit
        with pytest.raises(Exception):
            await cb.call(mock_fn)

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        mock_fn = AsyncMock(side_effect=Exception("fail"))

        # Open the circuit
        with pytest.raises(Exception):
            await cb.call(mock_fn)

        # Should reject immediately
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(AsyncMock())

    @pytest.mark.asyncio
    async def test_get_state_returns_info(self):
        """Test get_state returns circuit information."""
        cb = CircuitBreaker("test-circuit")

        state = cb.get_state()

        assert state["name"] == "test-circuit"
        assert state["state"] == "CLOSED"
        assert "failure_count" in state


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_success_decrements_failure_count(self):
        """Test successful call decrements failure count in closed state."""
        cb = CircuitBreaker("test")
        cb.failure_count = 2

        mock_fn = AsyncMock(return_value="success")
        await cb.call(mock_fn)

        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_half_open_success(self):
        """Test successful call in half-open moves toward closed."""
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=1,
                recovery_timeout=0.01,
            )
        )

        # Open the circuit
        with pytest.raises(Exception):
            await cb.call(AsyncMock(side_effect=Exception("fail")))

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # First call should be in half-open
        mock_fn = AsyncMock(return_value="success")
        await cb.call(mock_fn)

        # Should be closed after success
        assert cb.state == CircuitState.CLOSED


class TestRetryIntegration:
    """Integration tests for retry patterns."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry combined with circuit breaker."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=5))

        call_count = 0

        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return f"Success on attempt {call_count}"

        # Use circuit breaker to wrap retry
        async def call_with_retry():
            config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)
            return await retry_with_backoff(flaky_function, config)

        result = await cb.call(call_with_retry)

        assert result == "Success on attempt 3"
        assert call_count == 3
