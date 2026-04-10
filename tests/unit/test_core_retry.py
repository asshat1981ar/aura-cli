"""Unit tests for core/retry.py module.

Tests cover:
- Retry policy configuration
- Retry execution with various outcomes
- Exponential backoff calculation
- Circuit breaker integration
- Decorator functionality
- Retry state tracking
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call

from core.retry import (
    RetryPolicy,
    RetryPolicies,
    RetryStrategy,
    RetryState,
    RetryResult,
    RetryContext,
    with_retry,
    with_retry_sync,
    with_retry_or_raise,
    retry,
    retry_sync,
    is_transient_error,
    get_retry_recommendation,
    _calculate_delay,
    _should_retry,
)
from core.exceptions import (
    MCPConnectionError,
    RateLimitError,
    ConfigNotFoundError,
)


class TestRetryPolicy:
    """Tests for RetryPolicy configuration."""
    
    def test_default_policy(self):
        """Test default policy values."""
        policy = RetryPolicy()
        
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True
        assert policy.jitter_factor == 0.5
        assert policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
    
    def test_custom_policy(self):
        """Test custom policy configuration."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.LINEAR,
        )
        
        assert policy.max_retries == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 120.0
        assert policy.strategy == RetryStrategy.LINEAR
    
    def test_invalid_max_retries(self):
        """Test validation of negative max_retries."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryPolicy(max_retries=-1)
    
    def test_invalid_base_delay(self):
        """Test validation of negative base_delay."""
        with pytest.raises(ValueError, match="base_delay must be >= 0"):
            RetryPolicy(base_delay=-1.0)
    
    def test_invalid_max_delay(self):
        """Test validation of max_delay less than base_delay."""
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetryPolicy(base_delay=10.0, max_delay=5.0)
    
    def test_invalid_jitter_factor(self):
        """Test validation of jitter_factor out of range."""
        with pytest.raises(ValueError, match="jitter_factor must be between 0 and 1"):
            RetryPolicy(jitter_factor=1.5)


class TestRetryPolicies:
    """Tests for predefined retry policies."""
    
    def test_default_policy(self):
        """Test DEFAULT policy."""
        policy = RetryPolicies.DEFAULT
        assert policy.max_retries == 3
    
    def test_aggressive_policy(self):
        """Test AGGRESSIVE policy."""
        policy = RetryPolicies.AGGRESSIVE
        assert policy.max_retries == 5
        assert policy.base_delay == 0.5
    
    def test_conservative_policy(self):
        """Test CONSERVATIVE policy."""
        policy = RetryPolicies.CONSERVATIVE
        assert policy.max_retries == 3
        assert policy.base_delay == 2.0
    
    def test_no_retry_policy(self):
        """Test NO_RETRY policy."""
        policy = RetryPolicies.NO_RETRY
        assert policy.max_retries == 0
    
    def test_network_policy(self):
        """Test NETWORK policy."""
        policy = RetryPolicies.NETWORK
        assert MCPConnectionError in policy.retryable_exceptions
        assert "AURA-300" in policy.retryable_codes
    
    def test_mcp_policy(self):
        """Test MCP policy."""
        policy = RetryPolicies.MCP
        assert policy.max_retries == 3
        assert MCPConnectionError in policy.retryable_exceptions


class TestRetryState:
    """Tests for RetryState tracking."""
    
    def test_default_state(self):
        """Test default state values."""
        state = RetryState()
        
        assert state.attempt == 0
        assert state.max_attempts == 0
        assert state.last_exception is None
        assert state.cumulative_delay == 0.0
    
    def test_state_properties(self):
        """Test state boolean properties."""
        state = RetryState(max_attempts=3)
        
        assert state.is_first_attempt is True
        assert state.is_last_attempt is False
        
        state.attempt = 2
        assert state.is_first_attempt is False
        assert state.is_last_attempt is True
    
    def test_elapsed_time(self):
        """Test elapsed time tracking."""
        state = RetryState()
        
        # Should be very small immediately after creation
        assert state.elapsed_time >= 0.0
        assert state.elapsed_time < 1.0
    
    def test_to_dict(self):
        """Test state serialization."""
        state = RetryState(max_attempts=3)
        state.attempt = 1
        state.cumulative_delay = 2.5
        
        result = state.to_dict()
        
        assert result["attempt"] == 1
        assert result["max_attempts"] == 3
        assert result["cumulative_delay"] == 2.5
        assert "elapsed_time" in result


class TestRetryResult:
    """Tests for RetryResult dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        result = RetryResult(success=True, result="data", attempts=1)
        
        assert result.success is True
        assert result.failed is False
        assert result.unwrap() == "data"
        assert result.unwrap_or("default") == "data"
    
    def test_failure_result(self):
        """Test failed result."""
        error = ValueError("test")
        result = RetryResult(success=False, exception=error, attempts=3)
        
        assert result.success is False
        assert result.failed is True
        
        with pytest.raises(ValueError, match="test"):
            result.unwrap()
    
    def test_unwrap_or_default(self):
        """Test unwrap_or with default."""
        result = RetryResult(success=False, exception=ValueError("test"))
        
        assert result.unwrap_or("default") == "default"
    
    def test_unwrap_or_else(self):
        """Test unwrap_or_else with function."""
        result = RetryResult(success=False, exception=ValueError("test"))
        
        def handler(exc):
            return f"handled: {exc}"
        
        assert result.unwrap_or_else(handler) == "handled: test"


class TestWithRetry:
    """Tests for with_retry async function."""
    
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test success on first attempt."""
        mock_fn = AsyncMock(return_value="success")
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        
        result = await with_retry(mock_fn, policy=policy)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1
        assert mock_fn.call_count == 1
    
    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Test success after retry."""
        mock_fn = AsyncMock(side_effect=[Exception("fail"), "success"])
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
        
        result = await with_retry(mock_fn, policy=policy)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 2
    
    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """Test when all retries fail."""
        mock_fn = AsyncMock(side_effect=Exception("always fails"))
        policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=False)
        
        result = await with_retry(mock_fn, policy=policy)
        
        assert result.success is False
        assert result.attempts == 3  # 1 initial + 2 retries
        assert "always fails" in str(result.exception)
    
    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test non-retryable exception fails fast."""
        class NonRetryableError(Exception):
            pass
        
        mock_fn = AsyncMock(side_effect=NonRetryableError("fail"))
        policy = RetryPolicy(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),  # NonRetryableError not included
        )
        
        result = await with_retry(mock_fn, policy=policy)
        
        assert result.success is False
        assert result.attempts == 1  # No retries
    
    @pytest.mark.asyncio
    async def test_retry_by_error_code(self):
        """Test retry based on error code."""
        error = MCPConnectionError("connection failed")
        error.code = "AURA-301"
        
        mock_fn = AsyncMock(side_effect=[error, "success"])
        policy = RetryPolicy(
            max_retries=3,
            base_delay=0.01,
            jitter=False,
            retryable_codes={"AURA-301"},
        )
        
        result = await with_retry(mock_fn, policy=policy)
        
        assert result.success is True
        assert result.attempts == 2


class TestWithRetrySync:
    """Tests for with_retry_sync function."""
    
    def test_success_on_first_attempt(self):
        """Test success on first attempt."""
        mock_fn = Mock(return_value="success")
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        
        result = with_retry_sync(mock_fn, policy=policy)
        
        assert result.success is True
        assert result.result == "success"
    
    def test_success_after_retry(self):
        """Test success after retry."""
        mock_fn = Mock(side_effect=[Exception("fail"), "success"])
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
        
        result = with_retry_sync(mock_fn, policy=policy)
        
        assert result.success is True
        assert result.attempts == 2


class TestWithRetryOrRaise:
    """Tests for with_retry_or_raise function."""
    
    @pytest.mark.asyncio
    async def test_success_returns_result(self):
        """Test successful execution returns result."""
        mock_fn = AsyncMock(return_value="success")
        policy = RetryPolicy(max_retries=1, base_delay=0.01)
        
        result = await with_retry_or_raise(mock_fn, policy=policy)
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_failure_raises(self):
        """Test failure raises exception."""
        mock_fn = AsyncMock(side_effect=ValueError("fail"))
        policy = RetryPolicy(max_retries=1, base_delay=0.01, jitter=False)
        
        with pytest.raises(ValueError, match="fail"):
            await with_retry_or_raise(mock_fn, policy=policy)


class TestCalculateDelay:
    """Tests for _calculate_delay function."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(
            base_delay=1.0,
            exponential_base=2.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=False,
        )
        
        assert _calculate_delay(0, policy) == 1.0
        assert _calculate_delay(1, policy) == 2.0
        assert _calculate_delay(2, policy) == 4.0
    
    def test_linear_delay(self):
        """Test linear delay calculation."""
        policy = RetryPolicy(
            base_delay=1.0,
            strategy=RetryStrategy.LINEAR,
            jitter=False,
        )
        
        assert _calculate_delay(0, policy) == 1.0
        assert _calculate_delay(1, policy) == 2.0
        assert _calculate_delay(2, policy) == 3.0
    
    def test_fixed_delay(self):
        """Test fixed delay calculation."""
        policy = RetryPolicy(
            base_delay=2.0,
            strategy=RetryStrategy.FIXED,
            jitter=False,
        )
        
        assert _calculate_delay(0, policy) == 2.0
        assert _calculate_delay(5, policy) == 2.0
    
    def test_immediate_delay(self):
        """Test immediate (no) delay."""
        policy = RetryPolicy(strategy=RetryStrategy.IMMEDIATE)
        
        assert _calculate_delay(0, policy) == 0.0
        assert _calculate_delay(10, policy) == 0.0
    
    def test_max_delay_cap(self):
        """Test max delay is respected."""
        policy = RetryPolicy(
            base_delay=1.0,
            exponential_base=10.0,
            max_delay=30.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=False,
        )
        
        # Without cap this would be 100.0
        assert _calculate_delay(2, policy) == 30.0
    
    def test_jitter_adds_variance(self):
        """Test jitter adds random variance."""
        policy = RetryPolicy(
            base_delay=10.0,
            jitter=True,
            jitter_factor=0.5,
            strategy=RetryStrategy.FIXED,
        )
        
        delays = [_calculate_delay(0, policy) for _ in range(10)]
        
        # All delays should be between 10.0 and 15.0 (10.0 + 50%)
        for delay in delays:
            assert 10.0 <= delay <= 15.0
        
        # Not all should be identical (jitter working)
        assert len(set(delays)) > 1


class TestShouldRetry:
    """Tests for _should_retry function."""
    
    def test_retryable_exception(self):
        """Test retryable exception type."""
        policy = RetryPolicy(retryable_exceptions=(ValueError,))
        error = ValueError("test")
        
        assert _should_retry(error, policy) is True
    
    def test_non_retryable_exception(self):
        """Test non-retryable exception type."""
        policy = RetryPolicy(retryable_exceptions=(ValueError,))
        error = TypeError("test")
        
        assert _should_retry(error, policy) is False
    
    def test_retry_by_error_code_match(self):
        """Test retry when error code matches."""
        policy = RetryPolicy(
            retryable_exceptions=(Exception,),
            retryable_codes={"AURA-100"},
        )
        error = ConfigNotFoundError("test")
        error.code = "AURA-100"
        
        assert _should_retry(error, policy) is True
    
    def test_retry_by_error_code_no_match(self):
        """Test no retry when error code doesn't match."""
        policy = RetryPolicy(
            retryable_exceptions=(Exception,),
            retryable_codes={"AURA-100"},
        )
        error = ValueError("test")
        error.code = "AURA-999"
        
        assert _should_retry(error, policy) is False


class TestRetryDecorator:
    """Tests for retry decorators."""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        
        @retry(policy=RetryPolicy(max_retries=2, base_delay=0.01))
        async def success_fn():
            return "success"
        
        result = await success_fn()
        
        assert result.success is True
        assert result.result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_decorator_eventual_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @retry(policy=RetryPolicy(max_retries=3, base_delay=0.01, jitter=False))
        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        result = await flaky_fn()
        
        assert result.success is True
        assert result.attempts == 3
    
    def test_retry_sync_decorator(self):
        """Test sync retry decorator."""
        
        @retry_sync(policy=RetryPolicy(max_retries=2, base_delay=0.01))
        def sync_fn():
            return "success"
        
        result = sync_fn()
        
        assert result.success is True
        assert result.result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_decorator_max_retries_override(self):
        """Test retry decorator with max_retries override."""
        
        @retry(max_retries=1, base_delay=0.01)
        async def test_fn():
            raise ValueError("fail")
        
        result = await test_fn()
        
        assert result.success is False
        assert result.attempts == 2  # 1 initial + 1 retry


class TestRetryContext:
    """Tests for RetryContext."""
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test RetryContext as async context manager."""
        async with RetryContext(RetryPolicy(max_retries=1, base_delay=0.01)) as ctx:
            assert ctx.policy is not None
            assert ctx.state is not None
    
    @pytest.mark.asyncio
    async def test_run_in_context(self):
        """Test running function in context."""
        async with RetryContext(RetryPolicy(max_retries=1, base_delay=0.01)) as ctx:
            async def test_fn():
                return "success"
            
            result = await ctx.run(test_fn)
            
            assert result.success is True
            assert len(ctx.results) == 1
    
    @pytest.mark.asyncio
    async def test_get_summary(self):
        """Test getting summary from context."""
        async with RetryContext(RetryPolicy(max_retries=1, base_delay=0.01)) as ctx:
            async def success_fn():
                return "success"
            
            async def fail_fn():
                raise ValueError("fail")
            
            await ctx.run(success_fn)
            await ctx.run(fail_fn)
            
            summary = ctx.get_summary()
            
            assert summary["total_operations"] == 2
            assert summary["successful"] == 1
            assert summary["failed"] == 1
            # success_fn: 1 attempt, fail_fn: 2 attempts (1 initial + 1 retry)
            assert summary["total_attempts"] == 3


class TestIsTransientError:
    """Tests for is_transient_error function."""
    
    def test_transient_connection_error(self):
        """Test connection error is transient."""
        assert is_transient_error(ConnectionError("test")) is True
    
    def test_transient_timeout_error(self):
        """Test timeout error is transient."""
        assert is_transient_error(TimeoutError("test")) is True
    
    def test_transient_mcp_error(self):
        """Test MCP error is transient."""
        assert is_transient_error(MCPConnectionError("test")) is True
    
    def test_transient_rate_limit(self):
        """Test rate limit error is transient."""
        assert is_transient_error(RateLimitError("test")) is True
    
    def test_non_transient_error(self):
        """Test regular error is not transient."""
        assert is_transient_error(ValueError("test")) is False
    
    def test_transient_by_message(self):
        """Test error with transient message."""
        assert is_transient_error(ValueError("temporary failure")) is True
        assert is_transient_error(ValueError("try again later")) is True


class TestGetRetryRecommendation:
    """Tests for get_retry_recommendation function."""
    
    def test_rate_limit_recommendation(self):
        """Test recommendation for rate limit."""
        error = RateLimitError("rate limited")
        error.retry_after = 60
        
        rec = get_retry_recommendation(error)
        
        assert "Wait 60s" in rec
    
    def test_connection_error_recommendation(self):
        """Test recommendation for connection error."""
        error = MCPConnectionError("connection failed")
        
        rec = get_retry_recommendation(error)
        
        assert "network" in rec.lower()
    
    def test_timeout_recommendation(self):
        """Test recommendation for timeout."""
        error = TimeoutError("timed out")
        
        rec = get_retry_recommendation(error)
        
        assert "slow" in rec.lower() or "timeout" in rec.lower()
    
    def test_generic_transient_recommendation(self):
        """Test recommendation for generic transient error."""
        error = ConnectionError("network issue")
        
        rec = get_retry_recommendation(error)
        
        assert "transient" in rec.lower() or "retry" in rec.lower()
    
    def test_non_transient_recommendation(self):
        """Test recommendation for non-transient error."""
        error = ValueError("validation failed")
        
        rec = get_retry_recommendation(error)
        
        assert "intervention" in rec.lower() or "require" in rec.lower()


class TestRetryPolicyCallbacks:
    """Tests for retry policy callbacks."""
    
    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callback_calls = []
        
        def on_retry(attempt, exception, delay):
            callback_calls.append((attempt, str(exception), delay))
        
        policy = RetryPolicy(
            max_retries=2,
            base_delay=0.01,
            jitter=False,
            on_retry=on_retry,
        )
        
        mock_fn = AsyncMock(side_effect=[Exception("fail"), "success"])
        
        await with_retry(mock_fn, policy=policy)
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 0
        assert "fail" in callback_calls[0][1]
    
    @pytest.mark.asyncio
    async def test_on_exhausted_callback(self):
        """Test on_exhausted callback is called."""
        callback_calls = []
        
        def on_exhausted(attempts, exception):
            callback_calls.append((attempts, str(exception)))
        
        policy = RetryPolicy(
            max_retries=1,
            base_delay=0.01,
            jitter=False,
            on_exhausted=on_exhausted,
        )
        
        mock_fn = AsyncMock(side_effect=Exception("always fails"))
        
        await with_retry(mock_fn, policy=policy)
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 2  # 1 initial + 1 retry
