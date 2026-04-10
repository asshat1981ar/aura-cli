"""Tests for Agent SDK resilience patterns (Issue #378)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.agent_sdk.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    MCPHealthMonitor,
    MCPServerHealth,
    ResilientMCPClient,
    RetryConfig,
    get_health_monitor,
    reset_health_monitor,
    retry_with_backoff,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Successful call returns result."""
        cb = CircuitBreaker("test")
        mock_func = MagicMock(return_value="success")

        result = await cb.call(mock_func, "arg1", kwarg="value")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg="value")

    @pytest.mark.asyncio
    async def test_failure_counting(self):
        """Failures are counted correctly."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        for i in range(2):
            with pytest.raises(ValueError):
                await cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            assert cb.failure_count == i + 1
            assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Circuit opens after failure threshold."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))

        # First two failures
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Circuit should now be OPEN
        assert cb.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(lambda: "success")

    @pytest.mark.asyncio
    async def test_half_open_transition(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.01,  # Very short for testing
            ),
        )

        # Cause failure to open circuit
        with pytest.raises(ValueError):
            await cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Next call should transition to HALF_OPEN
        # Mock the internal state check by calling a failing function
        with pytest.raises(ValueError):
            await cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # After failure in half-open, should go back to open
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self):
        """Circuit closes after successes in half-open."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01, success_threshold=1))

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        await asyncio.sleep(0.02)

        # This call should work and close the circuit
        result = await cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_function_support(self):
        """Circuit breaker works with async functions."""
        cb = CircuitBreaker("test")

        async def async_func():
            return "async result"

        result = await cb.call(async_func)
        assert result == "async result"

    def test_get_state(self):
        """State reporting works correctly."""
        cb = CircuitBreaker("test")
        state = cb.get_state()

        assert state["name"] == "test"
        assert state["state"] == "CLOSED"
        assert state["failure_count"] == 0


class TestRetryWithBackoff:
    """Test retry logic."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Successful call doesn't retry."""

        def mock_func():
            mock_func.call_count = getattr(mock_func, "call_count", 0) + 1
            return "success"

        mock_func.__name__ = "mock_func"

        result = await retry_with_backoff(mock_func, RetryConfig(max_attempts=3))

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """Retry after failure, then succeed."""

        def mock_func():
            mock_func.call_count = getattr(mock_func, "call_count", 0) + 1
            if mock_func.call_count == 1:
                raise ValueError("fail")
            return "success"

        mock_func.__name__ = "mock_func"

        result = await retry_with_backoff(mock_func, RetryConfig(max_attempts=3, base_delay=0.01))

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Exhaust all retries and raise."""

        def mock_func():
            mock_func.call_count = getattr(mock_func, "call_count", 0) + 1
            raise ValueError("fail")

        mock_func.__name__ = "mock_func"

        with pytest.raises(ValueError, match="fail"):
            await retry_with_backoff(mock_func, RetryConfig(max_attempts=3, base_delay=0.01))

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Don't retry non-retryable exceptions."""

        def mock_func():
            mock_func.call_count = getattr(mock_func, "call_count", 0) + 1
            raise SyntaxError("fail")

        mock_func.__name__ = "mock_func"

        config = RetryConfig(max_attempts=3, retryable_exceptions=(ValueError,))

        with pytest.raises(SyntaxError):
            await retry_with_backoff(mock_func, config)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Retry works with async functions."""
        call_count = 0

        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return "success"

        result = await retry_with_backoff(async_func, RetryConfig(max_attempts=3, base_delay=0.01))

        assert result == "success"
        assert call_count == 2


class TestMCPHealthMonitor:
    """Test MCP health monitoring."""

    def test_register_server(self):
        """Server registration works."""
        reset_health_monitor()
        monitor = get_health_monitor()

        monitor.register_server("test-server", "http://localhost:8001")

        assert "test-server" in monitor._servers
        assert "test-server" in monitor._circuit_breakers

    def test_is_healthy_initial(self):
        """Server without health check is not healthy."""
        reset_health_monitor()
        monitor = get_health_monitor()

        monitor.register_server("test-server", "http://localhost:8001")

        # No health check done yet
        assert not monitor.is_healthy("test-server")

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Health check marks server as healthy on success."""
        pytest.importorskip("aiohttp", reason="aiohttp not installed")
        reset_health_monitor()
        monitor = MCPHealthMonitor(check_interval=0.01)
        monitor.register_server("test-server", "http://localhost:8001")

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=False)

            # Manually trigger a health check
            await monitor._check_server("test-server", "http://localhost:8001")

            health = monitor.get_health("test-server")
            assert health.healthy

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Health check marks server as unhealthy on failure."""
        pytest.importorskip("aiohttp", reason="aiohttp not installed")
        reset_health_monitor()
        monitor = MCPHealthMonitor()
        monitor.register_server("test-server", "http://localhost:8001")

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=False)

            await monitor._check_server("test-server", "http://localhost:8001")

            health = monitor.get_health("test-server")
            assert not health.healthy

    @pytest.mark.asyncio
    async def test_start_stop_monitor(self):
        """Health monitor can be started and stopped."""
        reset_health_monitor()
        monitor = MCPHealthMonitor(check_interval=0.01)
        monitor.register_server("test", "http://localhost:8001")

        await monitor.start()
        assert monitor._running
        assert monitor._task is not None

        await monitor.stop()
        assert not monitor._running

    def teardown_method(self):
        """Clean up after each test."""
        reset_health_monitor()


class TestResilientMCPClient:
    """Test resilient MCP client."""

    @pytest.mark.asyncio
    async def test_invoke_with_resilience(self):
        """Client invokes with retry and circuit breaker."""
        reset_health_monitor()
        monitor = get_health_monitor()
        monitor.register_server("dev_tools", "http://localhost:8001")

        client = ResilientMCPClient(health_monitor=monitor)

        # Mock the actual HTTP call
        with patch("requests.post") as mock_post:
            mock_post.return_value.ok = True
            mock_post.return_value.json.return_value = {"result": "success"}

            result = await client.invoke("dev_tools", "tool", {"arg": "value"})

            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_invoke_uses_fallback_on_error(self):
        """Client falls back to direct request on resilience error."""
        client = ResilientMCPClient(health_monitor=None)

        with patch("requests.post") as mock_post:
            mock_post.return_value.ok = True
            mock_post.return_value.json.return_value = {"result": "fallback"}

            result = await client.invoke("dev_tools", "tool", {})

            assert result == {"result": "fallback"}

    def test_get_health_monitor_singleton(self):
        """Health monitor is a singleton."""
        reset_health_monitor()
        monitor1 = get_health_monitor()
        monitor2 = get_health_monitor()

        assert monitor1 is monitor2


class TestMCPServerHealth:
    """Test MCP server health dataclass."""

    def test_health_creation(self):
        """Health object can be created."""
        health = MCPServerHealth(name="test", url="http://localhost:8001", healthy=True, response_time_ms=10.5, last_check=1234567890.0)

        assert health.name == "test"
        assert health.healthy
        assert health.response_time_ms == 10.5
