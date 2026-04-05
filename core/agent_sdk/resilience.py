"""Resilience patterns for Agent SDK: circuit breakers, retries, health checks.

Production hardening for Issue #378 - Advanced Agent Orchestration.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before half-open
    half_open_max_calls: int = 3        # Test calls in half-open
    success_threshold: int = 2          # Successes to close


class CircuitBreaker:
    """Circuit breaker for MCP server calls.
    
    Prevents cascading failures by rejecting requests to failing services.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()
        self._half_open_calls = 0
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            await self._transition_state()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit {self.name} HALF_OPEN limit reached"
                    )
                self._half_open_calls += 1
        
        # Execute outside lock
        try:
            result = await asyncio.wait_for(
                self._asyncify(func, *args, **kwargs),
                timeout=kwargs.pop('_timeout', 30.0)
            )
            await self._record_success()
            return result
        except Exception:
            await self._record_failure()
            raise
    
    async def _transition_state(self) -> None:
        """Check and transition circuit state."""
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) >= self.config.recovery_timeout:
                logger.info("Circuit %s transitioning to HALF_OPEN", self.name)
                self.state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self.success_count = 0
    
    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info("Circuit %s transitioning to CLOSED", self.name)
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self._half_open_calls = 0
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self) -> None:
        """Record failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit %s HALF_OPEN failed, returning to OPEN", self.name)
                self.state = CircuitState.OPEN
                self._half_open_calls = 0
            elif self.failure_count >= self.config.failure_threshold:
                logger.error("Circuit %s transitioning to OPEN (%d failures)", 
                           self.name, self.failure_count)
                self.state = CircuitState.OPEN
    
    def _asyncify(self, func: Callable[..., T], *args, **kwargs) -> asyncio.Future[T]:
        """Convert function to async if needed."""
        if asyncio.iscoroutinefunction(func):
            return func(*args, **kwargs)
        # Run sync function in thread pool
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit state for health checks."""
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


def _get_func_name(func: Callable) -> str:
    """Get name of function, handling mocks."""
    return getattr(func, '__name__', getattr(func, '_mock_name', repr(func)))


async def retry_with_backoff(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> T:
    """Execute function with exponential backoff retry.
    
    Args:
        func: Function to execute
        config: Retry configuration
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries exhausted
    """
    cfg = config or RetryConfig()
    last_exception: Optional[Exception] = None
    func_name = _get_func_name(func)
    
    for attempt in range(cfg.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        except cfg.retryable_exceptions as e:
            last_exception = e
            if attempt == cfg.max_attempts - 1:
                logger.error("Retry exhausted for %s: %s", func_name, e)
                raise
            
            # Calculate delay with exponential backoff
            delay = min(
                cfg.base_delay * (cfg.exponential_base ** attempt),
                cfg.max_delay
            )
            if cfg.jitter:
                delay *= (0.5 + random.random() * 0.5)  # Add 50-100% jitter
            
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %.2fs...",
                attempt + 1, cfg.max_attempts, func_name, e, delay
            )
            await asyncio.sleep(delay)
    
    raise last_exception or RuntimeError("Unexpected retry exit")


@dataclass
class MCPServerHealth:
    """Health status for an MCP server."""
    name: str
    url: str
    healthy: bool
    response_time_ms: float
    last_check: float
    error: Optional[str] = None
    consecutive_failures: int = 0


class MCPHealthMonitor:
    """Monitor health of MCP servers."""
    
    def __init__(self, check_interval: float = 30.0, timeout: float = 5.0) -> None:
        self.check_interval = check_interval
        self.timeout = timeout
        self._servers: Dict[str, str] = {}
        self._health: Dict[str, MCPServerHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_server(self, name: str, url: str) -> None:
        """Register an MCP server for health monitoring."""
        self._servers[name] = url
        self._circuit_breakers[name] = CircuitBreaker(
            name=f"mcp-{name}",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0
            )
        )
        logger.info("Registered MCP server %s at %s", name, url)
    
    async def start(self) -> None:
        """Start health monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("MCP health monitor started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MCP health monitor stopped")
    
    async def _monitor_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self._check_all_servers()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error: %s", e)
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_servers(self) -> None:
        """Check health of all registered servers."""
        tasks = [
            self._check_server(name, url)
            for name, url in self._servers.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_server(self, name: str, url: str) -> None:
        """Check health of a single server."""
        start = time.time()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/health",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    elapsed_ms = (time.time() - start) * 1000
                    healthy = resp.status == 200
                    
                    previous = self._health.get(name)
                    consecutive = 0 if healthy else (previous.consecutive_failures + 1 if previous else 1)
                    
                    self._health[name] = MCPServerHealth(
                        name=name,
                        url=url,
                        healthy=healthy,
                        response_time_ms=elapsed_ms,
                        last_check=time.time(),
                        consecutive_failures=consecutive
                    )
                    
                    if not healthy:
                        logger.warning("MCP server %s unhealthy (status %d)", 
                                     name, resp.status)
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            previous = self._health.get(name)
            consecutive = previous.consecutive_failures + 1 if previous else 1
            
            self._health[name] = MCPServerHealth(
                name=name,
                url=url,
                healthy=False,
                response_time_ms=elapsed_ms,
                last_check=time.time(),
                error=str(e),
                consecutive_failures=consecutive
            )
            
            if consecutive == 1 or consecutive % 5 == 0:
                logger.error("MCP server %s health check failed: %s", name, e)
    
    def get_health(self, name: Optional[str] = None) -> Union[MCPServerHealth, Dict[str, MCPServerHealth]]:
        """Get health status for one or all servers."""
        if name:
            return self._health.get(name) or MCPServerHealth(
                name=name, url="", healthy=False, 
                response_time_ms=0, last_check=0, error="Not registered"
            )
        return dict(self._health)
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a server."""
        return self._circuit_breakers.get(name)
    
    def is_healthy(self, name: str) -> bool:
        """Check if a server is currently healthy."""
        health = self._health.get(name)
        return health.healthy if health else False


class ResilientMCPClient:
    """MCP client with circuit breaker and retry resilience.
    
    Wraps MCP tool invocations with:
    - Circuit breaker pattern
    - Exponential backoff retry
    - Health check integration
    - Timeout enforcement
    """
    
    def __init__(
        self,
        health_monitor: Optional[MCPHealthMonitor] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> None:
        self.health_monitor = health_monitor
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            retryable_exceptions=(
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
            )
        )
        self._default_timeout = 30.0
    
    async def invoke(
        self,
        server: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Invoke an MCP tool with resilience.
        
        Args:
            server: MCP server name
            tool_name: Tool to invoke
            tool_args: Tool arguments
            timeout: Request timeout
        
        Returns:
            Tool response
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            TimeoutError: If request times out
            Various exceptions on failure
        """
        timeout = timeout or self._default_timeout
        
        # Check health monitor if available
        if self.health_monitor and not self.health_monitor.is_healthy(server):
            logger.warning("MCP server %s marked unhealthy, attempting anyway", server)
        
        # Get circuit breaker
        breaker = None
        if self.health_monitor:
            breaker = self.health_monitor.get_circuit_breaker(server)
        
        # Build the invocation function
        def _invoke():
            import requests
            cfg = self._get_config()
            port = cfg.mcp_ports.get(server)
            if not port:
                raise ValueError(f"Unknown MCP server: {server}")
            
            resp = requests.post(
                f"http://localhost:{port}/call",
                json={"tool_name": tool_name, "args": tool_args},
                timeout=timeout
            )
            return resp.json() if resp.ok else {"error": f"Server returned {resp.status_code}"}
        
        # Execute with circuit breaker and retry
        if breaker:
            return await breaker.call(
                lambda: asyncio.run(retry_with_backoff(_invoke, self.retry_config))
            )
        else:
            return await retry_with_backoff(_invoke, self.retry_config)
    
    def _get_config(self):
        """Get Agent SDK config."""
        from core.agent_sdk.config import AgentSDKConfig
        return AgentSDKConfig()


# Global health monitor instance
_health_monitor: Optional[MCPHealthMonitor] = None


def get_health_monitor() -> MCPHealthMonitor:
    """Get or create global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = MCPHealthMonitor()
    return _health_monitor


def reset_health_monitor() -> None:
    """Reset global health monitor (for testing)."""
    global _health_monitor
    _health_monitor = None
