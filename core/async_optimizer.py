"""Async execution optimizations for performance."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, List, TypeVar, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import time

from core.logging_utils import log_json

T = TypeVar('T')

# Thread pool for CPU-bound operations
_thread_pool = ThreadPoolExecutor(max_workers=4)


async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a synchronous function in a thread pool.
    
    Args:
        func: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, lambda: func(*args, **kwargs))


async def gather_with_concurrency(
    limit: int,
    *coros: asyncio.Coroutine,
) -> List[Any]:
    """Gather coroutines with concurrency limit.
    
    Args:
        limit: Maximum concurrent executions
        *coros: Coroutines to execute
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*(sem_coro(c) for c in coros))


async def batch_process(
    items: Iterable[T],
    processor: Callable[[T], asyncio.Coroutine],
    batch_size: int = 10,
    concurrency: int = 5,
) -> List[Any]:
    """Process items in batches with controlled concurrency.
    
    Args:
        items: Items to process
        processor: Async processor function
        batch_size: Number of items per batch
        concurrency: Max concurrent processors
        
    Returns:
        List of results
    """
    items_list = list(items)
    results = []
    
    for i in range(0, len(items_list), batch_size):
        batch = items_list[i:i + batch_size]
        
        start_time = time.time()
        batch_results = await gather_with_concurrency(
            concurrency,
            *(processor(item) for item in batch),
        )
        elapsed = time.time() - start_time
        
        log_json("DEBUG", "batch_processed", {
            "batch_size": len(batch),
            "elapsed_ms": round(elapsed * 1000, 2),
        })
        
        results.extend(batch_results)
    
    return results


def async_timed(name: str):
    """Decorator to time async function execution.
    
    Args:
        name: Operation name for logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                log_json("DEBUG", "async_operation_timed", {
                    "name": name,
                    "elapsed_ms": round(elapsed * 1000, 2),
                })
        return wrapper
    return decorator


class AsyncLazy:
    """Lazy initialization for expensive async resources."""
    
    def __init__(self, factory: Callable[[], asyncio.Coroutine]):
        self._factory = factory
        self._value: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def get(self) -> Any:
        """Get or create the value."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    self._value = await self._factory()
                    self._initialized = True
        return self._value
    
    async def reset(self) -> None:
        """Reset the lazy value."""
        async with self._lock:
            self._value = None
            self._initialized = False


class ConnectionPool:
    """Pool for reusable async connections."""
    
    def __init__(
        self,
        factory: Callable[[], asyncio.Coroutine],
        max_size: int = 10,
    ):
        self._factory = factory
        self._max_size = max_size
        self._available: asyncio.Queue = asyncio.Queue()
        self._in_use: set = set()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        async with self._lock:
            # Try to get from available
            try:
                conn = self._available.get_nowait()
                self._in_use.add(id(conn))
                return conn
            except asyncio.QueueEmpty:
                pass
            
            # Create new if under limit
            if len(self._in_use) < self._max_size:
                conn = await self._factory()
                self._in_use.add(id(conn))
                return conn
        
        # Wait for available connection
        conn = await self._available.get()
        async with self._lock:
            self._in_use.add(id(conn))
        return conn
    
    async def release(self, conn: Any) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            self._in_use.discard(id(conn))
        await self._available.put(conn)
    
    async def __aenter__(self):
        self._conn = await self.acquire()
        return self._conn
    
    async def __aexit__(self, *args):
        await self.release(self._conn)


def debounce(delay: float):
    """Debounce decorator for async functions.
    
    Args:
        delay: Minimum delay between executions in seconds
    """
    def decorator(func: Callable) -> Callable:
        last_call = 0
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal last_call
            now = time.time()
            
            if now - last_call < delay:
                return None
            
            last_call = now
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def throttle(max_calls: int, period: float):
    """Throttle decorator to limit call rate.
    
    Args:
        max_calls: Maximum calls allowed
        period: Time period in seconds
    """
    def decorator(func: Callable) -> Callable:
        calls: List[float] = []
        lock = asyncio.Lock()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal calls
            now = time.time()
            
            async with lock:
                # Remove old calls outside the period
                calls = [c for c in calls if now - c < period]
                
                if len(calls) >= max_calls:
                    # Wait until we can make another call
                    wait_time = period - (now - calls[0])
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        now = time.time()
                        calls = [c for c in calls if now - c < period]
                
                calls.append(now)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
