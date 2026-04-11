"""Network connectivity monitoring."""

import asyncio
import socket
from typing import Callable, Optional

from .models import ConnectivityStatus


class ConnectivityMonitor:
    """Monitor network connectivity status."""
    
    DEFAULT_HOSTS = [
        ("8.8.8.8", 53),
        ("1.1.1.1", 53),
    ]
    
    def __init__(
        self,
        check_interval: int = 30,
        hosts: Optional[list[tuple]] = None,
        timeout: int = 5,
    ):
        self.check_interval = check_interval
        self.hosts = hosts or self.DEFAULT_HOSTS.copy()
        self.timeout = timeout
        self._status = ConnectivityStatus.UNKNOWN
        self._callbacks: list[Callable[[ConnectivityStatus], None]] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    @property
    def status(self) -> ConnectivityStatus:
        return self._status
    
    @property
    def is_online(self) -> bool:
        return self._status == ConnectivityStatus.ONLINE
    
    def on_change(self, callback: Callable[[ConnectivityStatus], None]):
        self._callbacks.append(callback)
    
    async def start(self):
        """Start monitoring connectivity."""
        self._running = True
        await self._check()
        self._task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def check_now(self) -> ConnectivityStatus:
        return await self._check()
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check()
            except asyncio.CancelledError:
                break
    
    async def _check(self) -> ConnectivityStatus:
        """Check connectivity by probing hosts."""
        previous_status = self._status
        
        for host, port in self.hosts:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.timeout,
                )
                writer.close()
                await writer.wait_closed()
                self._status = ConnectivityStatus.ONLINE
                break
            except Exception:
                continue
        else:
            self._status = ConnectivityStatus.OFFLINE
        
        if self._status != previous_status:
            for callback in self._callbacks:
                try:
                    callback(self._status)
                except Exception:
                    pass
        
        return self._status
