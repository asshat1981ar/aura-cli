"""Offline command executor."""

import asyncio
from typing import Callable, Optional

from .models import CommandResult, CommandStatus
from .queue import CommandQueue


class OfflineExecutor:
    """Execute commands and queue them when offline."""
    
    def __init__(
        self,
        queue: Optional[CommandQueue] = None,
        connectivity_check: Optional[Callable[[], bool]] = None,
    ):
        self.queue = queue or CommandQueue()
        self._connectivity_check = connectivity_check
        self._handlers: dict[str, Callable] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_handler(self, command: str, handler: Callable):
        """Register a handler for a command type."""
        self._handlers[command] = handler
    
    def _is_online(self) -> bool:
        if self._connectivity_check:
            return self._connectivity_check()
        return True
    
    async def execute(
        self,
        command: str,
        *args,
        **kwargs,
    ) -> CommandResult:
        """Execute a command or queue it if offline."""
        if self._is_online():
            return await self._do_execute(command, args, kwargs)
        
        return await self._queue_command(command, args, kwargs)
    
    async def _do_execute(
        self,
        command: str,
        args: tuple,
        kwargs: dict,
    ) -> CommandResult:
        """Execute command synchronously."""
        handler = self._handlers.get(command)
        
        if not handler:
            return CommandResult(
                command=command,
                success=False,
                output=None,
                error=f"No handler registered for command: {command}",
            )
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                result = handler(*args, **kwargs)
            
            return CommandResult(
                command=command,
                success=True,
                output=result,
                error=None,
            )
        except Exception as e:
            return CommandResult(
                command=command,
                success=False,
                output=None,
                error=str(e),
            )
    
    async def _queue_command(
        self,
        command: str,
        args: tuple,
        kwargs: dict,
    ) -> CommandResult:
        """Queue command for later execution."""
        from .models import CommandPriority, QueuedCommand
        
        queued = QueuedCommand(
            command=command,
            args=args,
            kwargs=kwargs,
            priority=CommandPriority.NORMAL,
        )
        
        await self.queue.add(queued)
        
        return CommandResult(
            command=command,
            success=True,
            output={"queued": True, "command_id": queued.id},
            error=None,
            queued_id=queued.id,
        )
    
    async def start_sync_loop(self):
        """Start background sync loop."""
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
    
    async def stop_sync_loop(self):
        """Stop background sync loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def _sync_loop(self):
        """Background loop to sync queued commands."""
        while self._running:
            try:
                await asyncio.sleep(30)
                await self.sync()
            except asyncio.CancelledError:
                break
    
    async def sync(self) -> int:
        """Process all pending queued commands."""
        if not self._is_online():
            return 0
        
        commands = await self.queue.get_pending(limit=10)
        processed = 0
        
        for cmd in commands:
            await self.queue.update_status(cmd.id, CommandStatus.PROCESSING)
            
            result = await self._do_execute(cmd.command, cmd.args, cmd.kwargs)
            
            if result.success:
                await self.queue.remove(cmd.id)
                processed += 1
            else:
                await self.queue.update_status(cmd.id, CommandStatus.FAILED)
        
        return processed
