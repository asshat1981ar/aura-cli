"""MCP bi-directional event system with SSE support.

Enables external tools and MCP servers to signal events back to the
orchestrator asynchronously. Uses Server-Sent Events (SSE) for streaming
and a pub/sub EventBus for in-process coordination.
"""
import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from core.logging_utils import log_json


class EventType(str, Enum):
    """Standard MCP callback event types."""
    TOOL_COMPLETE = "tool.complete"
    TEST_READY = "test.ready"
    CI_COMPLETE = "ci.complete"
    FILE_CHANGED = "file.changed"
    AGENT_MESSAGE = "agent.message"
    HEALTH_CHECK = "health.check"
    PHASE_COMPLETE = "phase.complete"
    CUSTOM = "custom"


@dataclass
class MCPEvent:
    """An event from an MCP tool or external source."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""

    def to_sse(self) -> str:
        """Format as Server-Sent Event message."""
        return (
            f"id: {self.id}\n"
            f"event: {self.event_type.value}\n"
            f"data: {json.dumps(self.to_dict())}\n\n"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MCPEvent":
        event_type = data.get("event_type", "custom")
        try:
            et = EventType(event_type)
        except ValueError:
            et = EventType.CUSTOM
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            event_type=et,
            source=data.get("source", ""),
            data=data.get("data", {}),
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id", ""),
        )


class EventBus:
    """Pub/sub event bus for MCP callback events.

    Supports both synchronous and async callbacks, SSE stream creation
    for external consumers, and event history with filtering.
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[MCPEvent] = []
        self._max_history = max_history
        self._queues: dict[str, asyncio.Queue] = {}

    def subscribe(self, event_type: str | EventType, callback: Callable):
        """Subscribe to events of a specific type."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        self._subscribers[key].append(callback)

    def unsubscribe(self, event_type: str | EventType, callback: Callable):
        """Unsubscribe a callback from a specific event type."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        try:
            self._subscribers[key].remove(callback)
        except ValueError:
            pass

    def subscribe_all(self, callback: Callable):
        """Subscribe to all events."""
        self._subscribers["*"].append(callback)

    async def publish(self, event: MCPEvent):
        """Publish an event to all matching subscribers and SSE queues."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        key = event.event_type.value
        for callback in self._subscribers.get(key, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as exc:
                log_json("WARN", "event_subscriber_error",
                         details={"event": key, "error": str(exc)})

        for callback in self._subscribers.get("*", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception:
                pass

        for queue in self._queues.values():
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def publish_sync(self, event: MCPEvent):
        """Publish an event synchronously (for non-async contexts)."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        key = event.event_type.value
        for callback in self._subscribers.get(key, []) + self._subscribers.get("*", []):
            if not asyncio.iscoroutinefunction(callback):
                try:
                    callback(event)
                except Exception:
                    pass

    def create_sse_stream(self, stream_id: str | None = None) -> tuple[str, asyncio.Queue]:
        """Create a new SSE stream. Returns (stream_id, queue)."""
        sid = stream_id or str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._queues[sid] = queue
        return sid, queue

    def close_sse_stream(self, stream_id: str):
        """Close an SSE stream."""
        self._queues.pop(stream_id, None)

    def get_history(self, event_type: str | None = None,
                    since: float | None = None,
                    limit: int = 50) -> list[MCPEvent]:
        """Get event history with optional filters."""
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events[-limit:]


class CallbackRegistry:
    """Registry for MCP tool callbacks — maps correlation IDs to handlers."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._pending: dict[str, dict] = {}

    def register_callback(self, correlation_id: str,
                          callback: Callable,
                          timeout_seconds: float = 300):
        """Register a one-shot callback for when a tool signals completion.

        The internal subscriber is automatically removed after the callback fires
        to prevent the subscriber list from growing unboundedly.
        """
        self._pending[correlation_id] = {
            "callback": callback,
            "timeout": time.time() + timeout_seconds,
            "registered_at": time.time(),
        }

        async def _handler(event: MCPEvent):
            if event.correlation_id == correlation_id:
                # Self-remove after firing so the subscriber list stays bounded
                self.event_bus.unsubscribe(EventType.TOOL_COMPLETE, _handler)
                entry = self._pending.pop(correlation_id, None)
                if entry:
                    cb = entry["callback"]
                    if asyncio.iscoroutinefunction(cb):
                        await cb(event)
                    else:
                        cb(event)

        self.event_bus.subscribe(EventType.TOOL_COMPLETE, _handler)

    async def wait_for(self, correlation_id: str,
                       timeout: float = 60) -> MCPEvent | None:
        """Wait for a specific callback event."""
        sid = f"wait_{correlation_id}"
        _, queue = self.event_bus.create_sse_stream(sid)
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=min(1.0, deadline - time.time()),
                    )
                    if event.correlation_id == correlation_id:
                        return event
                except asyncio.TimeoutError:
                    continue
        finally:
            self.event_bus.close_sse_stream(sid)
        return None

    def cleanup_expired(self):
        """Remove expired pending callbacks."""
        now = time.time()
        expired = [cid for cid, entry in self._pending.items()
                   if entry["timeout"] < now]
        for cid in expired:
            self._pending.pop(cid, None)
