"""Inter-agent communication via message bus.

Provides pub/sub and request-response patterns. Falls back to an in-process
queue when Momento Topics is unavailable, following the circuit breaker
pattern from memory/momento_adapter.py.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# Well-known topic names
TOPIC_AGENT_REQUEST = "aura.agent.request"
TOPIC_AGENT_RESPONSE = "aura.agent.response"
TOPIC_ENVIRONMENT_EVENT = "aura.environment.event"
TOPIC_SERVER_EVENT = "aura.server.event"
TOPIC_HEALTH = "aura.health"


@dataclass
class Message:
    """A message on the bus."""

    topic: str
    payload: Dict[str, Any]
    sender: str = ""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "sender": self.sender,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }


class InProcessBus:
    """In-process message bus for local development and fallback."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_log: List[Message] = []
        self._max_log_size = 1000
        self._pending_responses: Dict[str, asyncio.Future] = {}

    def publish(self, topic: str, payload: Dict, sender: str = "") -> Message:
        """Publish a message to a topic."""
        msg = Message(topic=topic, payload=payload, sender=sender)
        self._message_log.append(msg)

        # Trim log
        if len(self._message_log) > self._max_log_size:
            self._message_log = self._message_log[-self._max_log_size:]

        # Notify subscribers
        for handler in self._subscribers.get(topic, []):
            try:
                handler(msg)
            except Exception:
                pass

        # Check for pending request-response
        if msg.correlation_id and msg.correlation_id in self._pending_responses:
            future = self._pending_responses.pop(msg.correlation_id)
            if not future.done():
                future.set_result(msg)

        return msg

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Subscribe to a topic with a handler function."""
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable) -> None:
        """Remove a subscription."""
        subs = self._subscribers.get(topic, [])
        if handler in subs:
            subs.remove(handler)

    def recent_messages(self, topic: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent messages, optionally filtered by topic."""
        msgs = self._message_log
        if topic:
            msgs = [m for m in msgs if m.topic == topic]
        return [m.as_dict() for m in msgs[-limit:]]


class MessageBus:
    """Message bus with Momento Topics backend and in-process fallback.

    Uses the circuit breaker pattern from memory/momento_adapter.py to
    gracefully degrade when the external backend is unavailable.
    """

    def __init__(self, momento_adapter=None):
        self._momento = momento_adapter
        self._local = InProcessBus()
        self._use_momento = momento_adapter is not None

    def publish(self, topic: str, payload: Dict, sender: str = "") -> Message:
        """Publish a message. Tries Momento first, falls back to local."""
        msg = self._local.publish(topic, payload, sender)

        if self._use_momento and self._momento:
            try:
                self._momento.publish(topic, msg.as_dict())
            except Exception:
                pass  # Local publish already succeeded

        return msg

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Subscribe to a topic."""
        self._local.subscribe(topic, handler)

    def request_response(
        self, topic: str, payload: Dict, sender: str = "", timeout: float = 30.0
    ) -> Optional[Dict]:
        """Send a request and wait for a correlated response (sync version).

        Uses correlation_id to match request to response.
        """
        correlation_id = str(uuid.uuid4())
        payload["_correlation_id"] = correlation_id

        # Publish request
        self.publish(topic, payload, sender)

        # Wait for correlated response on the response topic
        response_topic = f"{topic}.response"
        start = time.time()
        result = {"_timeout": True}

        def _handler(msg: Message):
            nonlocal result
            if msg.correlation_id == correlation_id:
                result = msg.payload

        self._local.subscribe(response_topic, _handler)

        while time.time() - start < timeout:
            if "_timeout" not in result:
                break
            time.sleep(0.1)

        self._local.unsubscribe(response_topic, _handler)

        return result if "_timeout" not in result else None

    def recent_messages(self, topic: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent messages from the local bus."""
        return self._local.recent_messages(topic, limit)
