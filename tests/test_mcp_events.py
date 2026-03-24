"""Tests for MCP bi-directional event system."""
import asyncio
import json
import unittest

from core.mcp_events import (
    EventBus, EventType, MCPEvent, CallbackRegistry,
)


class TestMCPEvent(unittest.TestCase):
    def test_creation(self):
        e = MCPEvent(event_type=EventType.TOOL_COMPLETE, source="test")
        self.assertEqual(e.event_type, EventType.TOOL_COMPLETE)
        self.assertEqual(e.source, "test")

    def test_to_dict(self):
        e = MCPEvent(event_type=EventType.CI_COMPLETE, data={"status": "ok"})
        d = e.to_dict()
        self.assertEqual(d["event_type"], "ci.complete")
        self.assertEqual(d["data"]["status"], "ok")

    def test_to_sse(self):
        e = MCPEvent(id="evt1", event_type=EventType.CUSTOM, source="s")
        sse = e.to_sse()
        self.assertIn("id: evt1", sse)
        self.assertIn("event: custom", sse)
        self.assertIn("data:", sse)

    def test_from_dict(self):
        e = MCPEvent.from_dict({
            "event_type": "tool.complete",
            "source": "linter",
            "data": {"result": "pass"},
        })
        self.assertEqual(e.event_type, EventType.TOOL_COMPLETE)
        self.assertEqual(e.source, "linter")

    def test_from_dict_unknown_type(self):
        e = MCPEvent.from_dict({"event_type": "unknown.type"})
        self.assertEqual(e.event_type, EventType.CUSTOM)


class TestEventBus(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.TOOL_COMPLETE, lambda e: received.append(e))

        event = MCPEvent(event_type=EventType.TOOL_COMPLETE, data={"ok": True})
        self._run(bus.publish(event))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data["ok"], True)

    def test_subscribe_all(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: received.append(e))

        self._run(bus.publish(MCPEvent(event_type=EventType.TOOL_COMPLETE)))
        self._run(bus.publish(MCPEvent(event_type=EventType.CI_COMPLETE)))
        self.assertEqual(len(received), 2)

    def test_publish_does_not_match_wrong_type(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.CI_COMPLETE, lambda e: received.append(e))

        self._run(bus.publish(MCPEvent(event_type=EventType.TOOL_COMPLETE)))
        self.assertEqual(len(received), 0)

    def test_publish_sync(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.CUSTOM, lambda e: received.append(e))

        event = MCPEvent(event_type=EventType.CUSTOM)
        bus.publish_sync(event)
        self.assertEqual(len(received), 1)

    def test_history(self):
        bus = EventBus()
        self._run(bus.publish(MCPEvent(event_type=EventType.TOOL_COMPLETE)))
        self._run(bus.publish(MCPEvent(event_type=EventType.CI_COMPLETE)))

        history = bus.get_history()
        self.assertEqual(len(history), 2)

    def test_history_filter_by_type(self):
        bus = EventBus()
        self._run(bus.publish(MCPEvent(event_type=EventType.TOOL_COMPLETE)))
        self._run(bus.publish(MCPEvent(event_type=EventType.CI_COMPLETE)))

        history = bus.get_history(event_type="tool.complete")
        self.assertEqual(len(history), 1)

    def test_history_limit(self):
        bus = EventBus(max_history=5)
        for i in range(10):
            self._run(bus.publish(MCPEvent(event_type=EventType.CUSTOM)))
        self.assertEqual(len(bus.get_history(limit=100)), 5)

    def test_sse_stream(self):
        bus = EventBus()
        sid, queue = bus.create_sse_stream("test-stream")
        self.assertEqual(sid, "test-stream")

        self._run(bus.publish(MCPEvent(event_type=EventType.CUSTOM)))
        self.assertFalse(queue.empty())

        bus.close_sse_stream("test-stream")
        self._run(bus.publish(MCPEvent(event_type=EventType.CUSTOM)))
        # Queue closed, should not receive

    def test_subscriber_error_does_not_break_bus(self):
        bus = EventBus()

        def bad_callback(e):
            raise RuntimeError("boom")

        bus.subscribe(EventType.CUSTOM, bad_callback)

        received = []
        bus.subscribe(EventType.CUSTOM, lambda e: received.append(e))

        self._run(bus.publish(MCPEvent(event_type=EventType.CUSTOM)))
        self.assertEqual(len(received), 1)


class TestCallbackRegistry(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_cleanup_expired(self):
        bus = EventBus()
        registry = CallbackRegistry(bus)
        registry.register_callback("c1", lambda e: None, timeout_seconds=-1)
        registry.cleanup_expired()
        self.assertEqual(len(registry._pending), 0)


class TestConsolidation(unittest.TestCase):
    """Quick smoke test for memory consolidation module."""

    def test_import(self):
        from memory.consolidation import (
            MemoryEntry, MemoryConsolidator, ConsolidationResult,
            NegativeExampleStore,
        )
        self.assertIsNotNone(MemoryEntry)
        self.assertIsNotNone(MemoryConsolidator)


if __name__ == "__main__":
    unittest.main()
