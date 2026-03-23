"""Tests for orchestrator_hub/ — registry, lifecycle, router, message_bus, hub."""
import time
import unittest

from orchestrator_hub.registry import AgentInfo, AgentRegistryHub, ServerInfo
from orchestrator_hub.lifecycle import ServerLifecycle, _listening
from orchestrator_hub.router import TaskRouter
from orchestrator_hub.message_bus import InProcessBus, MessageBus, Message


class TestAgentRegistryHub(unittest.TestCase):
    """Tests for AgentRegistryHub."""

    def setUp(self):
        self.registry = AgentRegistryHub()

    def test_register_agent(self):
        info = self.registry.register_agent(
            name="test_agent",
            agent_type="python",
            capabilities=["code", "test"],
            endpoint="http://localhost:9000",
            environment="gemini",
        )
        self.assertEqual(info.name, "test_agent")
        self.assertEqual(info.agent_type, "python")

    def test_list_agents(self):
        self.registry.register_agent("a1", "python", ["code"], "http://a1", "gemini")
        self.registry.register_agent("a2", "typescript", ["code"], "http://a2", "claude")
        agents = self.registry.list_agents()
        self.assertEqual(len(agents), 2)

    def test_deregister_agent(self):
        self.registry.register_agent("a1", "python", ["code"], "http://a1", "gemini")
        self.assertTrue(self.registry.deregister_agent("a1"))
        self.assertFalse(self.registry.deregister_agent("a1"))
        self.assertEqual(len(self.registry.list_agents()), 0)

    def test_discover_by_capability(self):
        self.registry.register_agent("a1", "python", ["code", "test"], "http://a1", "gemini")
        self.registry.register_agent("a2", "ts", ["code"], "http://a2", "claude")
        self.registry.register_agent("a3", "monitor", ["monitor"], "http://a3", "codex")

        code_agents = self.registry.discover("code")
        self.assertEqual(len(code_agents), 2)

        monitor_agents = self.registry.discover("monitor")
        self.assertEqual(len(monitor_agents), 1)

    def test_discover_excludes_unhealthy(self):
        self.registry.register_agent("a1", "python", ["code"], "http://a1", "gemini")
        self.registry.mark_unhealthy("a1")
        self.assertEqual(len(self.registry.discover("code")), 0)

    def test_discover_by_type(self):
        self.registry.register_agent("a1", "python", ["code"], "http://a1", "gemini")
        self.registry.register_agent("a2", "python", ["test"], "http://a2", "claude")
        result = self.registry.discover_by_type("python")
        self.assertEqual(len(result), 2)

    def test_discover_by_environment(self):
        self.registry.register_agent("a1", "python", ["code"], "http://a1", "gemini")
        self.registry.register_agent("a2", "ts", ["code"], "http://a2", "claude")
        result = self.registry.discover_by_environment("gemini")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "a1")

    def test_heartbeat(self):
        self.registry.register_agent("a1", "python", ["code"], "http://a1", "gemini")
        self.assertTrue(self.registry.heartbeat("a1"))
        self.assertFalse(self.registry.heartbeat("nonexistent"))

    def test_register_server(self):
        info = self.registry.register_server("dev_tools", 8001)
        self.assertEqual(info.name, "dev_tools")
        self.assertEqual(info.port, 8001)

    def test_list_servers(self):
        self.registry.register_server("dev_tools", 8001)
        self.registry.register_server("skills", 8002)
        servers = self.registry.list_servers()
        self.assertEqual(len(servers), 2)

    def test_update_server_status(self):
        self.registry.register_server("dev_tools", 8001)
        self.registry.update_server_status("dev_tools", "healthy")
        server = self.registry.get_server("dev_tools")
        self.assertEqual(server.status, "healthy")


class TestTaskRouter(unittest.TestCase):
    """Tests for TaskRouter."""

    def setUp(self):
        self.registry = AgentRegistryHub()
        self.router = TaskRouter(self.registry)

    def test_route_returns_none_for_empty_task(self):
        self.assertIsNone(self.router.route({}))
        self.assertIsNone(self.router.route({"goal": ""}))

    def test_route_matches_python_keywords(self):
        self.registry.register_agent("py", "python", ["python"], "http://py", "gemini")
        result = self.router.route({"goal": "Write a python script to parse CSV"})
        self.assertEqual(result, "py")

    def test_route_matches_typescript_keywords(self):
        self.registry.register_agent("ts", "ts", ["typescript"], "http://ts", "claude")
        result = self.router.route({"goal": "Build a typescript React component"})
        self.assertEqual(result, "ts")

    def test_route_returns_none_when_no_agents(self):
        result = self.router.route({"goal": "Write python code"})
        self.assertIsNone(result)

    def test_route_with_fallback_excludes_agents(self):
        self.registry.register_agent("py1", "python", ["python"], "http://py1", "gemini")
        self.registry.register_agent("py2", "python", ["python"], "http://py2", "claude")
        result = self.router.route_with_fallback(
            {"goal": "python script"}, exclude=["py1"]
        )
        self.assertEqual(result, "py2")

    def test_load_balance_round_robin(self):
        self.registry.register_agent("py1", "python", ["code"], "http://py1", "gemini")
        self.registry.register_agent("py2", "python", ["code"], "http://py2", "claude")
        first = self.router.load_balance("python")
        second = self.router.load_balance("python")
        self.assertNotEqual(first, second)


class TestInProcessBus(unittest.TestCase):
    """Tests for InProcessBus."""

    def setUp(self):
        self.bus = InProcessBus()

    def test_publish_creates_message(self):
        msg = self.bus.publish("test.topic", {"key": "value"}, sender="test")
        self.assertEqual(msg.topic, "test.topic")
        self.assertEqual(msg.payload["key"], "value")
        self.assertEqual(msg.sender, "test")

    def test_subscribe_receives_messages(self):
        received = []
        self.bus.subscribe("test.topic", lambda m: received.append(m))
        self.bus.publish("test.topic", {"data": 1})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["data"], 1)

    def test_subscribe_topic_filtering(self):
        received = []
        self.bus.subscribe("topic.a", lambda m: received.append(m))
        self.bus.publish("topic.a", {"a": 1})
        self.bus.publish("topic.b", {"b": 2})
        self.assertEqual(len(received), 1)

    def test_recent_messages(self):
        self.bus.publish("t1", {"a": 1})
        self.bus.publish("t2", {"b": 2})
        self.bus.publish("t1", {"c": 3})

        all_msgs = self.bus.recent_messages()
        self.assertEqual(len(all_msgs), 3)

        t1_msgs = self.bus.recent_messages(topic="t1")
        self.assertEqual(len(t1_msgs), 2)

    def test_unsubscribe(self):
        received = []
        handler = lambda m: received.append(m)
        self.bus.subscribe("t", handler)
        self.bus.publish("t", {"a": 1})
        self.bus.unsubscribe("t", handler)
        self.bus.publish("t", {"b": 2})
        self.assertEqual(len(received), 1)


class TestMessageBus(unittest.TestCase):
    """Tests for MessageBus (wrapper with fallback)."""

    def test_local_fallback(self):
        bus = MessageBus(momento_adapter=None)
        msg = bus.publish("test", {"key": "val"})
        self.assertIsNotNone(msg.message_id)

    def test_recent_messages(self):
        bus = MessageBus()
        bus.publish("t1", {"a": 1})
        bus.publish("t2", {"b": 2})
        msgs = bus.recent_messages()
        self.assertEqual(len(msgs), 2)


class TestServerLifecycle(unittest.TestCase):
    """Tests for ServerLifecycle."""

    def test_register_server(self):
        lifecycle = ServerLifecycle(project_root="/tmp")
        state = lifecycle.register("test", 9999, ["echo", "hello"])
        self.assertEqual(state.name, "test")
        self.assertEqual(state.port, 9999)

    def test_health_check_nonexistent(self):
        lifecycle = ServerLifecycle(project_root="/tmp")
        result = lifecycle.health_check("nonexistent")
        self.assertEqual(result["status"], "not_registered")

    def test_health_check_not_running(self):
        lifecycle = ServerLifecycle(project_root="/tmp")
        lifecycle.register("test", 59999, ["echo", "hello"])
        result = lifecycle.health_check("test")
        self.assertEqual(result["status"], "unhealthy")

    def test_list_servers(self):
        lifecycle = ServerLifecycle(project_root="/tmp")
        lifecycle.register("a", 9001, ["echo"])
        lifecycle.register("b", 9002, ["echo"])
        servers = lifecycle.list_servers()
        self.assertEqual(len(servers), 2)

    def test_stop_no_pid(self):
        lifecycle = ServerLifecycle(project_root="/tmp")
        lifecycle.register("test", 9999, ["echo"])
        result = lifecycle.stop("test")
        self.assertEqual(result["status"], "no_pid")


if __name__ == "__main__":
    unittest.main()
