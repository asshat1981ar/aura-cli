"""Tests for A2A protocol implementation."""

import asyncio
import unittest
from unittest.mock import MagicMock

from core.a2a.agent_card import AgentCard, AgentCapability
from core.a2a.task import A2ATask, TaskState, A2AMessage
from core.a2a.server import A2AServer
from core.a2a.client import A2AClient


class TestAgentCapability(unittest.TestCase):
    def test_creation(self):
        cap = AgentCapability(name="code_gen", description="Generates code")
        self.assertEqual(cap.name, "code_gen")
        self.assertEqual(cap.input_schema, {})


class TestAgentCard(unittest.TestCase):
    def test_default(self):
        card = AgentCard.default()
        self.assertEqual(card.name, "AURA CLI")
        self.assertTrue(len(card.capabilities) > 0)
        self.assertIn("a2a/1.0", card.supported_protocols)

    def test_custom_host_port(self):
        card = AgentCard.default(host="0.0.0.0", port=9000)
        self.assertEqual(card.url, "http://0.0.0.0:9000")

    def test_to_dict_roundtrip(self):
        card = AgentCard.default()
        d = card.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["name"], "AURA CLI")
        self.assertIsInstance(d["capabilities"], list)

    def test_to_json(self):
        card = AgentCard.default()
        j = card.to_json()
        import json

        parsed = json.loads(j)
        self.assertEqual(parsed["name"], "AURA CLI")

    def test_from_dict(self):
        data = {
            "name": "TestAgent",
            "description": "Test",
            "version": "1.0",
            "url": "http://test:8000",
            "capabilities": [
                {"name": "test_cap", "description": "test"},
            ],
        }
        card = AgentCard.from_dict(data)
        self.assertEqual(card.name, "TestAgent")
        self.assertEqual(len(card.capabilities), 1)
        self.assertEqual(card.capabilities[0].name, "test_cap")


class TestA2ATask(unittest.TestCase):
    def test_creation(self):
        task = A2ATask(capability="code_gen")
        self.assertEqual(task.state, TaskState.SUBMITTED)
        self.assertEqual(task.capability, "code_gen")

    def test_valid_transition(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        self.assertEqual(task.state, TaskState.WORKING)

    def test_invalid_transition(self):
        task = A2ATask()
        with self.assertRaises(ValueError):
            task.transition(TaskState.COMPLETED)

    def test_complete_lifecycle(self):
        task = A2ATask(capability="test")
        task.transition(TaskState.WORKING)
        task.add_message("user", "do something")
        task.add_message("agent", "done")
        task.add_artifact("result", {"data": 1})
        task.transition(TaskState.COMPLETED)
        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertEqual(len(task.messages), 2)
        self.assertEqual(len(task.artifacts), 1)

    def test_cancel_from_submitted(self):
        task = A2ATask()
        task.transition(TaskState.CANCELED)
        self.assertEqual(task.state, TaskState.CANCELED)

    def test_cancel_from_working(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.CANCELED)
        self.assertEqual(task.state, TaskState.CANCELED)

    def test_to_dict(self):
        task = A2ATask(capability="test")
        task.add_message("user", "hello")
        d = task.to_dict()
        self.assertEqual(d["capability"], "test")
        self.assertEqual(d["state"], "submitted")
        self.assertEqual(len(d["messages"]), 1)

    def test_input_required_transition(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.INPUT_REQUIRED)
        self.assertEqual(task.state, TaskState.INPUT_REQUIRED)
        task.transition(TaskState.WORKING)
        self.assertEqual(task.state, TaskState.WORKING)


class TestA2AServer(unittest.TestCase):
    def test_get_agent_card(self):
        server = A2AServer()
        card = server.get_agent_card()
        self.assertEqual(card["name"], "AURA CLI")

    def test_create_task_no_handler(self):
        server = A2AServer()
        task = asyncio.run(server.create_task("unknown", "do something"))
        self.assertEqual(task.state, TaskState.FAILED)

    def test_create_task_with_handler(self):
        server = A2AServer()

        def handler(task):
            return {"summary": "completed successfully"}

        server.register_handler("code_gen", handler)
        task = asyncio.run(server.create_task("code_gen", "write hello world"))
        self.assertEqual(task.state, TaskState.COMPLETED)

    def test_get_task(self):
        server = A2AServer()
        asyncio.run(server.create_task("test", "msg"))
        task_id = list(server.tasks.keys())[0]
        found = server.get_task(task_id)
        self.assertIsNotNone(found)
        self.assertIsNone(server.get_task("nonexistent"))

    def test_cancel_task(self):
        server = A2AServer()
        # Create a task that will fail (no handler)
        task = asyncio.run(server.create_task("x", "msg"))
        # Failed tasks can't be canceled
        self.assertFalse(server.cancel_task(task.id))


class TestA2AClient(unittest.TestCase):
    def test_init(self):
        client = A2AClient()
        self.assertEqual(len(client.peers), 0)

    def test_find_capable_peer_empty(self):
        client = A2AClient()
        self.assertIsNone(client.find_capable_peer("code_gen"))

    def test_list_peers_empty(self):
        client = A2AClient()
        self.assertEqual(client.list_peers(), [])


if __name__ == "__main__":
    unittest.main()
