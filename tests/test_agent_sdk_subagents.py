# tests/test_agent_sdk_subagents.py
"""Tests for Agent SDK subagent definitions."""

import unittest


class TestSubagentDefinitions(unittest.TestCase):
    """Test subagent creation for parallel work dispatch."""

    def test_get_all_subagents(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        agents = get_subagent_definitions()
        self.assertIsInstance(agents, dict)
        self.assertGreater(len(agents), 0)

    def test_required_subagents_present(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        agents = get_subagent_definitions()
        required = ["planning-agent", "implementation-agent", "verification-agent", "research-agent"]
        for name in required:
            self.assertIn(name, agents, f"Missing subagent: {name}")

    def test_subagent_has_description(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        for name, defn in get_subagent_definitions().items():
            self.assertTrue(
                len(defn.description) > 20,
                f"Subagent {name} needs a better description",
            )

    def test_subagent_has_tools(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        for name, defn in get_subagent_definitions().items():
            self.assertIsInstance(defn.tools, list)
            self.assertGreater(len(defn.tools), 0, f"Subagent {name} has no tools")

    def test_subagent_has_prompt(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        for name, defn in get_subagent_definitions().items():
            self.assertTrue(
                len(defn.prompt) > 20,
                f"Subagent {name} needs a better prompt",
            )

    def test_get_subagent_for_task_type(self):
        from core.agent_sdk.subagent_definitions import get_subagent_for_task

        agent = get_subagent_for_task("plan")
        self.assertEqual(agent, "planning-agent")

        agent = get_subagent_for_task("implement")
        self.assertEqual(agent, "implementation-agent")

        agent = get_subagent_for_task("verify")
        self.assertEqual(agent, "verification-agent")

        agent = get_subagent_for_task("research")
        self.assertEqual(agent, "research-agent")


if __name__ == "__main__":
    unittest.main()
