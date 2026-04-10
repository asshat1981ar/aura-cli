import unittest
from unittest.mock import MagicMock, patch


class TestAutoGenGroupChatAgent(unittest.TestCase):
    def test_agent_has_required_attributes(self):
        from agents.autogen_agent import AutoGenGroupChatAgent

        agent = AutoGenGroupChatAgent(brain=MagicMock(), model=MagicMock())
        self.assertEqual(agent.name, "autogen_group_chat")
        self.assertTrue(callable(getattr(agent, "run", None)))

    def test_run_returns_expected_schema(self):
        from agents.autogen_agent import AutoGenGroupChatAgent

        agent = AutoGenGroupChatAgent(brain=MagicMock(), model=MagicMock())
        with patch.object(
            agent,
            "_conduct_group_chat",
            return_value={
                "conversation": "Agent1: idea\nAgent2: refinement",
                "decisions": ["use adapter pattern"],
            },
        ):
            result = agent.run({"goal": "brainstorm auth approach"})
        self.assertIn("conversation", result)
        self.assertIn("decisions", result)
        self.assertIn("participants", result)

    def test_run_stores_to_brain(self):
        from agents.autogen_agent import AutoGenGroupChatAgent

        brain = MagicMock()
        agent = AutoGenGroupChatAgent(brain=brain, model=MagicMock())
        agent._autogen_available = True
        with patch.object(
            agent,
            "_conduct_group_chat",
            return_value={
                "conversation": "test",
                "decisions": [],
            },
        ):
            agent.run({"goal": "test goal"})
        brain.remember.assert_called_once()

    def test_fallback_when_autogen_unavailable(self):
        from agents.autogen_agent import AutoGenGroupChatAgent

        agent = AutoGenGroupChatAgent(brain=MagicMock(), model=MagicMock())
        # Force fallback path
        agent._autogen_available = False
        result = agent.run({"goal": "test"})
        self.assertIn("conversation", result)
        self.assertIn("[fallback]", result["conversation"])
        self.assertEqual(result["decisions"], [])
