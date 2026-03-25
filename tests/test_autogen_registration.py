import unittest
from unittest.mock import MagicMock

class TestAutoGenRegistration(unittest.TestCase):
    def test_autogen_agent_in_default_agents(self):
        from agents.registry import default_agents
        agents = default_agents(
            brain=MagicMock(), model=MagicMock(), config={"autogen": {"enabled": True}}
        )
        self.assertIn("autogen_group_chat", agents)

    def test_default_agents_works_without_config(self):
        from agents.registry import default_agents
        agents = default_agents(brain=MagicMock(), model=MagicMock())
        self.assertIn("autogen_group_chat", agents)

    def test_router_keywords_include_brainstorming(self):
        from orchestrator_hub.router import _TASK_KEYWORDS
        self.assertIn("brainstorming", _TASK_KEYWORDS)
        self.assertIn("brainstorm", _TASK_KEYWORDS["brainstorming"])
