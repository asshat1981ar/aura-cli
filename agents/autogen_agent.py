"""AutoGen group-chat agent adapter for AURA's agent registry."""

from __future__ import annotations

from typing import Any, Dict
from core.logging_utils import log_json


class AutoGenGroupChatAgent:
    """Wraps AutoGen GroupChat as an AURA agent with run() interface."""

    name = "autogen_group_chat"

    def __init__(self, brain=None, model=None, config: dict | None = None):
        self.brain = brain
        self.model = model
        self.config = config or {}
        self._autogen_available = self._check_autogen()

    @staticmethod
    def _check_autogen() -> bool:
        try:
            import autogen  # noqa: F401

            return True
        except ImportError:
            return False

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        goal = input_data.get("goal", "")
        if not self._autogen_available:
            log_json("WARN", "autogen_not_available", details={"message": "pyautogen not installed, using fallback"})
            return self._fallback_brainstorm(goal)

        result = self._conduct_group_chat(goal)
        if self.brain:
            self.brain.remember(f"AutoGen group chat for: {goal[:100]}. Decisions: {result.get('decisions', [])}")
        return {
            "conversation": result.get("conversation", ""),
            "decisions": result.get("decisions", []),
            "participants": self.config.get("agents", []),
        }

    def _conduct_group_chat(self, goal: str) -> Dict[str, Any]:
        from autogen import AssistantAgent, GroupChat, GroupChatManager

        agents_config = self.config.get(
            "agents",
            [
                {"name": "ideator", "system_message": "Generate creative solutions"},
                {"name": "critic", "system_message": "Find risks and flaws"},
                {"name": "synthesizer", "system_message": "Merge ideas into actionable plan"},
            ],
        )
        llm_config = self.config.get("llm_config", {})
        max_rounds = self.config.get("max_turns", 6)

        agents = [AssistantAgent(name=a["name"], system_message=a["system_message"], llm_config=llm_config) for a in agents_config]
        groupchat = GroupChat(agents=agents, messages=[], max_round=max_rounds)
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        import asyncio

        asyncio.run(manager.a_initiate_chat(agents[0], message=goal))
        messages = [msg.get("content", "") for msg in groupchat.messages]
        return {
            "conversation": "\n".join(messages),
            "decisions": [m for m in messages if "decision:" in m.lower()],
        }

    def _fallback_brainstorm(self, goal: str) -> Dict[str, Any]:
        return {
            "conversation": f"[fallback] Single-agent brainstorm for: {goal}",
            "decisions": [],
            "participants": [],
        }
