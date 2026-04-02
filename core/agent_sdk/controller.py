# core/agent_sdk/controller.py
"""AURA Meta-Controller — Agent SDK-powered closed-loop orchestrator.

Replaces rigid phase-sequenced orchestration with Claude-as-brain.
Claude dynamically selects tools, skills, agents, and workflows
based on goal context.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular deps and startup cost
_sdk_available: Optional[bool] = None


def _check_sdk() -> bool:
    """Check if claude-agent-sdk is installed."""
    global _sdk_available
    if _sdk_available is None:
        try:
            import claude_agent_sdk  # noqa: F401
            _sdk_available = True
        except ImportError:
            _sdk_available = False
    return _sdk_available


class AuraController:
    """Main entry point for Agent SDK-powered goal execution.

    Usage:
        controller = AuraController(config=config, project_root=root)
        result = await controller.run("Fix the authentication bug")
    """

    def __init__(
        self,
        config: Any,
        project_root: Path,
        brain: Any = None,
        model_adapter: Any = None,
        goal_queue: Any = None,
        goal_archive: Any = None,
    ) -> None:
        from core.agent_sdk.config import AgentSDKConfig

        self.config: AgentSDKConfig = config
        self.project_root = project_root
        self._brain = brain
        self._model_adapter = model_adapter
        self._goal_queue = goal_queue
        self._goal_archive = goal_archive

    def _build_prompt(self, goal: str) -> str:
        """Build the full prompt for the Agent SDK session."""
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(
            project_root=self.project_root,
            brain=self._brain,
        )
        context = builder.build(goal=goal)
        return builder.build_system_prompt(
            goal=goal,
            goal_type=context["goal_type"],
            context=context,
        )

    def _build_subagent_defs(self) -> Dict[str, Any]:
        """Build subagent definitions if enabled."""
        if not self.config.enable_subagents:
            return {}

        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        if not _check_sdk():
            # Return raw defs — caller will need to wrap them
            return {
                name: defn
                for name, defn in get_subagent_definitions().items()
            }

        from claude_agent_sdk import AgentDefinition

        return {
            name: AgentDefinition(
                description=defn.description,
                prompt=defn.prompt,
                tools=defn.tools,
            )
            for name, defn in get_subagent_definitions().items()
        }

    def _build_mcp_server(self) -> Any:
        """Create an MCP server with all AURA tools registered."""
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(
            project_root=self.project_root,
            brain=self._brain,
            model_adapter=self._model_adapter,
            goal_queue=self._goal_queue,
            goal_archive=self._goal_archive,
            config=self.config,
        )

        if not _check_sdk():
            # Return the raw tools list for testing without SDK
            return tools

        from claude_agent_sdk import tool, create_sdk_mcp_server

        sdk_tools = []
        for aura_tool in tools:
            @tool(aura_tool.name, aura_tool.description, aura_tool.input_schema.get("properties", {}))
            async def _handler(args, _t=aura_tool):
                result = _t.handler(args)
                import json
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, default=str),
                    }]
                }
            sdk_tools.append(_handler)

        return create_sdk_mcp_server("aura-tools", tools=sdk_tools)

    def _build_hooks(self) -> Dict[str, Any]:
        """Build hooks configuration if enabled."""
        if not self.config.enable_hooks:
            return {}

        from core.agent_sdk.hooks import create_hooks, reset_session_metrics

        reset_session_metrics()
        return create_hooks()

    def _build_options(self, goal: str) -> Any:
        """Build ClaudeAgentOptions for a session."""
        system_prompt = self._build_prompt(goal)
        mcp_server = self._build_mcp_server()
        subagents = self._build_subagent_defs()
        hooks = self._build_hooks()

        if not _check_sdk():
            # Return a mock-like namespace for testing
            class _MockOptions:
                pass
            opts = _MockOptions()
            opts.model = self.config.model
            opts.max_turns = self.config.max_turns
            opts.system_prompt = system_prompt
            opts.agents = subagents
            return opts

        from claude_agent_sdk import ClaudeAgentOptions

        mcp_servers = {"aura": mcp_server}
        allowed = list(self.config.allowed_tools)

        return ClaudeAgentOptions(
            cwd=str(self.project_root),
            model=self.config.model,
            max_turns=self.config.max_turns,
            max_budget_usd=self.config.max_budget_usd,
            permission_mode=self.config.permission_mode,
            allowed_tools=allowed,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            agents=subagents,
            hooks=hooks,
            thinking=self.config.thinking_config,
        )

    async def run(self, goal: str) -> Dict[str, Any]:
        """Execute a goal using the Agent SDK meta-controller.

        Returns a dict with:
            - result: The final text output
            - session_id: For resumption
            - metrics: Tool call metrics from the session
        """
        if not _check_sdk():
            raise RuntimeError(
                "claude-agent-sdk not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        from claude_agent_sdk import query, ResultMessage, SystemMessage
        from core.agent_sdk.hooks import get_session_metrics

        options = self._build_options(goal)
        session_id = None
        result_text = ""

        prompt = (
            f"Execute this development goal:\n\n{goal}\n\n"
            "Start by calling analyze_goal to understand the context, "
            "then proceed with the appropriate workflow."
        )

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
            elif isinstance(message, ResultMessage):
                result_text = message.result

        metrics = get_session_metrics().get_summary()

        return {
            "result": result_text,
            "session_id": session_id,
            "metrics": metrics,
        }

    async def run_with_client(self, goal: str) -> Dict[str, Any]:
        """Execute using ClaudeSDKClient for full lifecycle control.

        Use this when you need to interrupt, resume, or manage MCP servers
        at runtime.
        """
        if not _check_sdk():
            raise RuntimeError("claude-agent-sdk not installed.")

        from claude_agent_sdk import (
            ClaudeSDKClient, AssistantMessage, TextBlock, ResultMessage,
        )
        from core.agent_sdk.hooks import get_session_metrics

        options = self._build_options(goal)
        result_text = ""

        prompt = (
            f"Execute this development goal:\n\n{goal}\n\n"
            "Start by calling analyze_goal to understand the context, "
            "then proceed with the appropriate workflow."
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result_text += block.text
                elif isinstance(message, ResultMessage):
                    result_text = message.result

        metrics = get_session_metrics().get_summary()
        return {"result": result_text, "metrics": metrics}
