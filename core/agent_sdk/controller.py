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
        model_router: Any = None,
        workflow_executor: Any = None,
        session_store: Any = None,
        feedback: Any = None,
    ) -> None:
        from core.agent_sdk.config import AgentSDKConfig
        from core.agent_sdk.context_builder import ContextBuilder

        self.config: AgentSDKConfig = config
        self.project_root = project_root
        self._brain = brain
        self._model_adapter = model_adapter
        self._goal_queue = goal_queue
        self._goal_archive = goal_archive
        self.model_router = model_router
        self.workflow_executor = workflow_executor
        self.session_store = session_store
        self.feedback = feedback
        self.context_builder = ContextBuilder(
            project_root=project_root, brain=brain,
        )

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

    def _build_options(self, goal: str, model: str = None, resume: str = None) -> Any:
        """Build ClaudeAgentOptions for a session."""
        system_prompt = self._build_prompt(goal)
        mcp_server = self._build_mcp_server()
        subagents = self._build_subagent_defs()
        hooks = self._build_hooks()
        effective_model = model or self.config.model

        if not _check_sdk():
            # Return a mock-like namespace for testing
            class _MockOptions:
                pass
            opts = _MockOptions()
            opts.model = effective_model
            opts.max_turns = self.config.max_turns
            opts.system_prompt = system_prompt
            opts.agents = subagents
            return opts

        from claude_agent_sdk import ClaudeAgentOptions

        options_kwargs = dict(
            cwd=str(self.project_root),
            model=effective_model,
            max_turns=self.config.max_turns,
            max_budget_usd=self.config.max_budget_usd,
            permission_mode=self.config.permission_mode,
            allowed_tools=list(self.config.allowed_tools),
            system_prompt=system_prompt,
            mcp_servers={"aura": mcp_server},
            agents=subagents,
            hooks=hooks,
            thinking=self.config.thinking_config,
        )
        if resume:
            options_kwargs["resume"] = resume
        return ClaudeAgentOptions(**options_kwargs)

    async def run(self, goal: str, resume_session_id: str = None) -> Dict[str, Any]:
        """Execute a goal with adaptive routing, workflow selection, and feedback."""
        if not _check_sdk():
            raise RuntimeError(
                "claude-agent-sdk not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        from claude_agent_sdk import query, ResultMessage, SystemMessage, AssistantMessage
        from core.agent_sdk.hooks import get_session_metrics

        # 1. Build context
        context = self.context_builder.build(goal=goal)

        # 1b. Enrich with feedback data
        if self.feedback:
            context["failure_patterns"] = self.feedback.get_failure_patterns(context["goal_type"])
            context["skill_weights"] = self.feedback.skill_updater.get_weights()

        # 2. Select model via adaptive router
        model = self.model_router.select_model(context["goal_type"]) if self.model_router else self.config.model

        # 3. Select workflow template
        workflow = None
        if self.workflow_executor:
            workflow = self.workflow_executor.select_workflow(context["goal_type"])

        # 4. Create or resume session
        session_pk = None
        if self.session_store:
            if resume_session_id:
                session = self.session_store.get_session(resume_session_id)
                session_pk = session["id"] if session else None
            else:
                session_pk = self.session_store.create_session(
                    session_id="pending",
                    goal=goal, goal_type=context["goal_type"],
                    workflow=workflow.name if workflow else "freeform",
                    model_tier=model,
                )

        # 5. Build options
        options = self._build_options(goal, model=model, resume=resume_session_id)

        # 6. Execute via Agent SDK
        session_id = None
        result_text = ""
        total_cost = 0.0

        prompt = (
            f"Execute this development goal:\n\n{goal}\n\n"
            "Start by calling analyze_goal to understand the context, "
            "then proceed with the appropriate workflow."
        )
        if workflow:
            phase_names = " → ".join(p.tool_name for p in workflow.phases)
            prompt += f"\n\nRecommended workflow ({workflow.name}): {phase_names}"

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
            elif isinstance(message, AssistantMessage) and message.usage:
                tokens_in = message.usage.get("input_tokens", 0)
                tokens_out = message.usage.get("output_tokens", 0)
                if self.session_store and session_pk:
                    from core.agent_sdk.session_persistence import compute_cost
                    cost = compute_cost(model, tokens_in, tokens_out)
                    total_cost += cost
                    self.session_store.record_event(
                        session_pk, "sdk_turn", "agent_sdk", model,
                        tokens_in, tokens_out, True, None,
                    )
            elif isinstance(message, ResultMessage):
                result_text = message.result

        metrics = get_session_metrics().get_summary()

        # 7. Record outcome and trigger feedback
        success = bool(result_text and "error" not in result_text.lower()[:100])
        if self.feedback and session_pk:
            self.feedback.on_goal_complete(
                session_pk=session_pk, goal=goal,
                goal_type=context["goal_type"], model=model,
                skills_used=context.get("recommended_skills", []),
                success=success,
                verification_result={},
                cost=total_cost,
            )

        # 8. Update session status
        if self.session_store and session_pk:
            self.session_store.update_status(
                session_pk,
                "completed" if success else "failed",
                error_summary=None if success else "Goal execution failed",
            )

        return {
            "result": result_text,
            "session_id": session_id,
            "metrics": metrics,
            "total_cost_usd": total_cost,
            "success": success,
        }

    async def run_with_client(self, goal: str, resume_session_id: str = None) -> Dict[str, Any]:
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
