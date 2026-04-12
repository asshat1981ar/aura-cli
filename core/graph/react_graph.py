"""LangGraph ReAct engine implementation for AURA.

Defines a goal-oriented reasoning loop using StateGraph, supporting 
persistent threads and tool integration.
"""

from __future__ import annotations

import logging
from typing import Annotated, Dict, List, Optional, Mapping, TypedDict, Union, Any

logger = logging.getLogger(__name__)

class ReActGraphEngine:
    """Core LangGraph engine for ReAct-style reasoning."""

    def __init__(
        self, 
        llm_caller: callable, 
        tool_registry: callable, 
        checkpointer: Optional[Any] = None
    ):
        self.llm_caller = llm_caller
        self.tool_registry = tool_registry
        self.checkpointer = checkpointer
        # Lazy build graph to avoid top-level import errors
        self._app = None

    @property
    def app(self):
        if self._app is None:
            self._app = self._build_graph()
        return self._app

    def _build_graph(self):
        """Construct the StateGraph with agent and tool nodes."""
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            logger.error("langgraph not installed. ReActGraphEngine unavailable.")
            raise

        class AgentState(TypedDict):
            messages: Annotated[List[Any], "The sequence of messages"]
            goal: str
            memory: Dict[str, Any]
            next_action: Optional[str]

        workflow = StateGraph(AgentState)

        # 1. Define Nodes
        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", self._call_tools)

        # 2. Define Edges
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )

        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)

    async def _call_agent(self, state: Dict) -> Dict:
        """Invoke the LLM caller bridge."""
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            raise ImportError("langchain-core required for ReActGraphEngine")

        response_text = await self.llm_caller(state)
        
        next_action = None
        if "Action:" in response_text:
            next_action = response_text.split("Action:")[1].strip()
        
        return {
            "messages": [AIMessage(content=response_text)],
            "next_action": next_action
        }

    async def _call_tools(self, state: Dict) -> Dict:
        """Invoke the tool registry bridge."""
        try:
            from langchain_core.messages import ToolMessage
        except ImportError:
            raise ImportError("langchain-core required for ReActGraphEngine")

        tool_output = await self.tool_registry(state)
        return {
            "messages": [ToolMessage(content=tool_output, tool_call_id="only-one")]
        }

    def _should_continue(self, state: Dict) -> str:
        """Routing logic based on next_action."""
        if state.get("next_action"):
            return "continue"
        return "end"

    async def run(
        self, 
        goal: str, 
        initial_memory: Optional[Mapping] = None, 
        thread_id: str = "default"
    ):
        """Execute the graph for a given goal."""
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            raise ImportError("langchain-core required for ReActGraphEngine")

        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=goal)],
            "goal": goal,
            "memory": dict(initial_memory) if initial_memory else {},
            "next_action": None
        }

        final_state = await self.app.ainvoke(initial_state, config=config)
        
        from dataclasses import dataclass
        @dataclass
        class GraphResult:
            checkpoint: Dict
            memory: Dict
            
        return GraphResult(
            checkpoint={"thread_id": thread_id},
            memory=final_state.get("memory", {})
        )
