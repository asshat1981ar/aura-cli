#!/usr/bin/env python3
"""
Phase 1 Demo — ReAct Engine + Graph Workflow Engine working together.

Demonstrates:
  1. A ReAct agent solving a multi-step math task using tools.
  2. A graph workflow with conditional branching and a cycle.
  3. The ReAct loop used as a node inside a graph workflow.

Run from repo root:
    python examples/phase1_demo.py
"""
from __future__ import annotations

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from core.react_engine import ReActLoop, Tool, ToolSchema, HookRegistry
from core.graph_engine import StateGraph, END, react_node


# ============================================================================
# Demo 1: ReAct Agent solving a multi-step task
# ============================================================================

class DemoLLM:
    """Scripted LLM that simulates multi-step reasoning with tools."""

    def __init__(self):
        self._step = 0

    def think(self, goal, tools, history):
        self._step += 1

        # Check what observations we have so far
        observations = [h["content"] for h in history if h["role"] == "observe"]

        if self._step == 1:
            return {
                "thought": "I need to calculate 15 * 7 first.",
                "action": "multiply",
                "action_input": {"a": 15, "b": 7},
            }
        elif self._step == 2:
            return {
                "thought": f"15 * 7 = {observations[-1]}. Now I need to add 33.",
                "action": "add",
                "action_input": {"a": int(observations[-1]), "b": 33},
            }
        else:
            return {
                "thought": f"The final result is {observations[-1]}.",
                "final_answer": observations[-1],
            }


def demo_react_agent():
    """ReAct agent calculates (15 * 7) + 33 using tools."""
    print("=" * 60)
    print("DEMO 1: ReAct Agent — Multi-step Math Task")
    print("=" * 60)
    print("Goal: Calculate (15 * 7) + 33\n")

    # Define tools
    add_tool = Tool(
        name="add", description="Add two integers",
        fn=lambda a, b: a + b,
        schema=ToolSchema(
            parameters={"a": {"type": "integer"}, "b": {"type": "integer"}},
            required=["a", "b"],
        ),
    )
    multiply_tool = Tool(
        name="multiply", description="Multiply two integers",
        fn=lambda a, b: a * b,
        schema=ToolSchema(
            parameters={"a": {"type": "integer"}, "b": {"type": "integer"}},
            required=["a", "b"],
        ),
    )

    # Hook to observe the reasoning
    def trace_hook(payload):
        pass  # Hooks are called but we'll print from the trace instead

    hooks = HookRegistry(pre_think=[trace_hook])

    # Run the ReAct loop
    loop = ReActLoop(llm=DemoLLM(), tools=[add_tool, multiply_tool], max_steps=10, hooks=hooks)
    result = loop.run("Calculate (15 * 7) + 33")

    # Print trace
    for step in result.trace.steps:
        print(f"  [{step.phase.upper():>7}] {step.content}")

    print(f"\n  Final Answer: {result.answer}")
    print(f"  Status: {result.status}")
    print(f"  Steps: {len(result.trace.steps)}")
    assert result.answer == "138", f"Expected 138, got {result.answer}"
    print("  PASSED\n")


# ============================================================================
# Demo 2: Graph Workflow with conditional branching and a cycle
# ============================================================================

def demo_graph_workflow():
    """Code review workflow: lint → test → (pass → deploy | fail → fix → re-test)."""
    print("=" * 60)
    print("DEMO 2: Graph Workflow — Code Review with Retry Cycle")
    print("=" * 60)
    print("Workflow: lint → test → (pass → deploy | fail → fix → re-test)\n")

    def lint(state):
        state["lint_passed"] = True
        state["log"] = state.get("log", []) + ["lint: passed"]
        return state

    def run_tests(state):
        attempts = state.get("test_attempts", 0) + 1
        state["test_attempts"] = attempts
        # Fail on first attempt, pass on retry
        passed = attempts >= 2
        state["tests_passed"] = passed
        state["log"] = state.get("log", []) + [
            f"test (attempt {attempts}): {'passed' if passed else 'failed'}"
        ]
        return state

    def route_test_result(state):
        return "pass" if state.get("tests_passed") else "fail"

    def fix_code(state):
        state["log"] = state.get("log", []) + ["fix: applied code fix"]
        return state

    def deploy(state):
        state["deployed"] = True
        state["log"] = state.get("log", []) + ["deploy: success"]
        return state

    graph = StateGraph(name="code_review", max_iterations=20)
    graph.add_node("lint", lint)
    graph.add_node("test", run_tests)
    graph.add_node("fix", fix_code)
    graph.add_node("deploy", deploy)
    graph.set_entry_point("lint")
    graph.add_edge("lint", "test")
    graph.add_conditional_edges("test", route_test_result, {
        "pass": "deploy",
        "fail": "fix",
    })
    graph.add_edge("fix", "test")   # cycle: fix → re-test
    graph.add_edge("deploy", END)

    result = graph.compile().invoke({"code": "print('hello')"})

    print("  Execution log:")
    for entry in result.state.get("log", []):
        print(f"    - {entry}")

    print(f"\n  Deployed: {result.state.get('deployed')}")
    print(f"  Test attempts: {result.state.get('test_attempts')}")
    print(f"  Status: {result.status}")
    print(f"  Iterations: {len(result.steps)}")

    # Mermaid diagram
    mermaid = graph.compile().to_mermaid()
    print(f"\n  Mermaid diagram:\n")
    for line in mermaid.split("\n"):
        print(f"    {line}")

    assert result.state["deployed"] is True
    assert result.state["test_attempts"] == 2
    print("\n  PASSED\n")


# ============================================================================
# Demo 3: ReAct agent as a node inside a Graph Workflow
# ============================================================================

def demo_react_in_graph():
    """Graph: prepare_question → ReAct agent → format_answer."""
    print("=" * 60)
    print("DEMO 3: ReAct Agent as a Graph Node")
    print("=" * 60)
    print("Workflow: prepare → agent(ReAct) → format\n")

    class QuestionLLM:
        def think(self, goal, tools, history):
            if not any(h["role"] == "observe" for h in history):
                return {
                    "thought": "I should look this up.",
                    "action": "lookup",
                    "action_input": {"topic": goal},
                }
            return {
                "thought": "Found the answer.",
                "final_answer": "42 (the answer to everything)",
            }

    lookup = Tool(
        name="lookup", description="Look up a topic",
        fn=lambda topic: f"The answer to '{topic}' is 42.",
        schema=ToolSchema(parameters={"topic": {"type": "string"}}, required=["topic"]),
    )
    agent_loop = ReActLoop(llm=QuestionLLM(), tools=[lookup], max_steps=5)

    graph = StateGraph(name="agent_workflow")
    graph.add_node("prepare", lambda s: {**s, "goal": s.get("question", "unknown")})
    graph.add_node("agent", react_node(agent_loop))
    graph.add_node("format", lambda s: {**s, "output": f"Q: {s['goal']} A: {s['answer']}"})
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "agent")
    graph.add_edge("agent", "format")
    graph.add_edge("format", END)

    result = graph.compile().invoke({"question": "What is the meaning of life?"})

    print(f"  Question: {result.state['question']}")
    print(f"  Agent answer: {result.state['answer']}")
    print(f"  Formatted: {result.state['output']}")
    print(f"  Status: {result.status}")

    assert result.state["answer"] is not None
    assert "42" in result.state["answer"]
    print("\n  PASSED\n")


# ============================================================================

if __name__ == "__main__":
    demo_react_agent()
    demo_graph_workflow()
    demo_react_in_graph()
    print("=" * 60)
    print("All demos passed!")
    print("=" * 60)
