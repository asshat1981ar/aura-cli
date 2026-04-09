"""Tests for core/react_engine.py — ReAct Loop Engine."""
import os
import pytest
from unittest.mock import MagicMock, patch, call

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from core.react_engine import (
    Tool,
    ToolSchema,
    TraceStep,
    ReActTrace,
    ReActLoop,
    ReActResult,
    HookRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLLM:
    """Deterministic LLM for testing.

    Accepts a list of decision dicts. Returns them in order on each
    ``think`` call. If the list is exhausted, returns a final answer.
    """

    def __init__(self, decisions: list):
        self._decisions = list(decisions)
        self._call_count = 0

    def think(self, goal, tools, history):
        if self._call_count < len(self._decisions):
            d = self._decisions[self._call_count]
            self._call_count += 1
            return d
        return {"thought": "Done.", "final_answer": "fallback answer"}


def _add_tool(a: int, b: int) -> int:
    return a + b


def _search_tool(query: str) -> str:
    return f"Results for: {query}"


def _failing_tool(x: str) -> str:
    raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# ToolSchema
# ---------------------------------------------------------------------------

class TestToolSchema:
    def test_validate_ok(self):
        schema = ToolSchema(
            parameters={"query": {"type": "string"}},
            required=["query"],
        )
        assert schema.validate({"query": "hello"}) == []

    def test_validate_missing_required(self):
        schema = ToolSchema(
            parameters={"query": {"type": "string"}},
            required=["query"],
        )
        errors = schema.validate({})
        assert len(errors) == 1
        assert "Missing required" in errors[0]

    def test_validate_wrong_type(self):
        schema = ToolSchema(
            parameters={"count": {"type": "integer"}},
            required=[],
        )
        errors = schema.validate({"count": "not_a_number"})
        assert len(errors) == 1
        assert "expected type" in errors[0]

    def test_validate_unknown_type_passes(self):
        schema = ToolSchema(
            parameters={"x": {"type": "custom_type"}},
        )
        assert schema.validate({"x": "anything"}) == []

    def test_validate_empty_schema(self):
        schema = ToolSchema()
        assert schema.validate({"anything": 42}) == []


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TestTool:
    def test_creation(self):
        tool = Tool(name="add", description="Add two numbers", fn=_add_tool)
        assert tool.name == "add"
        assert tool.description == "Add two numbers"

    def test_tool_with_schema(self):
        schema = ToolSchema(
            parameters={"a": {"type": "integer"}, "b": {"type": "integer"}},
            required=["a", "b"],
        )
        tool = Tool(name="add", description="Add", fn=_add_tool, schema=schema)
        assert tool.schema.required == ["a", "b"]


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------

class TestHookRegistry:
    def test_fire_hooks(self):
        record = []
        hooks = HookRegistry(
            pre_think=[lambda p: record.append(("pre_think", p))],
            post_act=[lambda p: record.append(("post_act", p))],
            post_observe=[lambda p: record.append(("post_observe", p))],
        )
        hooks.fire("pre_think", {"step": 1})
        hooks.fire("post_act", {"action": "search"})
        hooks.fire("post_observe", {"obs": "result"})
        assert len(record) == 3
        assert record[0][0] == "pre_think"
        assert record[1][0] == "post_act"
        assert record[2][0] == "post_observe"

    def test_hook_error_does_not_crash(self):
        def bad_hook(p):
            raise ValueError("hook error")

        hooks = HookRegistry(pre_think=[bad_hook])
        # Should not raise
        hooks.fire("pre_think", {"step": 1})

    def test_unknown_phase(self):
        hooks = HookRegistry()
        # Should not raise for unknown phase
        hooks.fire("unknown_phase", {})


# ---------------------------------------------------------------------------
# ReActTrace
# ---------------------------------------------------------------------------

class TestReActTrace:
    def test_to_dict(self):
        trace = ReActTrace(run_id="abc123", goal="test goal")
        trace.steps.append(TraceStep(
            step_number=1, phase="think",
            timestamp="2026-01-01T00:00:00Z",
            content="thinking...",
            duration_ms=10.0,
        ))
        trace.final_answer = "42"
        trace.status = "completed"
        d = trace.to_dict()
        assert d["run_id"] == "abc123"
        assert d["goal"] == "test goal"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["phase"] == "think"
        assert d["final_answer"] == "42"
        assert d["status"] == "completed"


# ---------------------------------------------------------------------------
# ReActLoop
# ---------------------------------------------------------------------------

class TestReActLoop:
    def test_simple_final_answer(self):
        """LLM immediately returns a final answer."""
        llm = FakeLLM([
            {"thought": "I know this.", "final_answer": "Paris"},
        ])
        loop = ReActLoop(llm=llm, tools=[])
        result = loop.run("What is the capital of France?")
        assert result.answer == "Paris"
        assert result.status == "completed"
        assert len(result.trace.steps) == 1  # one think step

    def test_tool_use_then_answer(self):
        """Agent calls a tool, observes result, then answers."""
        llm = FakeLLM([
            {
                "thought": "I should search.",
                "action": "search",
                "action_input": {"query": "capital of France"},
            },
            {
                "thought": "I found it.",
                "final_answer": "Paris",
            },
        ])
        search = Tool(
            name="search", description="Search",
            fn=_search_tool,
            schema=ToolSchema(parameters={"query": {"type": "string"}}, required=["query"]),
        )
        loop = ReActLoop(llm=llm, tools=[search])
        result = loop.run("capital of France")

        assert result.answer == "Paris"
        assert result.status == "completed"
        # Step 1: think + act + observe. Step 2: think (final answer)
        phases = [s.phase for s in result.trace.steps]
        assert phases == ["think", "act", "observe", "think"]

    def test_unknown_tool(self):
        """Agent calls a tool that doesn't exist — gets error observation."""
        llm = FakeLLM([
            {
                "thought": "Try a tool.",
                "action": "nonexistent",
                "action_input": {},
            },
            {"thought": "Got error, done.", "final_answer": "error handled"},
        ])
        loop = ReActLoop(llm=llm, tools=[])
        result = loop.run("test")
        # The observation should contain an error about unknown tool
        obs_steps = [s for s in result.trace.steps if s.phase == "observe"]
        assert len(obs_steps) == 1
        assert "Unknown tool" in obs_steps[0].content

    def test_tool_execution_failure(self):
        """Tool raises an exception — captured as error observation."""
        llm = FakeLLM([
            {
                "thought": "Try failing tool.",
                "action": "fail",
                "action_input": {"x": "data"},
            },
            {"thought": "Handled error.", "final_answer": "recovered"},
        ])
        fail_tool = Tool(
            name="fail", description="Fails",
            fn=_failing_tool,
            schema=ToolSchema(parameters={"x": {"type": "string"}}),
        )
        loop = ReActLoop(llm=llm, tools=[fail_tool])
        result = loop.run("test")
        assert result.answer == "recovered"
        obs_steps = [s for s in result.trace.steps if s.phase == "observe"]
        assert "Error executing tool" in obs_steps[0].content

    def test_max_steps_reached(self):
        """Loop hits max_steps without final answer."""
        # LLM always returns an action
        decisions = [
            {
                "thought": f"Step {i}",
                "action": "search",
                "action_input": {"query": "loop"},
            }
            for i in range(20)
        ]
        llm = FakeLLM(decisions)
        search = Tool(name="search", description="Search", fn=_search_tool,
                       schema=ToolSchema(parameters={"query": {"type": "string"}}))
        loop = ReActLoop(llm=llm, tools=[search], max_steps=3)
        result = loop.run("test")
        assert result.status == "max_steps"
        assert result.answer is None

    def test_timeout(self):
        """Loop aborts on timeout."""
        import time

        class SlowLLM:
            def think(self, goal, tools, history):
                time.sleep(0.05)
                return {"thought": "slow", "action": "search", "action_input": {"query": "x"}}

        search = Tool(name="search", description="Search", fn=_search_tool,
                       schema=ToolSchema(parameters={"query": {"type": "string"}}))
        loop = ReActLoop(llm=SlowLLM(), tools=[search], max_steps=100, timeout_s=0.08)
        result = loop.run("test")
        assert result.status in ("aborted", "max_steps")

    def test_register_tool_duplicate_raises(self):
        llm = FakeLLM([])
        tool = Tool(name="search", description="Search", fn=_search_tool)
        loop = ReActLoop(llm=llm, tools=[tool])
        with pytest.raises(ValueError, match="already registered"):
            loop.register_tool(Tool(name="search", description="Dup", fn=_search_tool))

    def test_hooks_called_during_execution(self):
        """Verify hooks are called at correct phases."""
        record = []
        hooks = HookRegistry(
            pre_think=[lambda p: record.append("pre_think")],
            post_act=[lambda p: record.append("post_act")],
            post_observe=[lambda p: record.append("post_observe")],
        )
        llm = FakeLLM([
            {
                "thought": "Use tool.",
                "action": "search",
                "action_input": {"query": "test"},
            },
            {"thought": "Done.", "final_answer": "result"},
        ])
        search = Tool(name="search", description="Search", fn=_search_tool,
                       schema=ToolSchema(parameters={"query": {"type": "string"}}))
        loop = ReActLoop(llm=llm, tools=[search], hooks=hooks)
        result = loop.run("test")

        assert result.status == "completed"
        assert "pre_think" in record
        assert "post_act" in record
        assert "post_observe" in record

    def test_stop_condition(self):
        """Custom stop condition halts the loop."""
        llm = FakeLLM([
            {
                "thought": "Step 1",
                "action": "search",
                "action_input": {"query": "stop"},
            },
        ] * 10)
        search = Tool(name="search", description="Search", fn=_search_tool,
                       schema=ToolSchema(parameters={"query": {"type": "string"}}))

        def stop_after_2(trace):
            return len([s for s in trace.steps if s.phase == "think"]) >= 2

        loop = ReActLoop(llm=llm, tools=[search], max_steps=10, stop_condition=stop_after_2)
        result = loop.run("test")
        assert result.status == "completed"
        think_steps = [s for s in result.trace.steps if s.phase == "think"]
        assert len(think_steps) == 2

    def test_schema_validation_error(self):
        """Tool with invalid arguments gets validation error observation."""
        llm = FakeLLM([
            {
                "thought": "Add numbers.",
                "action": "add",
                "action_input": {"a": "not_int", "b": 2},
            },
            {"thought": "Done.", "final_answer": "handled"},
        ])
        schema = ToolSchema(
            parameters={"a": {"type": "integer"}, "b": {"type": "integer"}},
            required=["a", "b"],
        )
        add_tool = Tool(name="add", description="Add", fn=_add_tool, schema=schema)
        loop = ReActLoop(llm=llm, tools=[add_tool])
        result = loop.run("test")
        obs_steps = [s for s in result.trace.steps if s.phase == "observe"]
        assert "Invalid arguments" in obs_steps[0].content

    def test_no_action_no_final_answer(self):
        """LLM returns neither action nor final_answer — treated as error."""
        llm = FakeLLM([
            {"thought": "I'm confused."},
            {"thought": "Now I know.", "final_answer": "ok"},
        ])
        loop = ReActLoop(llm=llm, tools=[])
        result = loop.run("test")
        assert result.answer == "ok"
        obs_steps = [s for s in result.trace.steps if s.phase == "observe"]
        assert "neither an action nor a final answer" in obs_steps[0].content

    def test_trace_run_id(self):
        llm = FakeLLM([{"thought": "Done.", "final_answer": "yes"}])
        loop = ReActLoop(llm=llm)
        result = loop.run("test")
        assert len(result.trace.run_id) == 12

    def test_trace_total_duration(self):
        llm = FakeLLM([{"thought": "Done.", "final_answer": "yes"}])
        loop = ReActLoop(llm=llm)
        result = loop.run("test")
        assert result.trace.total_duration_ms > 0
