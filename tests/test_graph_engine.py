"""Tests for core/graph_engine.py — Graph Workflow Engine."""
import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from core.graph_engine import (
    END,
    StateReducer,
    NodeDef,
    EdgeDef,
    ConditionalEdgeDef,
    StateGraph,
    CompiledGraph,
    GraphResult,
    ExecutionStep,
    react_node,
    _validate_workflow_schema,
)


# ---------------------------------------------------------------------------
# Helpers — node functions
# ---------------------------------------------------------------------------

def classify_node(state):
    text = state.get("text", "")
    state["category"] = "bug" if "error" in text.lower() else "feature"
    return state


def handle_bug(state):
    state["result"] = "Bug report filed"
    return state


def handle_feature(state):
    state["result"] = "Feature added to backlog"
    return state


def increment_node(state):
    state["count"] = state.get("count", 0) + 1
    return state


def check_done(state):
    return "done" if state.get("count", 0) >= 3 else "continue"


# ---------------------------------------------------------------------------
# StateReducer
# ---------------------------------------------------------------------------

class TestStateReducer:
    def test_default_shallow_merge(self):
        reducer = StateReducer()
        result = reducer.merge({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_custom_reducer(self):
        def append_reducer(current, update):
            merged = dict(current)
            for k, v in update.items():
                if k in merged and isinstance(merged[k], list):
                    merged[k] = merged[k] + [v] if not isinstance(v, list) else merged[k] + v
                else:
                    merged[k] = v
            return merged

        reducer = StateReducer(reducer_fn=append_reducer)
        result = reducer.merge({"items": [1, 2]}, {"items": 3})
        assert result == {"items": [1, 2, 3]}


# ---------------------------------------------------------------------------
# StateGraph builder
# ---------------------------------------------------------------------------

class TestStateGraph:
    def test_add_node(self):
        g = StateGraph()
        g.add_node("a", classify_node)
        assert "a" in g._nodes

    def test_add_node_duplicate_raises(self):
        g = StateGraph()
        g.add_node("a", classify_node)
        with pytest.raises(ValueError, match="already exists"):
            g.add_node("a", classify_node)

    def test_add_node_end_reserved(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="reserved"):
            g.add_node(END, classify_node)

    def test_add_edge(self):
        g = StateGraph()
        g.add_node("a", classify_node)
        g.add_node("b", handle_bug)
        g.add_edge("a", "b")
        assert len(g._edges) == 1

    def test_add_edge_to_end(self):
        g = StateGraph()
        g.add_node("a", classify_node)
        g.add_edge("a", END)
        assert g._edges[0].target == END

    def test_add_edge_unknown_source_raises(self):
        g = StateGraph()
        g.add_node("b", handle_bug)
        with pytest.raises(ValueError, match="Unknown source"):
            g.add_edge("a", "b")

    def test_add_edge_unknown_target_raises(self):
        g = StateGraph()
        g.add_node("a", classify_node)
        with pytest.raises(ValueError, match="Unknown target"):
            g.add_edge("a", "nonexistent")

    def test_set_entry_point(self):
        g = StateGraph()
        g.add_node("start", classify_node)
        g.set_entry_point("start")
        assert g._entry_point == "start"

    def test_set_entry_point_unknown_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="Unknown"):
            g.set_entry_point("nonexistent")

    def test_compile_no_entry_raises(self):
        g = StateGraph()
        g.add_node("a", classify_node)
        with pytest.raises(ValueError, match="Entry point"):
            g.compile()

    def test_compile_no_nodes_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="at least one node"):
            g._entry_point = "x"
            g.compile()

    def test_add_conditional_edges(self):
        g = StateGraph()
        g.add_node("classify", classify_node)
        g.add_node("handle_bug", handle_bug)
        g.add_node("handle_feature", handle_feature)
        g.add_conditional_edges("classify", lambda s: s["category"], {
            "bug": "handle_bug",
            "feature": "handle_feature",
        })
        assert len(g._conditional_edges) == 1

    def test_get_graph_returns_self(self):
        g = StateGraph()
        assert g.get_graph() is g

    def test_chaining(self):
        g = StateGraph()
        result = (
            g.add_node("a", classify_node)
             .add_node("b", handle_bug)
             .add_edge("a", "b")
             .set_entry_point("a")
        )
        assert result is g


# ---------------------------------------------------------------------------
# CompiledGraph execution
# ---------------------------------------------------------------------------

class TestCompiledGraphExecution:
    def test_linear_workflow(self):
        """Simple A → B → END."""
        g = StateGraph(name="linear")
        g.add_node("classify", classify_node)
        g.add_node("handle_bug", handle_bug)
        g.set_entry_point("classify")
        g.add_edge("classify", "handle_bug")
        g.add_edge("handle_bug", END)

        result = g.compile().invoke({"text": "error in login"})
        assert result.status == "completed"
        assert result.state["category"] == "bug"
        assert result.state["result"] == "Bug report filed"
        assert len(result.steps) == 2

    def test_conditional_branching(self):
        """Conditional routing: classify → bug|feature handler."""
        g = StateGraph(name="conditional")
        g.add_node("classify", classify_node)
        g.add_node("handle_bug", handle_bug)
        g.add_node("handle_feature", handle_feature)
        g.set_entry_point("classify")
        g.add_conditional_edges("classify", lambda s: s["category"], {
            "bug": "handle_bug",
            "feature": "handle_feature",
        })
        g.add_edge("handle_bug", END)
        g.add_edge("handle_feature", END)

        # Bug path
        result = g.compile().invoke({"text": "error in login"})
        assert result.state["result"] == "Bug report filed"

        # Feature path
        result = g.compile().invoke({"text": "add dark mode"})
        assert result.state["result"] == "Feature added to backlog"

    def test_cyclic_graph_with_guard(self):
        """Loop: increment → check → (continue → increment | done → END)."""
        g = StateGraph(name="cyclic", max_iterations=20)
        g.add_node("increment", increment_node)
        g.set_entry_point("increment")
        g.add_conditional_edges("increment", check_done, {
            "continue": "increment",
            "done": END,
        })

        result = g.compile().invoke({"count": 0})
        assert result.status == "completed"
        assert result.state["count"] == 3

    def test_max_iterations_guard(self):
        """Infinite loop stopped by max_iterations."""
        def always_loop(state):
            state["n"] = state.get("n", 0) + 1
            return state

        g = StateGraph(max_iterations=5)
        g.add_node("loop", always_loop)
        g.set_entry_point("loop")
        g.add_edge("loop", "loop")  # infinite cycle

        result = g.compile().invoke({})
        assert result.status == "max_iterations"
        assert result.state["n"] == 5

    def test_node_error_captured(self):
        """Node that raises exception — error captured in state."""
        def bad_node(state):
            raise RuntimeError("boom")

        g = StateGraph()
        g.add_node("bad", bad_node)
        g.set_entry_point("bad")
        g.add_edge("bad", END)

        result = g.compile().invoke({})
        assert result.status == "completed"
        assert "__error__" in result.state
        assert "boom" in result.state["__error__"]

    def test_unknown_node_in_edge(self):
        """Reference to unknown node during execution."""
        g = StateGraph()
        g.add_node("start", lambda s: s)
        g.set_entry_point("start")
        # Manually add edge to nonexistent node
        compiled = g.compile()
        compiled._unconditional["start"] = ["nonexistent"]

        result = compiled.invoke({})
        assert result.status == "error"

    def test_implicit_end(self):
        """Node with no outgoing edges → implicit END."""
        g = StateGraph()
        g.add_node("only", lambda s: {**s, "done": True})
        g.set_entry_point("only")

        result = g.compile().invoke({})
        assert result.status == "completed"
        assert result.state["done"] is True

    def test_execution_trace(self):
        """Verify execution steps are recorded."""
        g = StateGraph()
        g.add_node("a", lambda s: {**s, "a": True})
        g.add_node("b", lambda s: {**s, "b": True})
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_edge("b", END)

        result = g.compile().invoke({})
        assert len(result.steps) == 2
        assert result.steps[0].node == "a"
        assert result.steps[1].node == "b"
        assert result.steps[0].duration_ms >= 0
        assert result.run_id != ""

    def test_state_isolation(self):
        """Node receives a copy of state, not the shared mutable state."""
        received_states = []

        def capturing_node(state):
            received_states.append(state)
            state["captured"] = True
            return state

        g = StateGraph()
        g.add_node("cap", capturing_node)
        g.set_entry_point("cap")
        g.add_edge("cap", END)

        initial = {"x": 1}
        result = g.compile().invoke(initial)
        # The initial state should not be mutated
        assert "captured" not in initial


# ---------------------------------------------------------------------------
# Subgraph nesting
# ---------------------------------------------------------------------------

class TestSubgraph:
    def test_subgraph_as_node(self):
        """A compiled graph used as a node in an outer graph."""
        # Inner graph: increment twice
        inner = StateGraph(name="inner")
        inner.add_node("inc1", increment_node)
        inner.add_node("inc2", increment_node)
        inner.set_entry_point("inc1")
        inner.add_edge("inc1", "inc2")
        inner.add_edge("inc2", END)
        compiled_inner = inner.compile()

        # Outer graph: setup → inner → finalize
        outer = StateGraph(name="outer")
        outer.add_node("setup", lambda s: {**s, "count": 0})
        outer.add_subgraph("inner", compiled_inner)
        outer.add_node("finalize", lambda s: {**s, "final": True})
        outer.set_entry_point("setup")
        outer.add_edge("setup", "inner")
        outer.add_edge("inner", "finalize")
        outer.add_edge("finalize", END)

        result = outer.compile().invoke({})
        assert result.state["count"] == 2
        assert result.state["final"] is True


# ---------------------------------------------------------------------------
# Mermaid visualization
# ---------------------------------------------------------------------------

class TestMermaid:
    def test_mermaid_output(self):
        g = StateGraph()
        g.add_node("a", lambda s: s)
        g.add_node("b", lambda s: s)
        g.add_node("c", lambda s: s)
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_conditional_edges("b", lambda s: "yes", {"yes": "c", "no": END})
        g.add_edge("c", END)

        mermaid = g.compile().to_mermaid()
        assert "graph TD" in mermaid
        assert "a --> b" in mermaid
        assert "b -->|yes| c" in mermaid
        assert "b -->|no| END" in mermaid
        assert "c --> END" in mermaid


# ---------------------------------------------------------------------------
# YAML/JSON loading
# ---------------------------------------------------------------------------

class TestYAMLLoading:
    def test_from_json(self):
        """Load and execute a graph from a JSON definition."""
        registry = {
            "classify_fn": classify_node,
            "handle_bug_fn": handle_bug,
            "handle_feature_fn": handle_feature,
            "route_fn": lambda s: s["category"],
        }
        definition = {
            "name": "test_workflow",
            "nodes": [
                {"name": "classify", "function": "classify_fn"},
                {"name": "handle_bug", "function": "handle_bug_fn"},
                {"name": "handle_feature", "function": "handle_feature_fn"},
            ],
            "entry_point": "classify",
            "conditional_edges": [
                {
                    "source": "classify",
                    "condition": "route_fn",
                    "mapping": {"bug": "handle_bug", "feature": "handle_feature"},
                },
            ],
            "edges": [
                {"source": "handle_bug", "target": "__end__"},
                {"source": "handle_feature", "target": "__end__"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(definition, f)
            f.flush()
            graph = StateGraph.from_json(f.name, node_registry=registry)

        result = graph.compile().invoke({"text": "error found"})
        assert result.state["result"] == "Bug report filed"
        os.unlink(f.name)

    def test_from_json_missing_function_raises(self):
        definition = {
            "nodes": [{"name": "a", "function": "missing_fn"}],
            "entry_point": "a",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(definition, f)
            f.flush()
            with pytest.raises(ValueError, match="not found in node_registry"):
                StateGraph.from_json(f.name, node_registry={})
        os.unlink(f.name)


class TestSchemaValidation:
    def test_valid_schema(self):
        data = {
            "nodes": [{"name": "a", "function": "fn"}],
            "entry_point": "a",
        }
        _validate_workflow_schema(data)  # should not raise

    def test_missing_required_keys(self):
        with pytest.raises(ValueError, match="Missing required"):
            _validate_workflow_schema({"name": "test"})

    def test_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown keys"):
            _validate_workflow_schema({
                "nodes": [{"name": "a", "function": "fn"}],
                "entry_point": "a",
                "bogus_key": True,
            })

    def test_invalid_node_spec(self):
        with pytest.raises(ValueError, match="name.*function"):
            _validate_workflow_schema({
                "nodes": [{"description": "missing name and function"}],
                "entry_point": "a",
            })

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_workflow_schema("not a dict")


# ---------------------------------------------------------------------------
# react_node adapter
# ---------------------------------------------------------------------------

class TestReactNode:
    def test_react_node_adapter(self):
        """Wrap a ReActLoop as a graph node."""
        # Mock the ReActLoop
        mock_loop = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "42"
        mock_result.status = "completed"
        mock_result.trace.to_dict.return_value = {"steps": []}
        mock_loop.run.return_value = mock_result

        node_fn = react_node(mock_loop, goal_key="question", answer_key="answer")
        output = node_fn({"question": "What is 6*7?"})

        mock_loop.run.assert_called_once_with("What is 6*7?")
        assert output["answer"] == "42"
        assert output["__react_status__"] == "completed"

    def test_react_node_in_graph(self):
        """Use a react_node inside a StateGraph."""
        mock_loop = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Paris"
        mock_result.status = "completed"
        mock_result.trace.to_dict.return_value = {"steps": []}
        mock_loop.run.return_value = mock_result

        g = StateGraph()
        g.add_node("setup", lambda s: {**s, "goal": "Capital of France?"})
        g.add_node("agent", react_node(mock_loop))
        g.add_node("format", lambda s: {**s, "formatted": f"Answer: {s.get('answer')}"})
        g.set_entry_point("setup")
        g.add_edge("setup", "agent")
        g.add_edge("agent", "format")
        g.add_edge("format", END)

        result = g.compile().invoke({})
        assert result.state["answer"] == "Paris"
        assert result.state["formatted"] == "Answer: Paris"
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# Five+ node workflow (acceptance criteria)
# ---------------------------------------------------------------------------

class TestFiveNodeWorkflow:
    def test_five_node_conditional_workflow(self):
        """5+ node workflow with conditional branching."""
        def ingest(state):
            state["ingested"] = True
            return state

        def analyze(state):
            state["severity"] = "high" if "critical" in state.get("text", "") else "low"
            return state

        def route_severity(state):
            return state["severity"]

        def handle_high(state):
            state["action"] = "page oncall"
            return state

        def handle_low(state):
            state["action"] = "add to backlog"
            return state

        def report(state):
            state["report"] = f"Action taken: {state['action']}"
            return state

        g = StateGraph(name="incident_response")
        g.add_node("ingest", ingest)
        g.add_node("analyze", analyze)
        g.add_node("handle_high", handle_high)
        g.add_node("handle_low", handle_low)
        g.add_node("report", report)
        g.set_entry_point("ingest")
        g.add_edge("ingest", "analyze")
        g.add_conditional_edges("analyze", route_severity, {
            "high": "handle_high",
            "low": "handle_low",
        })
        g.add_edge("handle_high", "report")
        g.add_edge("handle_low", "report")
        g.add_edge("report", END)

        # High severity
        r = g.compile().invoke({"text": "critical error in prod"})
        assert r.state["action"] == "page oncall"
        assert r.state["report"] == "Action taken: page oncall"
        assert r.status == "completed"
        assert len(r.steps) == 4  # ingest → analyze → handle_high → report

        # Low severity
        r = g.compile().invoke({"text": "minor UI tweak"})
        assert r.state["action"] == "add to backlog"
        assert len(r.steps) == 4
