"""
Graph Workflow Engine — directed-graph execution engine for AURA workflows.

Workflows are defined as configurations (YAML/JSON or programmatic API).
Nodes are agents, tools, or LLM calls. Edges define data flow with
optional conditions. Supports cyclic graphs with max-iteration guards
and subgraph nesting.

Usage (programmatic):
    from core.graph_engine import StateGraph

    def classify(state):
        state["category"] = "bug" if "error" in state["text"] else "feature"
        return state

    def handle_bug(state):
        state["result"] = "Filed bug report"
        return state

    graph = StateGraph(state_schema={"text": str, "category": str, "result": str})
    graph.add_node("classify", classify)
    graph.add_node("handle_bug", handle_bug)
    graph.add_node("handle_feature", lambda s: {**s, "result": "Added to backlog"})
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", lambda s: s["category"], {
        "bug": "handle_bug",
        "feature": "handle_feature",
    })
    workflow = graph.compile()
    result = workflow.invoke({"text": "error in login"})

Usage (YAML):
    workflow = StateGraph.from_yaml("my_workflow.yaml")
    compiled = workflow.compile()
    result = compiled.invoke(initial_state)
"""
from __future__ import annotations

import copy
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Type, Union

from core.logging_utils import log_json

# Sentinel for graph end
END = "__end__"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

@dataclass
class StateReducer:
    """Defines how partial node outputs merge into shared state.

    By default, node outputs are shallow-merged (dict.update).
    A custom reducer_fn can implement append-style or deep-merge logic.
    """
    reducer_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None

    def merge(self, current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        if self.reducer_fn:
            return self.reducer_fn(current, update)
        # Default: shallow merge
        merged = dict(current)
        merged.update(update)
        return merged


# ---------------------------------------------------------------------------
# Graph definition types
# ---------------------------------------------------------------------------

@dataclass
class NodeDef:
    """A node in the graph."""
    name: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    is_subgraph: bool = False


@dataclass
class EdgeDef:
    """An unconditional edge."""
    source: str
    target: str


@dataclass
class ConditionalEdgeDef:
    """A conditional edge with routing based on state."""
    source: str
    condition_fn: Callable[[Dict[str, Any]], str]
    mapping: Dict[str, str]  # condition result → target node name


# ---------------------------------------------------------------------------
# Compiled graph
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStep:
    """One step in a graph execution trace."""
    node: str
    timestamp: str
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    duration_ms: float
    iteration: int


@dataclass
class GraphResult:
    """Result of a compiled graph execution."""
    state: Dict[str, Any]
    steps: List[ExecutionStep] = field(default_factory=list)
    status: Literal["completed", "max_iterations", "error"] = "completed"
    run_id: str = ""
    total_duration_ms: float = 0.0


class CompiledGraph:
    """An executable workflow compiled from a StateGraph.

    Executes nodes following edges (unconditional and conditional),
    supports cycles with max-iteration guards, and nested subgraphs.
    """

    def __init__(
        self,
        nodes: Dict[str, NodeDef],
        edges: List[EdgeDef],
        conditional_edges: List[ConditionalEdgeDef],
        entry_point: str,
        state_schema: Optional[Dict[str, Any]],
        reducer: StateReducer,
        max_iterations: int,
        graph_name: str = "",
    ):
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point
        self._state_schema = state_schema
        self._reducer = reducer
        self._max_iterations = max_iterations
        self._graph_name = graph_name

        # Build adjacency for quick lookup
        self._unconditional: Dict[str, List[str]] = {}
        for e in edges:
            self._unconditional.setdefault(e.source, []).append(e.target)

        self._conditional: Dict[str, List[ConditionalEdgeDef]] = {}
        for ce in conditional_edges:
            self._conditional.setdefault(ce.source, []).append(ce)

    def invoke(self, initial_state: Dict[str, Any]) -> GraphResult:
        """Execute the graph starting from the entry point."""
        run_id = uuid.uuid4().hex[:12]
        start_time = time.monotonic()
        state = dict(initial_state)
        steps: List[ExecutionStep] = []
        iteration = 0
        current_node = self._entry_point

        log_json("INFO", "graph_invoke_start", details={
            "run_id": run_id, "graph": self._graph_name,
            "entry": self._entry_point,
        })

        try:
            while current_node != END and iteration < self._max_iterations:
                iteration += 1
                node_def = self._nodes.get(current_node)
                if node_def is None:
                    log_json("ERROR", "graph_unknown_node", details={
                        "run_id": run_id, "node": current_node,
                    })
                    return GraphResult(
                        state=state, steps=steps, status="error",
                        run_id=run_id,
                        total_duration_ms=(time.monotonic() - start_time) * 1000,
                    )

                # Execute node
                input_snapshot = copy.deepcopy(state)
                node_start = time.monotonic()
                try:
                    output = node_def.fn(copy.deepcopy(state))
                except Exception as exc:
                    log_json("WARN", "graph_node_error", details={
                        "run_id": run_id, "node": current_node,
                        "error": str(exc),
                    })
                    output = dict(state)
                    output["__error__"] = str(exc)

                node_ms = (time.monotonic() - node_start) * 1000

                # Merge output into state
                state = self._reducer.merge(state, output)

                steps.append(ExecutionStep(
                    node=current_node,
                    timestamp=_now_iso(),
                    input_state=input_snapshot,
                    output_state=copy.deepcopy(state),
                    duration_ms=node_ms,
                    iteration=iteration,
                ))

                # Resolve next node
                current_node = self._resolve_next(current_node, state)

            status = "completed" if current_node == END else "max_iterations"
            if status == "max_iterations":
                log_json("WARN", "graph_max_iterations", details={
                    "run_id": run_id, "iterations": iteration,
                    "max": self._max_iterations,
                })

        except Exception:
            status = "error"
            log_json("ERROR", "graph_invoke_error", details={
                "run_id": run_id, "error": traceback.format_exc(),
            })

        total_ms = (time.monotonic() - start_time) * 1000
        log_json("INFO", "graph_invoke_end", details={
            "run_id": run_id, "status": status,
            "iterations": iteration, "duration_ms": total_ms,
        })

        return GraphResult(
            state=state, steps=steps, status=status,
            run_id=run_id, total_duration_ms=total_ms,
        )

    def _resolve_next(self, current: str, state: Dict[str, Any]) -> str:
        """Determine the next node from edges."""
        # Conditional edges take priority
        cond_edges = self._conditional.get(current, [])
        for ce in cond_edges:
            try:
                result = ce.condition_fn(state)
                target = ce.mapping.get(result)
                if target is not None:
                    return target
            except Exception:
                log_json("WARN", "graph_condition_error", details={
                    "source": current, "error": traceback.format_exc(),
                })

        # Fall back to unconditional edges
        targets = self._unconditional.get(current, [])
        if targets:
            return targets[0]  # First unconditional edge

        # No edges — implicit end
        return END

    def to_mermaid(self) -> str:
        """Render the graph as a Mermaid diagram."""
        lines = ["graph TD"]
        for e in self._edges:
            target_label = "END" if e.target == END else e.target
            lines.append(f"    {e.source} --> {target_label}")
        for ce in self._conditional_edges:
            for label, target in ce.mapping.items():
                target_label = "END" if target == END else target
                lines.append(f"    {ce.source} -->|{label}| {target_label}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StateGraph builder
# ---------------------------------------------------------------------------

class StateGraph:
    """Builder for constructing graph workflows.

    Args:
        state_schema: Optional dict mapping key names to types for documentation.
        reducer: Custom state reducer (default: shallow merge).
        max_iterations: Maximum graph-execution iterations (cycle guard).
        name: Human-readable graph name.
    """

    def __init__(
        self,
        state_schema: Optional[Dict[str, Any]] = None,
        reducer: Optional[StateReducer] = None,
        max_iterations: int = 50,
        name: str = "",
    ):
        self._state_schema = state_schema
        self._reducer = reducer or StateReducer()
        self._max_iterations = max_iterations
        self._name = name
        self._nodes: Dict[str, NodeDef] = {}
        self._edges: List[EdgeDef] = []
        self._conditional_edges: List[ConditionalEdgeDef] = []
        self._entry_point: Optional[str] = None

    def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> "StateGraph":
        """Register a node.

        A node function receives the current state dict and returns a
        (possibly partial) state dict that gets merged via the reducer.
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        if name == END:
            raise ValueError(f"'{END}' is reserved as the graph end sentinel")
        self._nodes[name] = NodeDef(name=name, fn=fn)
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        """Add an unconditional edge from *source* to *target*.

        Use ``END`` (imported from this module) as *target* to mark a terminal edge.
        """
        self._validate_node_ref(source, is_source=True)
        if target != END:
            self._validate_node_ref(target, is_source=False)
        self._edges.append(EdgeDef(source=source, target=target))
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition_fn: Callable[[Dict[str, Any]], str],
        mapping: Dict[str, str],
    ) -> "StateGraph":
        """Add conditional edges from *source*, routed by *condition_fn*.

        *condition_fn* receives the current state and returns a string key.
        *mapping* maps those keys to target node names (or ``END``).
        """
        self._validate_node_ref(source, is_source=True)
        for target in mapping.values():
            if target != END:
                self._validate_node_ref(target, is_source=False)
        self._conditional_edges.append(
            ConditionalEdgeDef(source=source, condition_fn=condition_fn, mapping=mapping)
        )
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        """Designate the start node."""
        self._validate_node_ref(name, is_source=True)
        self._entry_point = name
        return self

    def compile(self) -> CompiledGraph:
        """Compile the graph into an executable ``CompiledGraph``."""
        if self._entry_point is None:
            raise ValueError("Entry point must be set before compiling (call set_entry_point)")
        if not self._nodes:
            raise ValueError("Graph must have at least one node")
        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=list(self._edges),
            conditional_edges=list(self._conditional_edges),
            entry_point=self._entry_point,
            state_schema=self._state_schema,
            reducer=self._reducer,
            max_iterations=self._max_iterations,
            graph_name=self._name,
        )

    # -- Subgraph support ---------------------------------------------------

    def add_subgraph(self, name: str, subgraph: CompiledGraph) -> "StateGraph":
        """Add a compiled graph as a node (subgraph nesting).

        The subgraph's ``invoke`` method will be called with the current state.
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")

        def _run_subgraph(state: Dict[str, Any]) -> Dict[str, Any]:
            result = subgraph.invoke(state)
            return result.state

        self._nodes[name] = NodeDef(name=name, fn=_run_subgraph, is_subgraph=True)
        return self

    # -- YAML / JSON loading ------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path], node_registry: Optional[Dict[str, Callable]] = None) -> "StateGraph":
        """Load a graph definition from a YAML file.

        The *node_registry* maps node function names (strings in YAML) to
        actual callables.
        """
        import yaml
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data, node_registry)

    @classmethod
    def from_json(cls, path: Union[str, Path], node_registry: Optional[Dict[str, Callable]] = None) -> "StateGraph":
        """Load a graph definition from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data, node_registry)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any], node_registry: Optional[Dict[str, Callable]] = None) -> "StateGraph":
        """Build a StateGraph from a parsed YAML/JSON dict.

        Expected schema:
            name: str
            max_iterations: int (optional, default 50)
            state_schema: dict (optional)
            nodes:
              - name: str
                function: str  (key in node_registry)
            entry_point: str
            edges:
              - source: str
                target: str
            conditional_edges:
              - source: str
                condition: str  (key in node_registry)
                mapping:
                  key: target_node
        """
        registry = node_registry or {}
        _validate_workflow_schema(data)

        graph = cls(
            state_schema=data.get("state_schema"),
            max_iterations=data.get("max_iterations", 50),
            name=data.get("name", ""),
        )

        # Nodes
        for node_spec in data.get("nodes", []):
            fn_name = node_spec["function"]
            fn = registry.get(fn_name)
            if fn is None:
                raise ValueError(f"Node function '{fn_name}' not found in node_registry")
            graph.add_node(node_spec["name"], fn)

        # Entry point
        graph.set_entry_point(data["entry_point"])

        # Edges
        for edge_spec in data.get("edges", []):
            target = edge_spec["target"]
            if target == "__end__":
                target = END
            graph.add_edge(edge_spec["source"], target)

        # Conditional edges
        for ce_spec in data.get("conditional_edges", []):
            cond_name = ce_spec["condition"]
            cond_fn = registry.get(cond_name)
            if cond_fn is None:
                raise ValueError(f"Condition function '{cond_name}' not found in node_registry")
            mapping = {
                k: (END if v == "__end__" else v)
                for k, v in ce_spec["mapping"].items()
            }
            graph.add_conditional_edges(ce_spec["source"], cond_fn, mapping)

        return graph

    # -- Validation helpers -------------------------------------------------

    def _validate_node_ref(self, name: str, is_source: bool) -> None:
        """Raise if a referenced node doesn't exist."""
        if name not in self._nodes:
            role = "source" if is_source else "target"
            raise ValueError(f"Unknown {role} node '{name}'. Add it with add_node() first.")

    def get_graph(self) -> "StateGraph":
        """Return self — convenience for chaining ``get_graph().to_mermaid()``."""
        return self

    def to_mermaid(self) -> str:
        """Render the graph definition as a Mermaid diagram (before compilation)."""
        compiled = self.compile()
        return compiled.to_mermaid()


# ---------------------------------------------------------------------------
# Schema validation for YAML/JSON workflows
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"nodes", "entry_point"}
_ALLOWED_TOP_KEYS = {"name", "max_iterations", "state_schema", "nodes", "entry_point", "edges", "conditional_edges"}


def _validate_workflow_schema(data: Dict[str, Any]) -> None:
    """Validate a workflow definition dict. Raises ValueError on problems."""
    if not isinstance(data, dict):
        raise ValueError("Workflow definition must be a dict")
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    unknown = set(data.keys()) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in workflow definition: {unknown}")
    for node_spec in data.get("nodes", []):
        if "name" not in node_spec or "function" not in node_spec:
            raise ValueError(f"Each node must have 'name' and 'function': {node_spec}")


# ---------------------------------------------------------------------------
# ReAct-as-node adapter
# ---------------------------------------------------------------------------

def react_node(react_loop: Any, goal_key: str = "goal", answer_key: str = "answer") -> Callable:
    """Wrap a ``ReActLoop`` so it can be used as a node in a StateGraph.

    Args:
        react_loop: A ``ReActLoop`` instance.
        goal_key: State key containing the goal string.
        answer_key: State key to write the answer into.

    Returns:
        A node function ``(state: dict) -> dict``.
    """
    def _node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        goal = state.get(goal_key, "")
        result = react_loop.run(goal)
        return {
            answer_key: result.answer,
            "__react_status__": result.status,
            "__react_trace__": result.trace.to_dict(),
        }
    return _node_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
