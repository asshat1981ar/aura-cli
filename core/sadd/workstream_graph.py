"""DAG-based workstream graph with state machine for parallel execution tracking."""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.sadd.types import WorkstreamResult, WorkstreamSpec


# ---------------------------------------------------------------------------
# Node wrapper — adds execution state around an immutable spec
# ---------------------------------------------------------------------------


@dataclass
class WorkstreamNode:
    """A workstream spec decorated with execution state."""

    spec: WorkstreamSpec
    status: Literal["pending", "running", "completed", "failed", "blocked"] = "pending"
    result: Optional[WorkstreamResult] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# ---------------------------------------------------------------------------
# DAG graph with topological ordering and state transitions
# ---------------------------------------------------------------------------


class WorkstreamGraph:
    """Directed acyclic graph of workstreams with dependency-aware scheduling.

    Nodes are ``WorkstreamNode`` instances keyed by workstream ID.  Edges
    encode *depends-on* relationships so that a workstream only becomes
    ready once all of its dependencies have completed.

    The graph validates acyclicity at construction time and exposes both
    static topological waves (``execution_waves``) and a dynamic readiness
    query (``ready_workstreams``) that accounts for runtime status.
    """

    def __init__(self, specs: list[WorkstreamSpec]) -> None:
        # --- node storage ---
        self._nodes: Dict[str, WorkstreamNode] = {}
        # adjacency: id -> list of workstream IDs that *depend on* id
        self._dependents: Dict[str, List[str]] = {}
        # reverse adjacency: id -> list of workstream IDs that id *depends on*
        self._dependencies: Dict[str, List[str]] = {}

        for spec in specs:
            self._nodes[spec.id] = WorkstreamNode(spec=spec)
            self._dependents[spec.id] = []
            self._dependencies[spec.id] = list(spec.depends_on)

        # Validate that all depends_on references exist
        all_ids = set(self._nodes)
        for ws_id, deps in self._dependencies.items():
            for dep_id in deps:
                if dep_id not in all_ids:
                    raise ValueError(f"Workstream {ws_id!r} depends on unknown ID {dep_id!r}")

        # Build forward adjacency (dependents)
        for ws_id, deps in self._dependencies.items():
            for dep_id in deps:
                self._dependents[dep_id].append(ws_id)

        # Validate no cycles via DFS
        self._validate_no_cycles()

    # -- cycle detection ----------------------------------------------------

    def _validate_no_cycles(self) -> None:
        """Raise ``ValueError`` if the dependency graph contains a cycle."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {ws_id: WHITE for ws_id in self._nodes}
        parent: Dict[str, Optional[str]] = {ws_id: None for ws_id in self._nodes}

        def dfs(node: str) -> Optional[list[str]]:
            color[node] = GRAY
            for neighbour in self._dependents[node]:
                if color[neighbour] == GRAY:
                    # Reconstruct cycle path
                    cycle = [neighbour, node]
                    cur = node
                    while cur != neighbour:
                        cur = parent[cur]  # type: ignore[assignment]
                        if cur is None:
                            break
                        cycle.append(cur)
                    cycle.reverse()
                    # Ensure the cycle starts and ends with the same node
                    if cycle[0] != cycle[-1]:
                        cycle.append(cycle[0])
                    return cycle
                if color[neighbour] == WHITE:
                    parent[neighbour] = node
                    result = dfs(neighbour)
                    if result is not None:
                        return result
            color[node] = BLACK
            return None

        for ws_id in self._nodes:
            if color[ws_id] == WHITE:
                cycle = dfs(ws_id)
                if cycle is not None:
                    path_str = " \u2192 ".join(cycle)
                    raise ValueError(f"Cycle detected: {path_str}")

    # -- topological waves --------------------------------------------------

    def execution_waves(self) -> list[list[str]]:
        """Return workstream IDs grouped into dependency waves (Kahn's algorithm).

        Wave 0 contains nodes with no dependencies.  Wave *n* contains nodes
        whose dependencies are all satisfied by waves 0 .. n-1.
        """
        in_degree: Dict[str, int] = {ws_id: len(deps) for ws_id, deps in self._dependencies.items()}
        queue: deque[str] = deque(ws_id for ws_id, deg in in_degree.items() if deg == 0)
        waves: list[list[str]] = []

        while queue:
            wave = list(queue)
            waves.append(wave)
            next_queue: deque[str] = deque()
            for ws_id in wave:
                for dep_id in self._dependents[ws_id]:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        next_queue.append(dep_id)
            queue = next_queue

        return waves

    # -- dynamic readiness --------------------------------------------------

    def ready_workstreams(self) -> list[str]:
        """Return IDs of workstreams that are pending and have all deps completed."""
        ready: list[str] = []
        for ws_id, node in self._nodes.items():
            if node.status != "pending":
                continue
            if all(self._nodes[dep].status == "completed" for dep in self._dependencies[ws_id]):
                ready.append(ws_id)
        return ready

    # -- state transitions --------------------------------------------------

    def mark_running(self, ws_id: str) -> None:
        """Transition a workstream to running state."""
        node = self._require_node(ws_id)
        node.status = "running"
        node.started_at = time.time()

    def mark_completed(self, ws_id: str, result: WorkstreamResult) -> None:
        """Transition a workstream to completed and store its result."""
        node = self._require_node(ws_id)
        node.status = "completed"
        node.result = result
        node.completed_at = time.time()

    def mark_failed(self, ws_id: str, error: str) -> None:
        """Transition a workstream to failed and block all transitive dependents."""
        node = self._require_node(ws_id)
        node.status = "failed"
        node.completed_at = time.time()
        node.result = WorkstreamResult(ws_id=ws_id, status="failed", error=error)

        # Block all transitive dependents via BFS
        queue: deque[str] = deque(self._dependents[ws_id])
        visited: set[str] = set()
        while queue:
            dep_id = queue.popleft()
            if dep_id in visited:
                continue
            visited.add(dep_id)
            dep_node = self._nodes[dep_id]
            if dep_node.status in ("pending", "blocked"):
                dep_node.status = "blocked"
                queue.extend(self._dependents[dep_id])

    # -- queries ------------------------------------------------------------

    def is_complete(self) -> bool:
        """True when no workstreams are pending or running."""
        terminal = {"completed", "failed", "blocked", "skipped"}
        return all(node.status in terminal for node in self._nodes.values())

    def blocked_workstreams(self) -> list[str]:
        """Return IDs of workstreams in blocked state."""
        return [ws_id for ws_id, node in self._nodes.items() if node.status == "blocked"]

    def get_node(self, ws_id: str) -> WorkstreamNode:
        """Return the node for *ws_id*, raising ``KeyError`` if not found."""
        return self._nodes[ws_id]

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the full graph state for persistence."""
        nodes_data: Dict[str, Any] = {}
        for ws_id, node in self._nodes.items():
            node_dict = dataclasses.asdict(node)
            nodes_data[ws_id] = node_dict
        return {
            "nodes": nodes_data,
            "dependents": {k: list(v) for k, v in self._dependents.items()},
            "dependencies": {k: list(v) for k, v in self._dependencies.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkstreamGraph:
        """Reconstruct a graph from serialized data, restoring node states."""
        nodes_data: Dict[str, Any] = data["nodes"]

        # Rebuild specs from serialized node data
        specs: list[WorkstreamSpec] = []
        for ws_id, node_dict in nodes_data.items():
            spec_dict = node_dict["spec"]
            spec = WorkstreamSpec(**{k: v for k, v in spec_dict.items() if k in {f.name for f in dataclasses.fields(WorkstreamSpec)}})
            specs.append(spec)

        # Construct graph (validates structure)
        graph = cls(specs)

        # Restore runtime state
        for ws_id, node_dict in nodes_data.items():
            node = graph._nodes[ws_id]
            node.status = node_dict["status"]
            node.started_at = node_dict.get("started_at")
            node.completed_at = node_dict.get("completed_at")
            result_data = node_dict.get("result")
            if result_data is not None:
                node.result = WorkstreamResult.from_dict(result_data)

        return graph

    # -- internal helpers ---------------------------------------------------

    def _require_node(self, ws_id: str) -> WorkstreamNode:
        """Return the node or raise ``KeyError`` with a clear message."""
        try:
            return self._nodes[ws_id]
        except KeyError:
            raise KeyError(f"Unknown workstream ID: {ws_id!r}") from None
