"""Top-level SADD session coordinator.

Manages the full lifecycle of a Spec-Aware Design Decomposition session:
decompose a design spec into workstreams and execute them with parallel
sub-agents, respecting dependency ordering.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.sadd.sub_agent_runner import SubAgentRunner
from core.sadd.types import (
    DesignSpec,
    SessionConfig,
    SessionReport,
    WorkstreamOutcome,
    WorkstreamResult,
)
from core.sadd.workstream_graph import WorkstreamGraph


def create_orchestrator_factory(
    brain: Any,
    project_root: str | Path = ".",
    model_adapter: Any = None,
    memory_store: Any = None,
) -> Callable[[], Any]:
    """Create a factory that produces fresh LoopOrchestrator instances.

    Each call to the returned factory creates an isolated orchestrator
    with its own agents but sharing the provided brain.
    """
    from pathlib import Path as _Path
    from agents.registry import default_agents
    from core.orchestrator import LoopOrchestrator
    from core.policy import Policy
    from memory.store import MemoryStore

    _root = _Path(project_root)
    _store = memory_store or MemoryStore(_root / "memory")

    def _factory() -> LoopOrchestrator:
        if model_adapter is None:
            from core.model_adapter import ModelAdapter

            _model = ModelAdapter()
        else:
            _model = model_adapter
        agents = default_agents(brain, _model)
        return LoopOrchestrator(
            agents=agents,
            brain=brain,
            model=_model,
            memory_store=_store,
            project_root=_root,
            policy=Policy(max_cycles=10),
        )

    return _factory


class SessionCoordinator:
    """Orchestrate a full SADD session: decompose spec -> execute workstreams -> report."""

    def __init__(
        self,
        design_spec: DesignSpec,
        orchestrator_factory: Callable[[], Any],
        brain: Any,
        config: SessionConfig = SessionConfig(),
        session_store: Any = None,
        mcp_bridge: Optional["MCPToolBridge"] = None,
    ) -> None:
        self._spec = design_spec
        self._orchestrator_factory = orchestrator_factory
        self._brain = brain
        self._config = config
        self._session_id = str(uuid.uuid4())
        self._graph: WorkstreamGraph | None = None
        self._results: Dict[str, WorkstreamResult] = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        self._retried: set[str] = set()
        self._store = session_store
        self._mcp_bridge = mcp_bridge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SessionReport:
        """Drive the full session to completion and return a report."""
        started_at = time.time()

        self._graph = WorkstreamGraph(self._spec.workstreams)
        self._logger.info(
            "SADD session %s started — %d workstreams from spec %r",
            self._session_id,
            len(self._spec.workstreams),
            self._spec.title,
        )
        print(f"🚀 SADD session {self._session_id} starting — {len(self._spec.workstreams)} workstreams")

        # Persist session if store is available.
        if self._store:
            self._store.create_session(self._spec, self._config, self._session_id)
            self._store.update_status(self._session_id, "running")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_parallel,
        ) as executor:
            while not self._graph.is_complete():
                ready = self._graph.ready_workstreams()

                if not ready:
                    # Nothing ready but graph not complete — deadlock or
                    # all remaining workstreams are blocked/failed.
                    self._logger.warning(
                        "SADD session %s: no ready workstreams and graph not complete — possible deadlock, stopping",
                        self._session_id,
                    )
                    break

                # Mark each ready workstream as running and submit.
                futures: Dict[concurrent.futures.Future[WorkstreamResult], str] = {}
                for ws_id in ready:
                    with self._lock:
                        self._graph.mark_running(ws_id)
                    future = executor.submit(self._execute_workstream, ws_id)
                    futures[future] = ws_id

                # Process completed futures as they finish.
                for future in concurrent.futures.as_completed(futures):
                    ws_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = WorkstreamResult(
                            ws_id=ws_id,
                            status="failed",
                            error=str(exc),
                        )

                    if result.status == "completed":
                        self._on_workstream_complete(ws_id, result)
                    else:
                        self._handle_failure(ws_id, result, executor, futures)

        # Clean up session-scoped memory tags.
        tag = f"sadd:{self._session_id}"
        if hasattr(self._brain, "forget_tagged"):
            try:
                self._brain.forget_tagged(tag)
            except Exception:
                self._logger.debug("Could not clean tagged memories for %s", tag)

        report = self._build_report(started_at)

        # Persist final report.
        if self._store:
            try:
                self._store.save_report(self._session_id, report)
            except Exception:
                self._logger.debug("Could not save session report")

        print(f"✅ SADD session complete — {report.completed}/{report.total_workstreams} workstreams succeeded")
        return report

    def resume(
        self,
        graph: WorkstreamGraph,
        completed_results: Dict[str, "WorkstreamResult"],
    ) -> SessionReport:
        """Resume an interrupted session from a pre-restored graph and results.

        Unlike ``run()``, this method does not rebuild the graph or create a new
        session record.  It restores the coordinator's internal state from the
        supplied graph and completed results, then drives only the remaining
        (non-completed) workstreams to completion.
        """
        started_at = time.time()

        self._graph = graph
        self._results = dict(completed_results)

        # Reset previously-failed and blocked workstreams to pending so they are
        # re-attempted. "failed" is not terminal from a resume perspective — the
        # user explicitly chose to retry. "blocked" workstreams were blocked because
        # their dependencies failed, so they must also be unblocked for retry.
        for node in graph.get_all_nodes():
            if node.status in ("failed", "blocked"):
                node.status = "pending"
                node.result = None

        remaining = [
            ws_id
            for ws_id, node in graph.iter_nodes()
            if node.status != "completed"
        ]
        self._logger.info(
            "SADD session %s resuming — %d workstreams, %d already completed, %d remaining",
            self._session_id,
            len(self._spec.workstreams),
            len(completed_results),
            len(remaining),
        )
        print(
            f"▶️  SADD session {self._session_id[:8]}... resuming — "
            f"{len(completed_results)} done, {len(remaining)} remaining"
        )

        # Mark session as running again if store is available (session already exists).
        if self._store:
            try:
                self._store.update_status(self._session_id, "running")
            except Exception:
                self._logger.debug("Could not update session status for resume")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_parallel,
        ) as executor:
            while not self._graph.is_complete():
                ready = self._graph.ready_workstreams()

                if not ready:
                    self._logger.warning(
                        "SADD session %s: no ready workstreams and graph not complete — possible deadlock, stopping",
                        self._session_id,
                    )
                    break

                futures: Dict[concurrent.futures.Future[WorkstreamResult], str] = {}
                for ws_id in ready:
                    with self._lock:
                        self._graph.mark_running(ws_id)
                    future = executor.submit(self._execute_workstream, ws_id)
                    futures[future] = ws_id

                for future in concurrent.futures.as_completed(futures):
                    ws_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = WorkstreamResult(
                            ws_id=ws_id,
                            status="failed",
                            error=str(exc),
                        )

                    if result.status == "completed":
                        self._on_workstream_complete(ws_id, result)
                    else:
                        self._handle_failure(ws_id, result, executor, futures)

        # Clean up session-scoped memory tags.
        tag = f"sadd:{self._session_id}"
        if hasattr(self._brain, "forget_tagged"):
            try:
                self._brain.forget_tagged(tag)
            except Exception:
                self._logger.debug("Could not clean tagged memories for %s", tag)

        report = self._build_report(started_at)

        if self._store:
            try:
                self._store.save_report(self._session_id, report)
            except Exception:
                self._logger.debug("Could not save session report")

        print(
            f"✅ SADD resume complete — {report.completed}/{report.total_workstreams} workstreams succeeded"
        )
        return report

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of current session state."""
        if self._graph is None:
            return {
                "session_id": self._session_id,
                "state": "not_started",
            }

        with self._lock:
            nodes = self._graph.to_dict()["nodes"]

        completed = sum(1 for n in nodes.values() if n["status"] == "completed")
        failed = sum(1 for n in nodes.values() if n["status"] == "failed")
        blocked = sum(1 for n in nodes.values() if n["status"] == "blocked")
        running = sum(1 for n in nodes.values() if n["status"] == "running")

        return {
            "session_id": self._session_id,
            "total": len(nodes),
            "completed": completed,
            "failed": failed,
            "blocked": blocked,
            "running": running,
        }

    # ------------------------------------------------------------------
    # Internal — workstream execution
    # ------------------------------------------------------------------

    def _execute_workstream(self, ws_id: str) -> WorkstreamResult:
        """Execute a single workstream via a SubAgentRunner."""
        assert self._graph is not None
        node = self._graph.get_node(ws_id)

        # Build context from completed dependency results.
        context_from_dependencies: Dict[str, WorkstreamResult] = {}
        for dep_id in node.spec.depends_on:
            if dep_id in self._results:
                context_from_dependencies[dep_id] = self._results[dep_id]

        runner = SubAgentRunner(
            workstream=node,
            orchestrator_factory=self._orchestrator_factory,
            brain=self._brain,
            context_from_dependencies=context_from_dependencies,
            mcp_bridge=self._mcp_bridge,
        )
        print(f"⏳ [{ws_id}] Starting: {node.spec.title}")

        return runner.run(
            max_cycles=self._config.max_cycles_per_workstream,
            dry_run=self._config.dry_run,
        )

    # ------------------------------------------------------------------
    # Internal — result handling
    # ------------------------------------------------------------------

    def _on_workstream_complete(self, ws_id: str, result: WorkstreamResult) -> None:
        """Record a successful workstream completion."""
        self._results[ws_id] = result
        with self._lock:
            assert self._graph is not None
            self._graph.mark_completed(ws_id, result)
        self._logger.info(
            "SADD workstream %s completed (%d cycles, %.1fs)",
            ws_id,
            result.cycles_used,
            result.elapsed_s,
        )
        print(f"✅ [{ws_id}] Done: {self._graph.get_node(ws_id).spec.title} ({result.elapsed_s:.1f}s)")
        self._checkpoint_and_log(ws_id, "workstream_completed")

    def _handle_failure(
        self,
        ws_id: str,
        result: WorkstreamResult,
        executor: concurrent.futures.ThreadPoolExecutor,
        futures: Dict[concurrent.futures.Future[WorkstreamResult], str],
    ) -> None:
        """Handle a workstream failure — retry once if configured, else record."""
        if self._config.retry_failed and ws_id not in self._retried:
            self._retried.add(ws_id)
            self._logger.warning(
                "SADD workstream %s failed, retrying once: %s",
                ws_id,
                result.error,
            )
            with self._lock:
                assert self._graph is not None
                # Reset to running for the retry attempt.
                node = self._graph.get_node(ws_id)
                node.status = "running"
                node.started_at = time.time()

            future = executor.submit(self._execute_workstream, ws_id)
            futures[future] = ws_id
            return

        # Final failure — record and propagate blocks.
        self._results[ws_id] = result
        with self._lock:
            assert self._graph is not None
            self._graph.mark_failed(ws_id, result.error or "unknown error")
        self._logger.error("SADD workstream %s failed permanently: %s", ws_id, result.error)
        print(f"❌ [{ws_id}] Failed: {result.error or 'unknown error'}")

        if self._config.fail_fast:
            # Cancel all pending futures that haven't started yet.
            for f in futures:
                f.cancel()
        self._checkpoint_and_log(ws_id, "workstream_failed")

    # ------------------------------------------------------------------
    # Internal — persistence helpers
    # ------------------------------------------------------------------

    def _checkpoint_and_log(self, ws_id: str, event_type: str) -> None:
        """Save a checkpoint and log an event if a store is configured."""
        if not self._store or self._graph is None:
            return
        try:
            graph_state = self._graph.to_dict()
            results_dict = {k: v.to_dict() for k, v in self._results.items()}
            self._store.save_checkpoint(self._session_id, graph_state, results_dict)
            result = self._results.get(ws_id)
            payload = {"status": result.status if result else "unknown"}
            if result and result.changed_files:
                payload["changed_files"] = result.changed_files
                for fp in result.changed_files:
                    self._store.record_artifact(self._session_id, ws_id, fp)
            self._store.log_event(self._session_id, ws_id, event_type, payload)
        except Exception:
            self._logger.debug("Checkpoint/log failed for %s", ws_id)

    # ------------------------------------------------------------------
    # Internal — report building
    # ------------------------------------------------------------------

    def _build_report(self, started_at: float) -> SessionReport:
        """Aggregate results into a SessionReport."""
        assert self._graph is not None
        elapsed = time.time() - started_at

        outcomes: List[WorkstreamOutcome] = []
        completed_count = 0
        failed_count = 0
        skipped_count = 0

        for ws in self._spec.workstreams:
            node = self._graph.get_node(ws.id)
            result = self._results.get(ws.id)

            status = node.status
            if status == "blocked":
                status = "skipped"

            if status == "completed":
                completed_count += 1
            elif status == "failed":
                failed_count += 1
            elif status in ("skipped", "blocked"):
                skipped_count += 1

            outcome = WorkstreamOutcome(
                id=ws.id,
                title=ws.title,
                status=status,
                cycles_used=result.cycles_used if result else 0,
                stop_reason=result.stop_reason if result else None,
                elapsed_s=result.elapsed_s if result else 0.0,
                artifacts=list(result.changed_files) if result else [],
            )
            outcomes.append(outcome)

        # Extract learnings from reflector outputs.
        learnings: List[str] = []
        for result in self._results.values():
            if result.reflector_output:
                learnings.append(result.reflector_output)

        return SessionReport(
            session_id=self._session_id,
            design_title=self._spec.title,
            total_workstreams=len(self._spec.workstreams),
            completed=completed_count,
            failed=failed_count,
            skipped=skipped_count,
            outcomes=outcomes,
            elapsed_s=elapsed,
            learnings=learnings,
            started_at=started_at,
        )
