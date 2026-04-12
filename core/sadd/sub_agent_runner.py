"""SADD sub-agent runner — wraps LoopOrchestrator for single workstream execution."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from core.sadd.types import WorkstreamResult
from core.sadd.workstream_graph import WorkstreamNode


class SubAgentRunner:
    """Execute a single workstream through a LoopOrchestrator instance."""

    def __init__(
        self,
        workstream: WorkstreamNode,
        orchestrator_factory: Callable[[], Any],  # returns LoopOrchestrator
        brain: Any,  # Brain instance
        context_from_dependencies: dict[str, WorkstreamResult],
        mcp_bridge: Any = None,  # Optional MCPToolBridge instance
    ) -> None:
        self._workstream = workstream
        self._orchestrator_factory = orchestrator_factory
        self._brain = brain
        self._dep_context = context_from_dependencies
        self._mcp_bridge = mcp_bridge
        self._logger = logging.getLogger(__name__)

    def run(self, max_cycles: int = 5, dry_run: bool = False) -> WorkstreamResult:
        """Execute the workstream and return a normalized WorkstreamResult."""
        started_at = time.time()
        ws = self._workstream.spec

        try:
            # 1. Inject dependency context into brain as tagged memories
            self._inject_dependency_context()

            # 2. Build enriched goal with dependency context prefix
            enriched_goal = self._build_enriched_goal()

            # 3. Build context_injection dict for the orchestrator
            context_injection = self._build_context_injection()

            # 4. Create orchestrator and run
            orchestrator = self._orchestrator_factory()
            result = orchestrator.run_loop(
                enriched_goal,
                max_cycles=max_cycles,
                dry_run=dry_run,
                context_injection=context_injection,
            )

            # 5. Normalize to WorkstreamResult
            return self._normalize_result(result, started_at)

        except Exception as exc:
            self._logger.error("Workstream %s failed: %s", ws.id, exc)
            return WorkstreamResult(
                ws_id=ws.id,
                status="failed",
                error=str(exc),
                elapsed_s=time.time() - started_at,
            )

    # -- internal helpers ---------------------------------------------------

    def _inject_dependency_context(self) -> None:
        """Store dependency results in brain as tagged memories for retrieval."""
        ws = self._workstream.spec
        tag = f"sadd:{ws.id}"

        for dep_id, dep_result in self._dep_context.items():
            changed = ", ".join(dep_result.changed_files) if dep_result.changed_files else "none"
            summary_text = f"Dependency workstream {dep_id}: status={dep_result.status}, changed_files=[{changed}], verification={dep_result.verification_summary or 'n/a'}"
            self._brain.remember_tagged(summary_text, tag)

    def _build_enriched_goal(self) -> str:
        """Prepend dependency context to the workstream goal text."""
        ws = self._workstream.spec

        if not self._dep_context:
            return ws.goal_text

        context_lines: list[str] = ["[SADD Context from completed dependencies]"]
        for dep_id, dep_result in self._dep_context.items():
            changed = ", ".join(dep_result.changed_files) if dep_result.changed_files else "none"
            context_lines.append(f"- {dep_id}: {dep_result.status}, changed files: {changed}")

        context_lines.append("")
        context_lines.append("[Goal]")
        context_lines.append(ws.goal_text)

        return "\n".join(context_lines)

    def _build_context_injection(self) -> dict[str, Any]:
        """Build context dict to pass into orchestrator.run_loop."""
        serialized: dict[str, Any] = {}
        for dep_id, dep_result in self._dep_context.items():
            serialized[dep_id] = dep_result.to_dict()

        context: dict[str, Any] = {"sadd_dependencies": serialized}

        # Enrich with MCP tool matches when a bridge is available.
        if self._mcp_bridge is not None:
            ws = self._workstream.spec
            matched = self._mcp_bridge.match_tools_for_goal(ws.goal_text)
            if matched:
                context["sadd_mcp_tools"] = self._mcp_bridge.build_tool_context(matched)

        return context

    def _normalize_result(self, run_loop_result: Any, started_at: float) -> WorkstreamResult:
        """Map orchestrator run_loop output to a WorkstreamResult."""
        ws = self._workstream.spec

        # Extract stop_reason
        stop_reason = getattr(run_loop_result, "stop_reason", None)
        if isinstance(run_loop_result, dict):
            stop_reason = run_loop_result.get("stop_reason")

        # Extract history
        history = getattr(run_loop_result, "history", None)
        if isinstance(run_loop_result, dict):
            history = run_loop_result.get("history", [])
        if history is None:
            history = []

        # Determine status
        status: str = "completed" if stop_reason == "PASS" else "failed"

        # Extract changed_files from history phase_outputs
        changed_files: list[str] = []
        for cycle in history:
            phase_outputs = cycle.get("phase_outputs", {}) if isinstance(cycle, dict) else getattr(cycle, "phase_outputs", {})
            if phase_outputs:
                apply_output = phase_outputs.get("apply", {})
                if isinstance(apply_output, dict):
                    files = apply_output.get("changed_files", [])
                    changed_files.extend(files)

        # Extract verification_summary from last cycle
        verification_summary = ""
        if history:
            last_cycle = history[-1]
            phase_outputs = last_cycle.get("phase_outputs", {}) if isinstance(last_cycle, dict) else getattr(last_cycle, "phase_outputs", {})
            if phase_outputs:
                verify_output = phase_outputs.get("verify", {})
                if isinstance(verify_output, dict):
                    verification_summary = verify_output.get("summary", "")
                elif isinstance(verify_output, str):
                    verification_summary = verify_output

        # Extract reflector_output from last cycle
        reflector_output = ""
        if history:
            last_cycle = history[-1]
            phase_outputs = last_cycle.get("phase_outputs", {}) if isinstance(last_cycle, dict) else getattr(last_cycle, "phase_outputs", {})
            if phase_outputs:
                reflect_output = phase_outputs.get("reflect", {})
                if isinstance(reflect_output, dict):
                    reflector_output = reflect_output.get("summary", "")
                elif isinstance(reflect_output, str):
                    reflector_output = reflect_output

        return WorkstreamResult(
            ws_id=ws.id,
            status=status,
            cycles_used=len(history),
            stop_reason=stop_reason,
            changed_files=changed_files,
            verification_summary=verification_summary,
            reflector_output=reflector_output,
            elapsed_s=time.time() - started_at,
        )
