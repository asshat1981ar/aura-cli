"""Phase dispatcher — thin delegation layer for orchestrator phase execution.

:class:`PhaseDispatcher` encapsulates the logic for mapping phase names to
agent callables, wrapping execution with lifecycle hooks, and collecting
phase outputs.  It is extracted from :class:`~core.orchestrator.LoopOrchestrator`
to make the dispatch concern independently testable and replaceable.

Typical usage::

    dispatcher = PhaseDispatcher(
        agents={"plan": planner, "act": coder, ...},
        hook_engine=HookEngine(config),
        config={"force_legacy_orchestrator": False},
    )
    result = dispatcher.dispatch("plan", {"goal": "Add retry logic"})
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from core.logging_utils import log_json


class PhaseDispatcher:
    """Maps phase names to agent callables and executes them with hook wrapping.

    This is a thin delegation layer extracted from
    :meth:`~core.orchestrator.LoopOrchestrator._run_phase`.  It does **not**
    own retry logic, failure routing, or cycle state — those remain in
    :class:`~core.orchestrator.LoopOrchestrator`.

    Args:
        agents: Dict mapping phase name strings to agent instances.  Each
            agent must expose a ``run(input_data: dict) -> dict`` method.
        hook_engine: A :class:`~core.hooks.HookEngine` instance used for
            pre/post phase hooks.  When ``None`` hooks are skipped entirely.
        config: Runtime config dict (e.g. loaded from ``aura.config.json``).
            Supports the ``"force_legacy_orchestrator"`` and
            ``"enable_new_orchestrator"`` flags.
        project_root: Optional string path of the project root, forwarded to
            async canary routing when ``enable_new_orchestrator`` is set.
    """

    #: Phases routed to the async canary path when ``enable_new_orchestrator``
    #: is enabled in config (mirrors the M5-002/M5-003 canary wave logic).
    CANARY_PHASES = frozenset(["mcp_discovery", "mcp_health", "code_search", "investigation"])

    def __init__(
        self,
        agents: Dict[str, Any],
        hook_engine: Any = None,
        config: Optional[Dict[str, Any]] = None,
        project_root: str = ".",
    ) -> None:
        self.agents = agents
        self.hook_engine = hook_engine
        self._config = config or {}
        self.project_root = project_root

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dispatch(self, phase_name: str, input_data: Dict) -> Dict:
        """Execute *phase_name* with the registered agent and lifecycle hooks.

        This method mirrors the behaviour of
        :meth:`~core.orchestrator.LoopOrchestrator._run_phase` exactly so
        that the orchestrator can delegate to it transparently.

        Execution flow:

        1. If ``force_legacy_orchestrator`` is set in config, bypass hooks and
           call the agent directly (M5 rollback path).
        2. Run pre-phase hooks via :attr:`hook_engine`.  If a hook blocks the
           phase, return ``{"_blocked_by_hook": True, "phase": phase_name}``.
        3. If ``enable_new_orchestrator`` is set and the phase is in
           :attr:`CANARY_PHASES`, attempt the async canary path.
        4. Run the registered agent synchronously.
        5. Run post-phase hooks (observational, never blocking).
        6. Return the agent output (or ``{}`` when no agent is registered).

        Args:
            phase_name: The phase identifier (e.g. ``"plan"``, ``"act"``,
                ``"verify"``).
            input_data: Arbitrary dict passed directly to ``agent.run()``.

        Returns:
            The dict returned by ``agent.run()``, an empty dict ``{}`` when no
            agent is registered for *phase_name*, or a hook-block sentinel dict.
        """
        # Emergency bypass (M5 rollback)
        if self._config.get("force_legacy_orchestrator"):
            agent = self.agents.get(phase_name)
            return agent.run(input_data) if agent else {}

        # Pre-phase hooks (guaranteed execution — cannot be bypassed by model)
        if self.hook_engine is not None:
            should_proceed, input_data = self.hook_engine.run_pre_hooks(phase_name, input_data)
            if not should_proceed:
                log_json(
                    "WARN",
                    "phase_blocked_by_hook",
                    details={"phase": phase_name},
                )
                return {"_blocked_by_hook": True, "phase": phase_name}

        # Canary wave routing (M5-002, M5-003)
        if phase_name in self.CANARY_PHASES and self._config.get("enable_new_orchestrator"):
            canary_result = self._try_canary(phase_name, input_data)
            if canary_result is not None:
                return canary_result

        agent = self.agents.get(phase_name)
        if not agent:
            return {}

        result = agent.run(input_data)

        # Post-phase hooks (observational)
        if self.hook_engine is not None:
            self.hook_engine.run_post_hooks(
                phase_name,
                result if isinstance(result, dict) else {},
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_canary(self, phase_name: str, input_data: Dict) -> Optional[Dict]:
        """Attempt async canary routing for *phase_name*.

        Returns the async result dict on success, or ``None`` to fall through
        to the synchronous path.
        """
        try:
            import anyio
            import asyncio
            from core.types import TaskRequest, ExecutionContext

            req = TaskRequest(
                task_id=f"canary_{uuid.uuid4().hex[:8]}",
                agent_name=phase_name,
                input_data=input_data,
                context=ExecutionContext(project_root=str(self.project_root)),
            )

            async def _call() -> Any:
                return await self._dispatch_async(req)

            log_json(
                "INFO",
                "phase_dispatcher_canary_routing",
                details={"phase": phase_name},
            )
            try:
                asyncio.get_running_loop()
                task_res = anyio.from_thread.run(_call)
            except RuntimeError:
                task_res = anyio.run(_call)

            if task_res.status == "success":
                return task_res.output

            log_json(
                "ERROR",
                "phase_dispatcher_canary_failed",
                details={"phase": phase_name, "error": task_res.error},
            )
        except Exception as exc:
            log_json(
                "ERROR",
                "phase_dispatcher_canary_exception",
                details={"phase": phase_name, "error": str(exc)},
            )
        return None

    async def _dispatch_async(self, request: Any) -> Any:
        """Thin async wrapper that falls back to the local agent dict."""
        import anyio
        from core.types import TaskResult

        agent = self.agents.get(request.agent_name)
        if not agent:
            return TaskResult(
                task_id=request.task_id,
                status="error",
                output={},
                error=f"Agent {request.agent_name} not found",
            )
        try:
            result = await anyio.to_thread.run_sync(agent.run, request.input_data)
            return TaskResult(task_id=request.task_id, status="success", output=result)
        except Exception as exc:
            return TaskResult(
                task_id=request.task_id,
                status="error",
                output={},
                error=str(exc),
            )
