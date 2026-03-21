"""A2A protocol server for AURA — serves Agent Card and handles task requests.

Integrates with AURA's existing FastAPI infrastructure to add A2A endpoints:
- GET  /.well-known/agent.json  — Agent Card discovery
- POST /a2a/tasks               — Create and execute a task
- GET  /a2a/tasks/{id}          — Get task status
- POST /a2a/tasks/{id}/cancel   — Cancel a task
"""
from typing import Callable

from core.a2a.agent_card import AgentCard
from core.a2a.task import A2ATask, TaskState
from core.logging_utils import log_json


class A2AServer:
    """Lightweight A2A server that integrates with AURA's existing FastAPI."""

    def __init__(self, agent_card: AgentCard | None = None):
        self.agent_card = agent_card or AgentCard.default()
        self.tasks: dict[str, A2ATask] = {}
        self.handlers: dict[str, Callable] = {}

    def register_handler(self, capability: str, handler: Callable):
        """Register a handler for a capability."""
        self.handlers[capability] = handler
        log_json("INFO", "a2a_handler_registered",
                 details={"capability": capability})

    def get_agent_card(self) -> dict:
        """GET /.well-known/agent.json"""
        return self.agent_card.to_dict()

    async def create_task(self, capability: str, message: str,
                          metadata: dict | None = None) -> A2ATask:
        """Create and execute a task."""
        task = A2ATask(capability=capability, metadata=metadata or {})
        task.add_message("user", message)
        self.tasks[task.id] = task

        handler = self.handlers.get(capability)
        if handler:
            task.transition(TaskState.WORKING)
            try:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(task)
                else:
                    result = handler(task)
                result = result or {}
                task.add_message("agent", str(result.get("summary", "Done")))
                if "artifacts" in result:
                    for artifact in result["artifacts"]:
                        task.add_artifact(**artifact)
                task.transition(TaskState.COMPLETED)
            except Exception as exc:
                task.add_message("agent", f"Failed: {exc}")
                task.transition(TaskState.FAILED)
                log_json("WARN", "a2a_task_failed",
                         details={"task_id": task.id, "error": str(exc)})
        else:
            task.add_message("agent", f"Unknown capability: {capability}")
            task.transition(TaskState.FAILED)

        return task

    def get_task(self, task_id: str) -> A2ATask | None:
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if task and task.state in (
            TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED
        ):
            task.transition(TaskState.CANCELED)
            return True
        return False

    def get_fastapi_router(self, dependencies: list | None = None):
        """Return an APIRouter with A2A routes and optional dependencies (e.g. auth).

        Args:
            dependencies: List of FastAPI ``Depends(...)`` objects applied to all
                routes in the router (e.g. ``[Depends(require_auth)]``).

        Returns:
            APIRouter with A2A routes mounted.
        """
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse

        router = APIRouter(dependencies=dependencies or [])

        @router.get("/.well-known/agent.json")
        async def agent_card():
            return JSONResponse(self.get_agent_card())

        @router.post("/a2a/tasks")
        async def create_task(body: dict):
            task = await self.create_task(
                capability=body.get("capability", ""),
                message=body.get("message", ""),
                metadata=body.get("metadata"),
            )
            return JSONResponse(task.to_dict())

        @router.get("/a2a/tasks/{task_id}")
        async def get_task(task_id: str):
            task = self.get_task(task_id)
            if not task:
                return JSONResponse({"error": "not found"}, status_code=404)
            return JSONResponse(task.to_dict())

        @router.post("/a2a/tasks/{task_id}/cancel")
        async def cancel_task(task_id: str):
            if self.cancel_task(task_id):
                return JSONResponse({"status": "canceled"})
            return JSONResponse({"error": "cannot cancel"}, status_code=400)

        log_json("INFO", "a2a_routes_registered")
        return router

    def register_fastapi_routes(self, app):
        """Register A2A routes on an existing FastAPI app (no auth).

        Prefer :meth:`get_fastapi_router` with ``dependencies=[Depends(require_auth)]``
        to protect these endpoints behind bearer-token authentication.
        """
        router = self.get_fastapi_router()
        app.include_router(router)
