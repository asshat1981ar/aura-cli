"""Pipeline run and webhook endpoints for AURA API.

Extracted from server.py as part of Sprint 1 server decomposition.
Provides /run endpoint and /webhook/* endpoints for pipeline execution.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from aura_cli.api.middleware.auth import require_auth
from core.logging_utils import log_json
from core.running_runs import deregister_run, register_run

try:
    from prometheus_client import Counter, Gauge
    _PROMETHEUS_AVAILABLE = True
    pipeline_runs_total = Counter("aura_pipeline_runs", "Total pipeline runs", ["status"])
    active_pipeline_runs = Gauge("aura_active_pipeline_runs", "Currently executing pipelines")
    goal_queue_depth = Gauge("aura_goal_queue_depth", "Number of goals waiting in the webhook queue")
except ImportError:
    _PROMETHEUS_AVAILABLE = False

router = APIRouter(tags=["runs"])

# In-memory goal queue for webhook-submitted goals
_webhook_goal_queue: Dict[str, Dict[str, Any]] = {}


class RunRequest(BaseModel):
    """Request body for POST /run."""

    goal: str
    max_cycles: int = 1
    dry_run: bool = False


class WebhookGoalRequest(BaseModel):
    """Request body for POST /webhook/goal."""

    goal: str
    priority: int = 5
    dry_run: bool = False
    # metadata carries pipeline-routing context from n8n:
    # - complexity: "low" | "medium" | "high"
    # - pipeline_run_id: str (trace ID from P1/P3)
    # - quality_gate_critique: str (injected by P3 before AURA act phase)
    metadata: Dict[str, Any] = {}


class WebhookPlanReviewRequest(BaseModel):
    """Request body for POST /webhook/plan-review."""

    task_bundle: Dict[str, Any]
    goal: str
    pipeline_run_id: str = ""


async def _resolve_runtime_component(name: str) -> Any:
    """Resolve a runtime component (orchestrator, etc.)."""
    # Import here to avoid circular dependencies
    from aura_cli.server import _resolve_runtime_component as _resolve
    return await _resolve(name)


@router.post("/run")
async def run_pipeline(
    req: RunRequest, _: Dict[str, Any] = Depends(require_auth)
) -> Dict[str, Any]:
    """Trigger a goal-oriented pipeline run via LoopOrchestrator.

    Requires the ``AGENT_API_ENABLE_RUN=1`` environment variable to be set.
    Returns a ``run_id`` that can be used to track or cancel the run.

    Args:
        req: RunRequest with goal, max_cycles, and dry_run options.

    Returns:
        Dict with run_id and status.

    Raises:
        HTTPException: 403 if run endpoint is disabled.
    """
    if os.getenv("AGENT_API_ENABLE_RUN") != "1":
        raise HTTPException(
            status_code=403, detail="Run endpoint disabled; set AGENT_API_ENABLE_RUN=1"
        )

    run_id = secrets.token_hex(12)

    async def _background_run() -> None:
        register_run(run_id)
        try:
            active_orchestrator = await _resolve_runtime_component("orchestrator")
            await asyncio.to_thread(
                active_orchestrator.run_loop,
                req.goal,
                req.max_cycles,
                req.dry_run,
            )
        finally:
            deregister_run(run_id)

    asyncio.create_task(_background_run())
    return {"run_id": run_id, "status": "accepted"}


# ---------------------------------------------------------------------------
# n8n pipeline integration webhooks
# ---------------------------------------------------------------------------


@router.post("/webhook/goal")
async def webhook_goal(
    req: WebhookGoalRequest, _: Dict[str, Any] = Depends(require_auth)
) -> Dict[str, Any]:
    """Enqueue a goal submitted by n8n (P1 fast lane or direct trigger).

    Args:
        req: WebhookGoalRequest with goal, priority, and metadata.

    Returns:
        Dict with goal_id and status.
    """
    goal_id = secrets.token_hex(12)
    entry: Dict[str, Any] = {
        "goal_id": goal_id,
        "goal": req.goal,
        "priority": req.priority,
        "dry_run": req.dry_run,
        "metadata": req.metadata,
        "status": "queued",
        "queued_at": time.time(),
    }
    _webhook_goal_queue[goal_id] = entry
    if _PROMETHEUS_AVAILABLE:
        goal_queue_depth.set(
            sum(1 for e in _webhook_goal_queue.values() if e["status"] == "queued")
        )
    log_json(
        "INFO",
        "aura_webhook_goal_received",
        details={
            "goal_id": goal_id,
            "goal": req.goal[:200],
            "pipeline_run_id": req.metadata.get("pipeline_run_id", ""),
            "complexity": req.metadata.get("complexity", "unknown"),
        },
    )

    # Fire-and-forget: run the cycle in background so n8n can poll /webhook/status
    async def _run_cycle() -> None:
        if _PROMETHEUS_AVAILABLE:
            active_pipeline_runs.inc()
            goal_queue_depth.set(
                sum(1 for e in _webhook_goal_queue.values() if e["status"] == "queued")
            )
        try:
            _webhook_goal_queue[goal_id]["status"] = "running"
            _webhook_goal_queue[goal_id]["started_at"] = time.time()
            active_orchestrator = await _resolve_runtime_component("orchestrator")
            result = await asyncio.to_thread(
                active_orchestrator.run_cycle,
                req.goal,
            )
            _webhook_goal_queue[goal_id].update({
                "status": "done",
                "result": result,
                "completed_at": time.time(),
            })
            if _PROMETHEUS_AVAILABLE:
                pipeline_runs_total.labels(status="success").inc()
        except Exception as exc:
            _webhook_goal_queue[goal_id].update({
                "status": "failed",
                "error": str(exc),
                "completed_at": time.time(),
            })
            log_json(
                "WARN", "aura_webhook_goal_failed", details={"goal_id": goal_id, "error": str(exc)}
            )
            if _PROMETHEUS_AVAILABLE:
                pipeline_runs_total.labels(status="failure").inc()
        finally:
            if _PROMETHEUS_AVAILABLE:
                active_pipeline_runs.dec()

    asyncio.create_task(_run_cycle())
    return {"status": "queued", "goal_id": goal_id}


@router.get("/webhook/status/{goal_id}")
async def webhook_goal_status(
    goal_id: str, _: Dict[str, Any] = Depends(require_auth)
) -> Dict[str, Any]:
    """Poll the status of a webhook-submitted goal (used by n8n P1 fast lane).

    Args:
        goal_id: The goal ID returned by /webhook/goal.

    Returns:
        Dict with goal status, result (if done), or error (if failed).

    Raises:
        HTTPException: 404 if goal not found.
    """
    entry = _webhook_goal_queue.get(goal_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found")
    result: Dict[str, Any] = {
        "goal_id": goal_id,
        "status": entry["status"],
        "goal": entry["goal"],
        "queued_at": entry["queued_at"],
    }
    if entry["status"] == "done":
        result["result"] = entry.get("result", {})
        result["completed_at"] = entry.get("completed_at")
    elif entry["status"] == "failed":
        result["error"] = entry.get("error", "unknown error")
        result["completed_at"] = entry.get("completed_at")
    elif entry["status"] == "running":
        result["started_at"] = entry.get("started_at")
    return result


@router.post("/webhook/plan-review")
async def webhook_plan_review(
    req: WebhookPlanReviewRequest, _: Dict[str, Any] = Depends(require_auth)
) -> Dict[str, Any]:
    """Format a task bundle as a plan text for Dev Suite Quality Gate (P2) review.

    Called by P3 Pipeline Coordinator before triggering AURA act phase.
    Returns human-readable plan text that Dev Suite agents can critique.

    Args:
        req: WebhookPlanReviewRequest with task_bundle and goal.

    Returns:
        Dict with review_payload containing formatted plan text.
    """
    plan_steps = req.task_bundle.get("plan", [])
    if isinstance(plan_steps, str):
        plan_text = plan_steps
    elif isinstance(plan_steps, list):
        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))
    else:
        plan_text = str(plan_steps)

    review_payload = {
        "goal": req.goal,
        "pipeline_run_id": req.pipeline_run_id,
        "plan_text": plan_text,
        "task_bundle_keys": list(req.task_bundle.keys()),
        "file_targets": req.task_bundle.get("file_targets", []),
        "critique": req.task_bundle.get("critique", ""),
    }
    log_json(
        "INFO",
        "aura_webhook_plan_review_requested",
        details={
            "goal": req.goal[:200],
            "pipeline_run_id": req.pipeline_run_id,
            "plan_steps": len(plan_steps) if isinstance(plan_steps, list) else 1,
        },
    )
    return {"status": "ok", "review_payload": review_payload}
