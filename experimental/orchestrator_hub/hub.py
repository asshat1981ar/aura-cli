"""MCP Orchestrator Hub — central control plane for multi-environment coordination.

Endpoints:
  GET  /health                → hub health + registered server/agent counts
  GET  /servers               → list all registered MCP servers
  POST /servers/{name}/start  → start a managed server
  POST /servers/{name}/stop   → stop a managed server
  GET  /servers/{name}/health → check individual server health
  GET  /agents                → list all registered agents with capabilities
  POST /agents/register       → register an agent
  POST /agents/{name}/heartbeat → update agent heartbeat
  POST /dispatch              → route a task to the best agent
  POST /broadcast             → publish message to all agents
  GET  /environments          → list active environments
  GET  /messages              → recent message bus messages
  GET  /metrics               → Prometheus-compatible metrics

Start:
  uvicorn orchestrator_hub.hub:app --port 8010

Auth (optional):
  Set HUB_TOKEN env var
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from core.logging_utils import log_json
from orchestrator_hub.registry import AgentRegistryHub
from orchestrator_hub.lifecycle import ServerLifecycle
from orchestrator_hub.router import TaskRouter
from orchestrator_hub.message_bus import MessageBus, TOPIC_AGENT_REQUEST

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AURA MCP Orchestrator Hub",
    description="Central control plane for multi-environment MCP coordination.",
    version="1.0.0",
)

_TOKEN = os.getenv("HUB_TOKEN", "")
_SERVER_START = time.time()

# Core components (lazily initialized on first use)
_registry: Optional[AgentRegistryHub] = None
_lifecycle: Optional[ServerLifecycle] = None
_router: Optional[TaskRouter] = None
_bus: Optional[MessageBus] = None

# Metrics counters
_request_counts: Dict[str, int] = {}
_request_errors: Dict[str, int] = {}


def _get_registry() -> AgentRegistryHub:
    global _registry
    if _registry is None:
        _registry = AgentRegistryHub()
        _register_default_servers()
    return _registry


def _get_lifecycle() -> ServerLifecycle:
    global _lifecycle
    if _lifecycle is None:
        _lifecycle = ServerLifecycle(project_root=_ROOT)
    return _lifecycle


def _get_router() -> TaskRouter:
    global _router
    if _router is None:
        _router = TaskRouter(_get_registry())
    return _router


def _get_bus() -> MessageBus:
    global _bus
    if _bus is None:
        _bus = MessageBus()
    return _bus


def _register_default_servers() -> None:
    """Register the known AURA MCP servers in the registry."""
    registry = _get_registry()
    default_servers = {
        "dev_tools": 8001,
        "skills": 8002,
        "control": 8003,
        "thinking": 8004,
        "agentic_loop": 8006,
        "copilot": 8007,
        "hub": 8010,
        "docker": 8011,
        "kubernetes": 8012,
        "neo4j": 8013,
        "redis": 8014,
        "notification": 8015,
        "monitoring": 8016,
        "weaviate": 8017,
    }
    for name, port in default_servers.items():
        registry.register_server(name, port, server_type="internal_http")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _TOKEN:
        return
    if authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _track(endpoint: str) -> None:
    _request_counts[endpoint] = _request_counts.get(endpoint, 0) + 1


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class AgentRegistration(BaseModel):
    name: str
    agent_type: str
    capabilities: List[str]
    endpoint: str
    environment: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskDispatch(BaseModel):
    goal: str = ""
    task: str = ""
    environment: Optional[str] = None
    exclude_agents: List[str] = Field(default_factory=list)


class BroadcastMessage(BaseModel):
    topic: str
    payload: Dict[str, Any]
    sender: str = "hub"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    _track("health")
    registry = _get_registry()
    return {
        "status": "ok",
        "server": "orchestrator_hub",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _SERVER_START, 1),
        "agent_count": len(registry.list_agents()),
        "server_count": len(registry.list_servers()),
    }


# -- Server management --


@app.get("/servers")
async def list_servers(_: None = Depends(_check_auth)):
    _track("servers_list")
    registry = _get_registry()
    return {"servers": [s.as_dict() for s in registry.list_servers()]}


@app.post("/servers/{name}/start")
async def start_server(name: str, _: None = Depends(_check_auth)):
    _track("server_start")
    lifecycle = _get_lifecycle()
    state = lifecycle.get_server(name)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not managed by lifecycle.")
    try:
        result = lifecycle.start(name)
        log_json("INFO", "hub_server_start", details=result)
        return result
    except Exception as exc:
        log_json("ERROR", "hub_server_start_failed", details={"server": name, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Server start failed") from None


@app.post("/servers/{name}/stop")
async def stop_server(name: str, _: None = Depends(_check_auth)):
    _track("server_stop")
    lifecycle = _get_lifecycle()
    try:
        result = lifecycle.stop(name)
        log_json("INFO", "hub_server_stop", details=result)
        return result
    except Exception as exc:
        log_json("ERROR", "hub_server_stop_failed", details={"server": name, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Server stop failed") from None


@app.get("/servers/{name}/health")
async def server_health(name: str, _: None = Depends(_check_auth)):
    _track("server_health")
    lifecycle = _get_lifecycle()
    state = lifecycle.get_server(name)
    if state is None:
        # Fall back to registry check
        registry = _get_registry()
        server = registry.get_server(name)
        if server is None:
            raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")
        return {"name": name, "status": server.status, "port": server.port}
    return lifecycle.health_check(name)


# -- Agent management --


@app.get("/agents")
async def list_agents(_: None = Depends(_check_auth)):
    _track("agents_list")
    registry = _get_registry()
    return {"agents": [a.as_dict() for a in registry.list_agents()]}


@app.post("/agents/register")
async def register_agent(reg: AgentRegistration, _: None = Depends(_check_auth)):
    _track("agent_register")
    registry = _get_registry()
    info = registry.register_agent(
        name=reg.name,
        agent_type=reg.agent_type,
        capabilities=reg.capabilities,
        endpoint=reg.endpoint,
        environment=reg.environment,
        metadata=reg.metadata,
    )
    log_json("INFO", "hub_agent_registered", details={"name": reg.name, "type": reg.agent_type})
    return info.as_dict()


@app.post("/agents/{name}/heartbeat")
async def agent_heartbeat(name: str, _: None = Depends(_check_auth)):
    _track("agent_heartbeat")
    registry = _get_registry()
    if not registry.heartbeat(name):
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")
    return {"name": name, "status": "ok"}


# -- Task dispatch --


@app.post("/dispatch")
async def dispatch_task(task: TaskDispatch, _: None = Depends(_check_auth)):
    _track("dispatch")
    router = _get_router()
    task_dict = {"goal": task.goal, "task": task.task}

    if task.exclude_agents:
        agent_name = router.route_with_fallback(task_dict, exclude=task.exclude_agents)
    else:
        agent_name = router.route(task_dict)

    if agent_name is None:
        return {"status": "no_agent_found", "task": task_dict}

    # Publish dispatch event
    bus = _get_bus()
    bus.publish(
        TOPIC_AGENT_REQUEST,
        {
            "agent": agent_name,
            "goal": task.goal or task.task,
            "environment": task.environment,
        },
        sender="hub",
    )

    registry = _get_registry()
    agent = registry.get_agent(agent_name)

    log_json("INFO", "hub_task_dispatched", details={"agent": agent_name, "goal": task.goal or task.task})
    return {
        "status": "dispatched",
        "agent": agent_name,
        "agent_endpoint": agent.endpoint if agent else None,
        "environment": agent.environment if agent else None,
    }


# -- Broadcast --


@app.post("/broadcast")
async def broadcast(msg: BroadcastMessage, _: None = Depends(_check_auth)):
    _track("broadcast")
    bus = _get_bus()
    message = bus.publish(msg.topic, msg.payload, sender=msg.sender)
    return {"status": "published", "message_id": message.message_id, "topic": msg.topic}


# -- Messages --


@app.get("/messages")
async def get_messages(
    topic: Optional[str] = None,
    limit: int = 50,
    _: None = Depends(_check_auth),
):
    _track("messages")
    bus = _get_bus()
    return {"messages": bus.recent_messages(topic, limit)}


# -- Environments --


@app.get("/environments")
async def list_environments(_: None = Depends(_check_auth)):
    _track("environments")
    try:
        from environments.manager import EnvironmentManager

        mgr = EnvironmentManager(project_root=_ROOT)
        envs = mgr.list_environments()
        return {"environments": [e.as_dict() for e in envs]}
    except Exception as exc:
        log_json("ERROR", "hub_list_environments_failed", details={"error": str(exc)})
        return {"environments": [], "error": "Failed to list environments"}


# -- Metrics --


@app.get("/metrics")
async def get_metrics(_: None = Depends(_check_auth)):
    _track("metrics")
    uptime = round(time.time() - _SERVER_START, 1)
    total_requests = sum(_request_counts.values())
    total_errors = sum(_request_errors.values())

    registry = _get_registry()

    return {
        "uptime_seconds": uptime,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_requests, 1), 4),
        "agent_count": len(registry.list_agents()),
        "server_count": len(registry.list_servers()),
        "endpoints": _request_counts.copy(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("HUB_PORT", "8010"))
    uvicorn.run("orchestrator_hub.hub:app", host="0.0.0.0", port=port, reload=False)
