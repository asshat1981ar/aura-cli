"""WebSocket endpoints for AURA API.

Extracted from api_server.py as part of Sprint 1 server decomposition.
Provides /ws and /ws/logs endpoints for real-time updates.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """WebSocket connection manager for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        disconnected: List[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


# Global connection manager instance
manager = ConnectionManager()


def _get_goal_queue_data() -> Dict[str, Any]:
    """Read goal queue from the JSON file."""
    from pathlib import Path

    queue_path = Path("memory/goal_queue.json")
    if not queue_path.exists():
        return {"queue": [], "in_flight": {}}

    try:
        with open(queue_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return {"queue": raw, "in_flight": {}}
        return raw
    except (json.JSONDecodeError, IOError):
        return {"queue": [], "in_flight": {}}


def _get_goal_archive() -> List[Dict[str, Any]]:
    """Read completed goals from archive."""
    from pathlib import Path

    archive_path = Path("memory/goal_archive.jsonl")
    goals: List[Dict[str, Any]] = []
    if archive_path.exists():
        try:
            with open(archive_path) as f:
                for line in f:
                    if line.strip():
                        goals.append(json.loads(line))
        except (json.JSONDecodeError, IOError):
            pass
    return goals


def _get_agents_from_registry() -> List[Dict[str, Any]]:
    """Get agents from registry for WebSocket initial data."""
    try:
        from core.mcp_agent_registry import list_registered_services

        services = list_registered_services()
        return [{"name": s.get("name", "unknown"), "status": "ok"} for s in services]
    except Exception:
        return []


def _format_goals_for_api(queue_data: Dict[str, Any], archive: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format goals for API response."""
    goals: List[Dict[str, Any]] = []

    # Add queued goals
    if isinstance(queue_data, dict):
        for goal in queue_data.get("queue", []):
            if isinstance(goal, dict):
                goals.append(
                    {
                        "id": goal.get("id", "unknown"),
                        "description": goal.get("description", ""),
                        "status": "queued",
                        "created_at": goal.get("created_at", ""),
                    }
                )

        # Add in-flight goals
        for goal_id, goal in queue_data.get("in_flight", {}).items():
            if isinstance(goal, dict):
                goals.append(
                    {
                        "id": goal_id,
                        "description": goal.get("description", ""),
                        "status": "in_progress",
                        "created_at": goal.get("created_at", ""),
                    }
                )

    # Add archived goals
    for goal in archive[-50:]:  # Last 50 archived
        if isinstance(goal, dict):
            goals.append(
                {
                    "id": goal.get("id", "unknown"),
                    "description": goal.get("description", ""),
                    "status": goal.get("status", "completed"),
                    "created_at": goal.get("created_at", ""),
                    "completed_at": goal.get("completed_at", ""),
                }
            )

    # Sort by created_at descending
    goals.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return goals


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates.

    Sends initial data with goals and agents, then handles ping/pong.
    """
    await manager.connect(websocket)

    # Send initial data
    queue_data = _get_goal_queue_data()
    archive = _get_goal_archive()
    goals = _format_goals_for_api(queue_data, archive)
    agents = _get_agents_from_registry()

    await websocket.send_json(
        {
            "type": "initial",
            "payload": {
                "goals": goals[:20],  # Send last 20
                "agents": agents,
            },
        }
    )

    try:
        while True:
            # Wait for client messages (keepalive)
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time log streaming.

    Legacy endpoint for backward compatibility.
    """
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
