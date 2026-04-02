"""FastAPI server for AURA Web UI and API endpoints.

Provides REST API and WebSocket endpoints for the React dashboard.
"""

from __future__ import annotations

import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.logging_utils import log_json
from core.goal_queue import GoalQueue
from core.goal_archive import GoalArchive

try:
    from core.auth import init_auth, get_auth_manager, UserRole
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# In-memory stores (replace with proper database in production)
goals_store: List[Dict[str, Any]] = []
agents_store: List[Dict[str, Any]] = [
    {
        "id": "planner-1",
        "name": "Planner Agent",
        "type": "planner",
        "status": "idle",
        "capabilities": ["planning", "analysis"],
        "last_seen": datetime.utcnow().isoformat(),
        "stats": {"tasks_completed": 42, "tasks_failed": 3, "avg_execution_time": 5.2},
    },
    {
        "id": "coder-1",
        "name": "Coder Agent",
        "type": "coder",
        "status": "busy",
        "current_task": "Implementing feature X",
        "capabilities": ["coding", "refactoring", "testing"],
        "last_seen": datetime.utcnow().isoformat(),
        "stats": {"tasks_completed": 128, "tasks_failed": 7, "avg_execution_time": 12.5},
    },
    {
        "id": "critic-1",
        "name": "Critic Agent",
        "type": "critic",
        "status": "idle",
        "capabilities": ["review", "critique", "validation"],
        "last_seen": datetime.utcnow().isoformat(),
        "stats": {"tasks_completed": 89, "tasks_failed": 1, "avg_execution_time": 3.1},
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    log_json("INFO", "api_server_starting")
    if AUTH_AVAILABLE:
        init_auth(secret_key="aura-secret-key-change-in-production")
        auth = get_auth_manager()
        # Create default admin user
        try:
            auth.create_user("admin", password="admin", role=UserRole.ADMIN)
        except Exception:
            pass  # User may already exist
    yield
    # Shutdown
    log_json("INFO", "api_server_shutting_down")

app = FastAPI(
    title="AURA API",
    description="API for AURA CLI Web Dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user."""
    if not AUTH_AVAILABLE:
        return {"username": "anonymous", "role": "admin"}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        auth = get_auth_manager()
        user = auth.get_current_user(credentials.credentials)
        return user.to_dict()
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Auth endpoints
@app.post("/api/auth/login")
async def login(credentials: dict):
    """Authenticate user and return JWT token."""
    if not AUTH_AVAILABLE:
        return {
            "access_token": "mock-token",
            "user": {"username": "admin", "role": "admin"},
        }
    
    auth = get_auth_manager()
    user = auth.authenticate_user(credentials.get("username"), credentials.get("password"))
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = auth.create_access_token(user)
    return {
        "access_token": token,
        "user": user.to_dict(),
    }

# Goals endpoints
@app.get("/api/goals")
async def get_goals(user: dict = Depends(get_current_user)):
    """Get all goals."""
    return goals_store

@app.post("/api/goals")
async def create_goal(goal: dict, user: dict = Depends(get_current_user)):
    """Create a new goal."""
    new_goal = {
        "id": f"goal-{len(goals_store) + 1}",
        "description": goal.get("description", ""),
        "status": "pending",
        "priority": goal.get("priority", 1),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "cycles": 0,
        "max_cycles": 10,
    }
    goals_store.insert(0, new_goal)
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "goal_created",
        "payload": new_goal,
    })
    
    return new_goal

@app.post("/api/goals/{goal_id}/cancel")
async def cancel_goal(goal_id: str, user: dict = Depends(get_current_user)):
    """Cancel a running goal."""
    for goal in goals_store:
        if goal["id"] == goal_id:
            goal["status"] = "failed"
            goal["updated_at"] = datetime.utcnow().isoformat()
            
            await manager.broadcast({
                "type": "goal_updated",
                "payload": goal,
            })
            
            return {"success": True}
    
    raise HTTPException(status_code=404, detail="Goal not found")

# Agents endpoints
@app.get("/api/agents")
async def get_agents(user: dict = Depends(get_current_user)):
    """Get all agents."""
    return agents_store

@app.get("/api/agents/{agent_id}/logs")
async def get_agent_logs(agent_id: str, user: dict = Depends(get_current_user)):
    """Get logs for a specific agent."""
    return []  # Placeholder

# Stats endpoint
@app.get("/api/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    """Get system statistics."""
    pending = len([g for g in goals_store if g["status"] == "pending"])
    running = len([g for g in goals_store if g["status"] == "running"])
    completed = len([g for g in goals_store if g["status"] == "completed"])
    failed = len([g for g in goals_store if g["status"] == "failed"])
    
    return {
        "goals": {
            "total": len(goals_store),
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
        },
        "agents": {
            "total": len(agents_store),
            "active": len([a for a in agents_store if a["status"] == "busy"]),
            "idle": len([a for a in agents_store if a["status"] == "idle"]),
        },
        "system": {
            "uptime_seconds": 3600,
            "memory_usage_mb": 256,
            "cpu_percent": 15.5,
        },
    }

# WebSocket endpoint for real-time logs
@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming."""
    await manager.connect(websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to log stream",
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Handle ping/keepalive
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log_json("ERROR", "websocket_error", {"error": str(e)})
        manager.disconnect(websocket)

# Serve static files (production build)
try:
    app.mount("/static", StaticFiles(directory="web-ui/dist/assets"), name="static")
    
    @app.get("/", response_class=FileResponse)
    async def serve_index():
        return "web-ui/dist/index.html"
        
    @app.get("/{path:path}", response_class=FileResponse)
    async def serve_spa(path: str):
        if path.startswith("api/") or path.startswith("ws/"):
            raise HTTPException(status_code=404)
        return "web-ui/dist/index.html"
except RuntimeError:
    pass  # dist folder doesn't exist in development

# Function to broadcast log messages
def broadcast_log(log_entry: dict):
    """Broadcast a log entry to all connected WebSocket clients."""
    asyncio.create_task(manager.broadcast({
        "type": "log",
        "payload": log_entry,
    }))
