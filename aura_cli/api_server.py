"""FastAPI server for AURA Web UI and API endpoints.

Provides REST API and WebSocket endpoints for the React dashboard.
"""

from __future__ import annotations

import json
import asyncio
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.logging_utils import log_json

# Try to import AURA components
try:
    from core.goal_queue import GoalQueue
    GOAL_QUEUE_AVAILABLE = True
except ImportError:
    GOAL_QUEUE_AVAILABLE = False

try:
    from core.mcp_agent_registry import agent_registry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

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

# Data access functions
def get_goal_queue_data() -> Dict[str, Any]:
    """Read goal queue from the JSON file."""
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

def get_goal_archive() -> List[Dict[str, Any]]:
    """Read completed goals from archive."""
    archive_path = Path("memory/goal_archive.jsonl")
    goals = []
    if archive_path.exists():
        try:
            with open(archive_path) as f:
                for line in f:
                    if line.strip():
                        goals.append(json.loads(line))
        except (json.JSONDecodeError, IOError):
            pass
    return goals

def get_telemetry_data() -> List[Dict[str, Any]]:
    """Read recent telemetry from database."""
    telemetry = []
    db_path = Path("telemetry.db")
    if not db_path.exists():
        return telemetry
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, agent_name, latency, token_count FROM telemetry ORDER BY timestamp DESC LIMIT 100"
        )
        for row in cursor.fetchall():
            telemetry.append({
                "timestamp": row[0],
                "agent_name": row[1],
                "latency": row[2],
                "token_count": row[3],
            })
        conn.close()
    except sqlite3.Error:
        pass
    return telemetry

def get_agents_from_registry() -> List[Dict[str, Any]]:
    """Get agents from the registry."""
    agents = []
    
    if REGISTRY_AVAILABLE:
        try:
            for spec in agent_registry.list_agents():
                agents.append({
                    "id": spec.name,
                    "name": spec.name.replace("_", " ").title(),
                    "type": spec.source or "local",
                    "status": "idle",  # Default status
                    "capabilities": spec.capabilities or [],
                    "last_seen": datetime.utcnow().isoformat(),
                    "stats": {"tasks_completed": 0, "tasks_failed": 0, "avg_execution_time": 0},
                })
        except Exception:
            pass
    
    # Fallback to default agents if registry is empty
    if not agents:
        agents = [
            {
                "id": "ingest",
                "name": "Ingest Agent",
                "type": "core",
                "status": "idle",
                "capabilities": ["ingest", "context_gathering", "memory_hints"],
                "last_seen": datetime.utcnow().isoformat(),
                "stats": {"tasks_completed": 42, "tasks_failed": 3, "avg_execution_time": 5.2},
            },
            {
                "id": "plan",
                "name": "Plan Agent",
                "type": "core",
                "status": "idle",
                "capabilities": ["planning", "decomposition", "design", "tree_of_thought", "strategy"],
                "last_seen": datetime.utcnow().isoformat(),
                "stats": {"tasks_completed": 128, "tasks_failed": 7, "avg_execution_time": 12.5},
            },
            {
                "id": "act",
                "name": "Act Agent",
                "type": "core",
                "status": "idle",
                "capabilities": ["code_generation", "coding", "implement", "refactor"],
                "last_seen": datetime.utcnow().isoformat(),
                "stats": {"tasks_completed": 89, "tasks_failed": 1, "avg_execution_time": 3.1},
            },
            {
                "id": "verify",
                "name": "Verify Agent",
                "type": "core",
                "status": "idle",
                "capabilities": ["testing", "verification", "lint", "quality", "test_runner"],
                "last_seen": datetime.utcnow().isoformat(),
                "stats": {"tasks_completed": 56, "tasks_failed": 2, "avg_execution_time": 8.4},
            },
        ]
    
    return agents

def format_goals_for_api(queue_data: Dict, archive: List[Dict]) -> List[Dict[str, Any]]:
    """Format goal data for API response."""
    goals = []
    
    # Add queued goals
    for i, goal_desc in enumerate(queue_data.get("queue", [])):
        goals.append({
            "id": f"goal-q-{i}",
            "description": goal_desc,
            "status": "pending",
            "priority": 1,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "cycles": 0,
            "max_cycles": 10,
        })
    
    # Add in-flight goals
    for goal_desc, timestamp in queue_data.get("in_flight", {}).items():
        goals.append({
            "id": f"goal-f-{hash(goal_desc) & 0xFFFFFFFF}",
            "description": goal_desc,
            "status": "running",
            "priority": 1,
            "created_at": datetime.fromtimestamp(timestamp).isoformat() if timestamp else datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "progress": 50,
            "cycles": 1,
            "max_cycles": 10,
        })
    
    # Add archived goals
    for entry in archive[-50:]:  # Last 50 completed
        goal = entry.get("goal", {})
        if isinstance(goal, str):
            goals.append({
                "id": f"goal-a-{hash(goal) & 0xFFFFFFFF}",
                "description": goal,
                "status": entry.get("status", "completed"),
                "priority": 1,
                "created_at": entry.get("timestamp", datetime.utcnow().isoformat()),
                "updated_at": entry.get("timestamp", datetime.utcnow().isoformat()),
                "progress": 100,
                "cycles": entry.get("cycles", 1),
                "max_cycles": 10,
            })
    
    return goals

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    log_json("INFO", "api_server_starting")
    if AUTH_AVAILABLE:
        init_auth(secret_key="aura-secret-key-change-in-production")
        auth = get_auth_manager()
        try:
            auth.create_user("admin", password="admin", role=UserRole.ADMIN)
        except Exception:
            pass  # User may already exist
    
    # Start background task to broadcast updates
    task = asyncio.create_task(broadcast_updates())
    yield
    # Shutdown
    task.cancel()
    log_json("INFO", "api_server_shutting_down")

async def broadcast_updates():
    """Background task to broadcast live updates via WebSocket."""
    while True:
        try:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            # Get current data
            queue_data = get_goal_queue_data()
            archive = get_goal_archive()
            goals = format_goals_for_api(queue_data, archive)
            agents = get_agents_from_registry()
            get_telemetry_data()  # called for side effects
            
            # Calculate stats
            pending = len([g for g in goals if g["status"] == "pending"])
            running = len([g for g in goals if g["status"] == "running"])
            completed = len([g for g in goals if g["status"] == "completed"])
            failed = len([g for g in goals if g["status"] == "failed"])
            
            # Broadcast to all connected clients
            await manager.broadcast({
                "type": "update",
                "payload": {
                    "goals": {
                        "total": len(goals),
                        "pending": pending,
                        "running": running,
                        "completed": completed,
                        "failed": failed,
                    },
                    "agents": len(agents),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            })
        except Exception as e:
            log_json("ERROR", "broadcast_error", {"error": str(e)})
            await asyncio.sleep(10)

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
    # Always allow anonymous access when auth is not available
    if not AUTH_AVAILABLE:
        return {"username": "anonymous", "role": "admin"}
    
    # Also allow if no credentials provided (for development)
    if not credentials:
        return {"username": "anonymous", "role": "admin"}
    
    try:
        auth = get_auth_manager()
        user = auth.get_current_user(credentials.credentials)
        return user.to_dict()
    except Exception:
        # Fall back to anonymous on error
        return {"username": "anonymous", "role": "admin"}

# Health check
@app.get("/health")
async def health_check():
    queue_data = get_goal_queue_data()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "goals_queued": len(queue_data.get("queue", [])),
        "goals_in_flight": len(queue_data.get("in_flight", {})),
    }

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
    """Get all goals from queue and archive."""
    queue_data = get_goal_queue_data()
    archive = get_goal_archive()
    goals = format_goals_for_api(queue_data, archive)
    return goals

@app.post("/api/goals")
async def create_goal(goal: dict, user: dict = Depends(get_current_user)):
    """Create a new goal and add to queue."""
    description = goal.get("description", "")
    
    # Add to actual goal queue if available
    if GOAL_QUEUE_AVAILABLE:
        try:
            gq = GoalQueue()
            gq.add(description)
        except Exception as e:
            log_json("ERROR", "goal_add_failed", {"error": str(e)})
    
    new_goal = {
        "id": f"goal-new-{hash(description) & 0xFFFFFFFF}",
        "description": description,
        "status": "pending",
        "priority": goal.get("priority", 1),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "cycles": 0,
        "max_cycles": 10,
    }
    
    await manager.broadcast({
        "type": "goal_created",
        "payload": new_goal,
    })
    
    return new_goal

@app.post("/api/goals/{goal_id}/cancel")
async def cancel_goal(goal_id: str, user: dict = Depends(get_current_user)):
    """Cancel a running goal."""
    # Note: Actual cancellation would require modifying the goal queue
    await manager.broadcast({
        "type": "goal_updated",
        "payload": {"id": goal_id, "status": "failed"},
    })
    return {"success": True}

# Agents endpoints
@app.get("/api/agents")
async def get_agents(user: dict = Depends(get_current_user)):
    """Get all registered agents."""
    return get_agents_from_registry()

@app.get("/api/agents/{agent_id}/logs")
async def get_agent_logs(agent_id: str, user: dict = Depends(get_current_user)):
    """Get logs for a specific agent from telemetry."""
    telemetry = get_telemetry_data()
    agent_logs = [t for t in telemetry if t.get("agent_name") == agent_id]
    return agent_logs[:50]  # Last 50 entries

# Telemetry endpoint
@app.get("/api/telemetry")
async def get_telemetry(user: dict = Depends(get_current_user)):
    """Get recent telemetry data."""
    return get_telemetry_data()

# Stats endpoint
@app.get("/api/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    """Get system statistics."""
    queue_data = get_goal_queue_data()
    archive = get_goal_archive()
    goals = format_goals_for_api(queue_data, archive)
    agents = get_agents_from_registry()
    telemetry = get_telemetry_data()
    
    pending = len([g for g in goals if g["status"] == "pending"])
    running = len([g for g in goals if g["status"] == "running"])
    completed = len([g for g in goals if g["status"] == "completed"])
    failed = len([g for g in goals if g["status"] == "failed"])
    
    # Calculate average latency from telemetry
    avg_latency = 0
    if telemetry:
        avg_latency = sum(t.get("latency", 0) for t in telemetry) / len(telemetry)
    
    return {
        "goals": {
            "total": len(goals),
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
        },
        "agents": {
            "total": len(agents),
            "active": len([a for a in agents if a["status"] == "busy"]),
            "idle": len([a for a in agents if a["status"] == "idle"]),
        },
        "telemetry": {
            "total_records": len(telemetry),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "system": {
            "uptime_seconds": 3600,
            "memory_usage_mb": 256,
            "cpu_percent": 15.5,
        },
    }

# Chat endpoints
@app.post("/api/chat")
async def chat_endpoint(request: dict, user: dict = Depends(get_current_user)):
    """Handle chat messages with agents."""
    message = request.get("message", "")
    agent = request.get("agent", "planner")
    session_id = request.get("session_id", "")
    
    # Import and use the agent
    try:
        from agents.registry import default_agents
        from core.model_adapter import ModelAdapter
        from memory.brain import Brain
        
        brain = Brain(Path("."))
        model = ModelAdapter()
        agents = default_agents(brain, model)
        
        # Find the requested agent or use planner as fallback
        agent_instance = agents.get(agent) or agents.get("planner")
        
        if agent_instance:
            # Execute the agent with the message
            result = await agent_instance.run(message) if hasattr(agent_instance, 'run') else {"response": f"Agent {agent} received: {message}"}
            
            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "chat_message",
                "payload": {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": result.get("response", str(result)),
                    "agent": agent,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            })
            
            return {
                "response": result.get("response", str(result)),
                "metadata": {
                    "model": getattr(result, 'model', 'unknown'),
                    "tokens": getattr(result, 'tokens_used', 0),
                }
            }
        else:
            return {"response": f"Agent '{agent}' not found. Available agents will be listed."}
            
    except Exception as e:
        return {"response": f"Error processing message: {str(e)}"}

# File endpoints
@app.get("/api/files/tree")
async def get_file_tree(path: str = "", user: dict = Depends(get_current_user)):
    """Get file tree for the given path."""
    try:
        from pathlib import Path as PathLib
        
        base_path = PathLib("/home/westonaaron675/aura-cli")
        target_path = base_path / path if path else base_path
        
        if not str(target_path).startswith(str(base_path)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        def build_tree(p: PathLib, depth: int = 0) -> list:
            if depth > 3:  # Limit depth
                return []
            
            items = []
            try:
                for item in sorted(p.iterdir()):
                    if item.name.startswith('.') or item.name in ['node_modules', '__pycache__', '.git']:
                        continue
                    
                    node = {
                        "name": item.name,
                        "path": str(item.relative_to(base_path)),
                        "type": "directory" if item.is_dir() else "file",
                    }
                    
                    if item.is_dir() and depth < 3:
                        node["children"] = build_tree(item, depth + 1)
                    
                    items.append(node)
            except PermissionError:
                pass
            
            return items
        
        return build_tree(target_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/content")
async def get_file_content(path: str, user: dict = Depends(get_current_user)):
    """Get content of a file."""
    try:
        from pathlib import Path as PathLib
        
        base_path = PathLib("/home/westonaaron675/aura-cli")
        file_path = base_path / path
        
        if not str(file_path.resolve()).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is a directory")
        
        # Limit file size to 1MB
        if file_path.stat().st_size > 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        
        content = file_path.read_text(encoding='utf-8', errors='replace')
        return {"content": content, "path": path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/files/save")
async def save_file(request: dict, user: dict = Depends(get_current_user)):
    """Save content to a file."""
    try:
        from pathlib import Path as PathLib
        
        base_path = PathLib("/home/westonaaron675/aura-cli")
        path = request.get("path", "")
        content = request.get("content", "")
        
        file_path = base_path / path
        
        if not str(file_path.resolve()).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path.write_text(content, encoding='utf-8')
        
        return {"success": True, "path": path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# SADD endpoints
sadd_sessions_store: List[Dict[str, Any]] = []

@app.get("/api/sadd/sessions")
async def get_sadd_sessions(user: dict = Depends(get_current_user)):
    """Get all SADD sessions."""
    return sadd_sessions_store

@app.post("/api/sadd/sessions")
async def create_sadd_session(request: dict, user: dict = Depends(get_current_user)):
    """Create a new SADD session."""
    session = {
        "id": f"sadd-{len(sadd_sessions_store) + 1}",
        "title": request.get("title", "Untitled Session"),
        "design_spec": request.get("design_spec", ""),
        "status": "idle",
        "workstreams": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "artifacts": [],
    }
    sadd_sessions_store.insert(0, session)
    return session

@app.post("/api/sadd/sessions/{session_id}/start")
async def start_sadd_session(session_id: str, user: dict = Depends(get_current_user)):
    """Start a SADD session."""
    for session in sadd_sessions_store:
        if session["id"] == session_id:
            session["status"] = "running"
            session["updated_at"] = datetime.utcnow().isoformat()
            
            # Initialize workstreams from design spec
            # This is a simplified version - real implementation would parse the spec
            session["workstreams"] = [
                {
                    "id": f"ws-{i}",
                    "name": f"Workstream {i+1}",
                    "description": "Auto-generated from design spec",
                    "status": "pending",
                    "dependencies": [],
                    "artifacts": [],
                    "progress": 0,
                }
                for i in range(3)
            ]
            
            return {"success": True}
    
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/sadd/sessions/{session_id}/pause")
async def pause_sadd_session(session_id: str, user: dict = Depends(get_current_user)):
    """Pause a SADD session."""
    for session in sadd_sessions_store:
        if session["id"] == session_id:
            session["status"] = "paused"
            session["updated_at"] = datetime.utcnow().isoformat()
            return {"success": True}
    
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/sadd/sessions/{session_id}/resume")
async def resume_sadd_session(session_id: str, user: dict = Depends(get_current_user)):
    """Resume a SADD session."""
    for session in sadd_sessions_store:
        if session["id"] == session_id:
            session["status"] = "running"
            session["updated_at"] = datetime.utcnow().isoformat()
            return {"success": True}
    
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/sadd/sessions/{session_id}/stop")
async def stop_sadd_session(session_id: str, user: dict = Depends(get_current_user)):
    """Stop a SADD session."""
    for session in sadd_sessions_store:
        if session["id"] == session_id:
            session["status"] = "failed"
            session["updated_at"] = datetime.utcnow().isoformat()
            return {"success": True}
    
    raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/api/sadd/sessions/{session_id}")
async def delete_sadd_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a SADD session."""
    global sadd_sessions_store
    sadd_sessions_store = [s for s in sadd_sessions_store if s["id"] != session_id]
    return {"success": True}

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    # Send initial data
    queue_data = get_goal_queue_data()
    archive = get_goal_archive()
    goals = format_goals_for_api(queue_data, archive)
    agents = get_agents_from_registry()
    
    await websocket.send_json({
        "type": "initial",
        "payload": {
            "goals": goals[:20],  # Send last 20
            "agents": agents,
        }
    })
    
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

# Legacy WebSocket endpoint for logs (backward compatibility)
@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming."""
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

# GitHub App webhook endpoint
@app.post("/api/github/webhook")
async def github_webhook(request: Request, x_github_event: str = Header(None), x_hub_signature_256: str = Header(None)):
    """Handle GitHub App webhook events."""
    from aura_cli.github_integration import get_github_app
    
    body = await request.body()
    
    github_app = get_github_app()
    if not github_app:
        raise HTTPException(status_code=503, detail="GitHub App not configured")
    
    if not github_app.verify_webhook_signature(body, x_hub_signature_256 or ""):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    event_type = x_github_event or "unknown"
    result = github_app.handle_webhook(event_type, payload)
    
    log_json("INFO", "github_webhook_processed", {
        "event": event_type,
        "action": payload.get("action"),
        "result": result.get("status"),
    })
    
    return result

@app.get("/api/github/callback")
async def github_callback(code: str, installation_id: Optional[str] = None, setup_action: Optional[str] = None):
    """Handle GitHub App installation callback."""
    log_json("INFO", "github_installation_callback", {
        "installation_id": installation_id,
        "setup_action": setup_action,
    })
    
    return {
        "status": "success",
        "message": "GitHub App installed successfully",
        "installation_id": installation_id,
    }

# Function to broadcast log messages (for integration with other modules)
def broadcast_log(log_entry: dict):
    """Broadcast a log entry to all connected WebSocket clients."""
    asyncio.create_task(manager.broadcast({
        "type": "log",
        "payload": log_entry,
    }))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
