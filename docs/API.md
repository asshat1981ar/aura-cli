# AURA API Documentation

## Base URL
```
http://localhost:8000/api
```

## Authentication
All endpoints require authentication via Bearer token in the Authorization header:
```
Authorization: Bearer <token>
```

## Core Endpoints

### Goals

#### List Goals
```http
GET /goals
```
Returns all goals in the queue and archive.

**Response:**
```json
[
  {
    "id": "goal_001",
    "title": "Implement feature X",
    "status": "pending",
    "priority": 5,
    "created_at": "2024-01-01T00:00:00Z"
  }
]
```

#### Add Goal
```http
POST /goals
Content-Type: application/json

{
  "title": "New goal",
  "description": "Detailed description",
  "priority": 3
}
```

#### Get Goal
```http
GET /goals/{goal_id}
```

#### Update Goal
```http
PUT /goals/{goal_id}
Content-Type: application/json

{
  "status": "completed"
}
```

#### Delete Goal
```http
DELETE /goals/{goal_id}
```

### Agents

#### List Agents
```http
GET /agents
```
Returns all registered agents.

**Response:**
```json
[
  {
    "id": "planner",
    "name": "Planner Agent",
    "status": "idle",
    "capabilities": ["planning", "design"]
  }
]
```

#### Get Agent Overview
```http
GET /agents/overview
```
Returns comprehensive overview with metrics.

#### Get Agent Metrics
```http
GET /agents/{agent_id}/metrics
```

**Response:**
```json
{
  "agent_id": "planner",
  "total_executions": 150,
  "success_rate": 95.3,
  "avg_latency_ms": 245,
  "errors": 7
}
```

#### Get Agent Logs
```http
GET /agents/{agent_id}/logs?limit=50
```

#### Control Agent
```http
POST /agents/{agent_id}/pause
POST /agents/{agent_id}/resume
POST /agents/{agent_id}/restart
```

### Workflows (n8n)

#### List Workflows
```http
GET /workflows
```

**Response:**
```json
[
  {
    "id": "WF-0-master-dispatcher",
    "name": "Master Dispatcher",
    "nodes": 15,
    "active": true
  }
]
```

#### Get Workflow
```http
GET /workflows/{workflow_id}
```

#### Execute Workflow
```http
POST /workflows/{workflow_id}/execute
Content-Type: application/json

{
  "input": "data"
}
```

#### Toggle Workflow
```http
POST /workflows/{workflow_id}/activate
POST /workflows/{workflow_id}/deactivate
```

### MCP Servers

#### List MCP Servers
```http
GET /mcp/servers
```

**Response:**
```json
[
  {
    "id": "filesystem",
    "name": "Filesystem",
    "type": "stdio",
    "status": "connected",
    "tools_count": 4
  }
]
```

#### Get Server Tools
```http
GET /mcp/servers/{server_id}/tools
```

#### Execute MCP Tool
```http
POST /mcp/servers/{server_id}/tools/{tool_name}/execute
Content-Type: application/json

{
  "param1": "value1"
}
```

### Terminal

#### Execute Command
```http
POST /terminal/execute
Content-Type: application/json

{
  "command": "ls -la",
  "session_id": "optional-session-id",
  "cwd": "/home/user"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "output": "...",
  "exit_code": 0,
  "cwd": "/home/user"
}
```

#### List Sessions
```http
GET /terminal/sessions
```

#### Get Session History
```http
GET /terminal/sessions/{session_id}
```

#### Clear Session
```http
POST /terminal/sessions/{session_id}/clear
```

#### Delete Session
```http
DELETE /terminal/sessions/{session_id}
```

### Telemetry & Monitoring

#### Get Telemetry
```http
GET /telemetry?limit=100
```

#### Get Telemetry Summary
```http
GET /telemetry/summary
```

**Response:**
```json
{
  "total_records": 1000,
  "avg_latency_ms": 45.2,
  "success_rate": 96.5,
  "by_agent": {},
  "by_hour": []
}
```

#### Get System Stats
```http
GET /stats
```

#### Get Health Status
```http
GET /health
```

### Coverage & Testing

#### Get Coverage
```http
GET /coverage
```

#### Get Coverage Gaps
```http
GET /coverage/gaps
```

#### Get Test Suite Info
```http
GET /tests
```

#### Run Tests
```http
POST /tests/run
```

### Settings

#### Get Settings
```http
GET /settings
```

#### Update Settings
```http
POST /settings
Content-Type: application/json

{
  "theme": "dark",
  "refresh_interval": 5
}
```

#### Get MCP Configuration
```http
GET /settings/mcp
```

#### Update MCP Configuration
```http
POST /settings/mcp
Content-Type: application/json

{
  "mcpServers": {}
}
```

### Chat

#### Send Message
```http
POST /chat
Content-Type: application/json

{
  "message": "Hello",
  "agent": "planner",
  "session_id": "uuid"
}
```

### Files

#### Get File Tree
```http
GET /files/tree?path=
```

#### Get File Content
```http
GET /files/content?path=/path/to/file
```

## WebSocket

Connect to WebSocket for real-time updates:
```
ws://localhost:8000/ws
```

### Message Types

#### Initial Data
```json
{
  "type": "initial",
  "goals": [],
  "agents": []
}
```

#### Goal Updates
```json
{
  "type": "goal_update",
  "goal_id": "...",
  "status": "running"
}
```

#### Agent Updates
```json
{
  "type": "agent_status_change",
  "agent_id": "...",
  "status": "busy"
}
```

#### Terminal Output
```json
{
  "type": "terminal_output",
  "session_id": "...",
  "output": "..."
}
```

## Error Responses

All errors follow this format:
```json
{
  "detail": "Error message",
  "status_code": 400
}
```

Common status codes:
- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Internal Server Error
