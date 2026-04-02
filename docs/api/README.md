# AURA API Documentation

## Overview

AURA provides a comprehensive REST API and WebSocket interface for programmatic access to all AURA functionality.

## Base URL

```
http://localhost:8000/api
```

## Authentication

All API endpoints (except health check) require authentication via JWT Bearer token.

### Login

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "username": "admin",
    "role": "admin"
  }
}
```

### Using the Token

Include the token in the Authorization header:

```http
GET /api/goals
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

## Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-02T12:00:00Z"
}
```

### Goals

#### List Goals

```http
GET /api/goals
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": "goal-1",
    "description": "Fix authentication bug",
    "status": "running",
    "priority": 1,
    "created_at": "2026-04-02T10:00:00Z",
    "updated_at": "2026-04-02T10:05:00Z",
    "progress": 45,
    "cycles": 2,
    "max_cycles": 10
  }
]
```

#### Create Goal

```http
POST /api/goals
Authorization: Bearer <token>
Content-Type: application/json

{
  "description": "Add user profile feature",
  "priority": 2
}
```

**Response:**
```json
{
  "id": "goal-42",
  "description": "Add user profile feature",
  "status": "pending",
  "priority": 2,
  "created_at": "2026-04-02T12:00:00Z",
  "updated_at": "2026-04-02T12:00:00Z",
  "progress": 0,
  "cycles": 0,
  "max_cycles": 10
}
```

#### Cancel Goal

```http
POST /api/goals/{goal_id}/cancel
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true
}
```

### Agents

#### List Agents

```http
GET /api/agents
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": "planner-1",
    "name": "Planner Agent",
    "type": "planner",
    "status": "idle",
    "capabilities": ["planning", "analysis"],
    "last_seen": "2026-04-02T12:00:00Z",
    "stats": {
      "tasks_completed": 42,
      "tasks_failed": 3,
      "avg_execution_time": 5.2
    }
  }
]
```

### Statistics

#### Get System Stats

```http
GET /api/stats
Authorization: Bearer <token>
```

**Response:**
```json
{
  "goals": {
    "total": 10,
    "pending": 2,
    "running": 3,
    "completed": 4,
    "failed": 1
  },
  "agents": {
    "total": 5,
    "active": 2,
    "idle": 3
  },
  "system": {
    "uptime_seconds": 3600,
    "memory_usage_mb": 256,
    "cpu_percent": 15.5
  }
}
```

## WebSocket

### Real-time Logs

Connect to the WebSocket endpoint for real-time log streaming:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/logs');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

**Message Types:**

#### Connected
```json
{
  "type": "connected",
  "message": "Connected to log stream"
}
```

#### Log Entry
```json
{
  "type": "log",
  "payload": {
    "id": "log-123",
    "timestamp": "2026-04-02T12:00:00Z",
    "level": "INFO",
    "event": "goal_completed",
    "message": "Goal achieved successfully"
  }
}
```

#### Ping/Pong
```json
{"type": "ping"}
{"type": "pong"}
```

### GitHub Webhooks

```http
POST /api/github/webhook
X-GitHub-Event: pull_request
X-Hub-Signature-256: sha256=...

{
  "action": "opened",
  "pull_request": {
    "number": 123,
    "title": "Fix bug"
  }
}
```

## Error Responses

### 401 Unauthorized
```json
{
  "detail": "Not authenticated"
}
```

### 404 Not Found
```json
{
  "detail": "Goal not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "description"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Rate Limiting

API requests are limited to:
- 100 requests per minute for authenticated users
- 20 requests per minute for unauthenticated users

## SDK Examples

### Python

```python
import requests

# Login
response = requests.post('http://localhost:8000/api/auth/login', json={
    'username': 'admin',
    'password': 'admin'
})
token = response.json()['access_token']

# List goals
goals = requests.get(
    'http://localhost:8000/api/goals',
    headers={'Authorization': f'Bearer {token}'}
).json()

# Create goal
new_goal = requests.post(
    'http://localhost:8000/api/goals',
    headers={'Authorization': f'Bearer {token}'},
    json={'description': 'Add new feature'}
).json()
```

### JavaScript

```javascript
// Login
const login = async () => {
  const response = await fetch('http://localhost:8000/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'admin', password: 'admin' })
  });
  const { access_token } = await response.json();
  return access_token;
};

// Create goal
const createGoal = async (token, description) => {
  const response = await fetch('http://localhost:8000/api/goals', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ description })
  });
  return response.json();
};
```

## OpenAPI Schema

The full OpenAPI schema is available at:

```
http://localhost:8000/openapi.json
```

Interactive documentation (Swagger UI) is available at:

```
http://localhost:8000/docs
```
