# AURA API Guide

Complete reference for the AURA REST API and WebSocket endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Most endpoints require authentication via Bearer token:

```http
Authorization: Bearer <token>
```

## REST API Endpoints

### Goals

#### List Goals
```http
GET /api/goals
```

Query parameters:
- `status` (optional): Filter by status (`pending`, `running`, `completed`, `failed`)

Response:
```json
{
  "goals": [
    {
      "id": "goal-q-0",
      "description": "Refactor goal queue",
      "status": "pending",
      "priority": 1,
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00",
      "progress": 0,
      "cycles": 0,
      "max_cycles": 10
    }
  ]
}
```

#### Create Goal
```http
POST /api/goals
```

Request body:
```json
{
  "description": "Implement new feature",
  "priority": 1,
  "max_cycles": 10
}
```

#### Get Goal Detail
```http
GET /api/goals/{goal_id}
```

Response:
```json
{
  "id": "goal-q-0",
  "description": "Refactor goal queue",
  "status": "running",
  "history": [
    {
      "cycle": 1,
      "phase": "planning",
      "outcome": "success",
      "duration_s": 5.2,
      "timestamp": "2024-01-01T00:01:00"
    }
  ]
}
```

#### Cancel Goal
```http
POST /api/goals/{goal_id}/cancel
```

### Agents

#### List Agents
```http
GET /api/agents
```

Response:
```json
{
  "agents": [
    {
      "id": "planner",
      "name": "Planner Agent",
      "status": "idle",
      "capabilities": ["planning", "analysis"]
    }
  ]
}
```

#### Get Agent Detail
```http
GET /api/agents/{agent_id}
```

#### Control Agent
```http
POST /api/agents/{agent_id}/pause
POST /api/agents/{agent_id}/resume
POST /api/agents/{agent_id}/restart
```

### GitHub Integration

#### List Pull Requests
```http
GET /api/github/prs?state=open
```

Response:
```json
[
  {
    "id": 1,
    "number": 42,
    "title": "Add new feature",
    "state": "open",
    "user": {
      "login": "developer1",
      "avatar_url": "..."
    },
    "additions": 150,
    "deletions": 30,
    "comments": 3
  }
]
```

#### Get PR Detail
```http
GET /api/github/prs/{pr_number}
```

#### Get PR Reviews
```http
GET /api/github/prs/{pr_number}/reviews
```

#### Get PR Comments
```http
GET /api/github/prs/{pr_number}/comments
```

### Notifications

#### Get Notification Status
```http
GET /api/notifications/status
```

#### Test Notification
```http
POST /api/notifications/test
```

Request body:
```json
{
  "channel": "slack"
}
```

### Performance

#### Get Performance Stats
```http
GET /api/performance/stats
```

Response:
```json
{
  "cache": {
    "hits": 100,
    "misses": 20,
    "hit_rate": 0.83,
    "size": 120
  },
  "memory": {
    "rss_mb": 150.5,
    "vms_mb": 450.2,
    "percent": 2.5
  }
}
```

#### Clear Cache
```http
POST /api/performance/cache/clear
```

### SADD Sessions

#### List Sessions
```http
GET /api/sadd/sessions
```

#### Create Session
```http
POST /api/sadd/sessions
```

#### Control Session
```http
POST /api/sadd/sessions/{session_id}/pause
POST /api/sadd/sessions/{session_id}/resume
POST /api/sadd/sessions/{session_id}/stop
DELETE /api/sadd/sessions/{session_id}
```

### MCP Servers

#### List Servers
```http
GET /api/mcp/servers
```

#### Get Server Tools
```http
GET /api/mcp/servers/{server_id}/tools
```

#### Execute Tool
```http
POST /api/mcp/servers/{server_id}/tools/{tool_name}/execute
```

Request body:
```json
{
  "param1": "value1",
  "param2": "value2"
}
```

## WebSocket API

### Real-time Updates

Connect to WebSocket endpoint:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Message Types

#### Initial Data
```json
{
  "type": "initial",
  "payload": {
    "goals": [...],
    "agents": [...]
  }
}
```

#### GitHub PR Events
```json
{
  "type": "github_pr_event",
  "payload": {
    "event": "pull_request",
    "action": "opened",
    "pr_number": 42,
    "pr_title": "Add feature",
    "repo": "owner/repo",
    "sender": "username",
    "status": "processing"
  }
}
```

#### Ping/Pong
```json
// Client sends
{"type": "ping"}

// Server responds
{"type": "pong"}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error description"
}
```

Common status codes:
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Internal Server Error
- `503` - Service Unavailable

## Rate Limiting

API requests are limited to 100 requests per minute per IP address.

Response headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
```

## Webhook Integration

### GitHub Webhook

Endpoint: `POST /api/github/webhook`

Configure your GitHub App to send events to this endpoint.

Supported events:
- `pull_request`
- `pull_request_review`
- `pull_request_review_comment`
- `issue_comment`
- `push`
- `installation`

## SDK Examples

### Python

```python
import requests

# List goals
response = requests.get(
    "http://localhost:8000/api/goals",
    headers={"Authorization": "Bearer token"}
)
goals = response.json()

# Create goal
response = requests.post(
    "http://localhost:8000/api/goals",
    headers={"Authorization": "Bearer token"},
    json={
        "description": "Implement feature",
        "priority": 1
    }
)
```

### JavaScript

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

// Fetch PRs
const response = await fetch('/api/github/prs?state=open', {
  headers: {
    'Authorization': 'Bearer token'
  }
});
const prs = await response.json();
```
