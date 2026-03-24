"""
Notification MCP Server — sends notifications via Slack, Discord, PagerDuty, email, and webhooks.

Tools exposed:
  Messaging : slack_send, discord_send
  Incidents : pagerduty_trigger
  Email     : email_send
  Generic   : webhook_fire

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → uptime and per-tool call/error counts

Start:
  uvicorn tools.notification_mcp:app --port 8015

Auth (optional):
  Set NOTIFICATION_MCP_TOKEN env var
"""
from __future__ import annotations

import os
import smtplib
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

import requests

from fastapi import Depends, FastAPI, Header, HTTPException
from tools.mcp_types import ToolCallRequest, ToolResult
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Notification MCP", version="1.0.0")
_TOKEN = os.getenv("NOTIFICATION_MCP_TOKEN", "")
_SERVER_START = time.time()
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _TOKEN:
        return
    if authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Tool descriptors (MCP schema format)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: Dict[str, Dict] = {
    "slack_send": {
        "description": "Send a message to a Slack channel.",
        "input": {
            "channel": {"type": "string", "description": "Slack channel (e.g. #general or channel ID)", "required": True},
            "text": {"type": "string", "description": "Message text", "required": True},
            "thread_ts": {"type": "string", "description": "Thread timestamp for replies (optional)"},
        },
    },
    "discord_send": {
        "description": "Send a message via a Discord webhook.",
        "input": {
            "webhook_url": {"type": "string", "description": "Discord webhook URL", "required": True},
            "content": {"type": "string", "description": "Message content", "required": True},
        },
    },
    "pagerduty_trigger": {
        "description": "Trigger a PagerDuty incident.",
        "input": {
            "routing_key": {"type": "string", "description": "PagerDuty integration routing key", "required": True},
            "summary": {"type": "string", "description": "Incident summary", "required": True},
            "severity": {"type": "string", "description": "Severity: critical, error, warning, info", "default": "warning"},
        },
    },
    "email_send": {
        "description": "Send an email via SMTP.",
        "input": {
            "to": {"type": "string", "description": "Recipient email address", "required": True},
            "subject": {"type": "string", "description": "Email subject", "required": True},
            "body": {"type": "string", "description": "Email body text", "required": True},
        },
    },
    "webhook_fire": {
        "description": "Fire a generic HTTP webhook.",
        "input": {
            "url": {"type": "string", "description": "Webhook URL", "required": True},
            "payload": {"type": "object", "description": "JSON payload to send", "required": True},
            "method": {"type": "string", "description": "HTTP method (POST, PUT, PATCH)", "default": "POST"},
        },
    },
}


def _build_descriptor(name: str) -> Dict:
    schema = _TOOL_SCHEMAS[name]
    return {
        "name": name,
        "description": schema["description"],
        "inputSchema": {
            "type": "object",
            "properties": schema.get("input", {}),
            "required": [k for k, v in schema.get("input", {}).items() if v.get("required")],
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _slack_send(args: Dict) -> Any:
    channel = args.get("channel", "").strip()
    if not channel:
        raise ValueError("'channel' is required.")
    text = args.get("text", "").strip()
    if not text:
        raise ValueError("'text' is required.")
    bot_token = os.getenv("SLACK_BOT_TOKEN", "")
    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN environment variable is not set.")

    payload: Dict[str, Any] = {"channel": channel, "text": text}
    thread_ts = args.get("thread_ts", "").strip()
    if thread_ts:
        payload["thread_ts"] = thread_ts

    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {bot_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error: {data.get('error', 'unknown')}")
    return {"channel": channel, "ts": data.get("ts"), "ok": True}


def _discord_send(args: Dict) -> Any:
    webhook_url = args.get("webhook_url", "").strip()
    if not webhook_url:
        raise ValueError("'webhook_url' is required.")
    content = args.get("content", "").strip()
    if not content:
        raise ValueError("'content' is required.")

    resp = requests.post(
        webhook_url,
        json={"content": content},
        timeout=30,
    )
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"Discord webhook failed with status {resp.status_code}: {resp.text}")
    return {"ok": True, "status_code": resp.status_code}


def _pagerduty_trigger(args: Dict) -> Any:
    routing_key = args.get("routing_key", "").strip()
    if not routing_key:
        raise ValueError("'routing_key' is required.")
    summary = args.get("summary", "").strip()
    if not summary:
        raise ValueError("'summary' is required.")
    severity = args.get("severity", "warning").strip()
    if severity not in ("critical", "error", "warning", "info"):
        raise ValueError("'severity' must be one of: critical, error, warning, info.")

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": summary,
            "severity": severity,
            "source": "aura-notification-mcp",
        },
    }
    resp = requests.post(
        "https://events.pagerduty.com/v2/enqueue",
        json=payload,
        timeout=30,
    )
    data = resp.json()
    if resp.status_code != 202:
        raise RuntimeError(f"PagerDuty API error: {data}")
    return {"status": data.get("status"), "dedup_key": data.get("dedup_key")}


def _email_send(args: Dict) -> Any:
    to = args.get("to", "").strip()
    if not to:
        raise ValueError("'to' is required.")
    subject = args.get("subject", "").strip()
    if not subject:
        raise ValueError("'subject' is required.")
    body = args.get("body", "").strip()
    if not body:
        raise ValueError("'body' is required.")

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    if not smtp_host:
        raise ValueError("SMTP_HOST environment variable is not set.")
    if not smtp_user:
        raise ValueError("SMTP_USER environment variable is not set.")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        if smtp_password:
            server.login(smtp_user, smtp_password)
        server.send_message(msg)

    return {"to": to, "subject": subject, "sent": True}


def _webhook_fire(args: Dict) -> Any:
    url = args.get("url", "").strip()
    if not url:
        raise ValueError("'url' is required.")
    payload = args.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("'payload' is required and must be a dict.")
    method = args.get("method", "POST").strip().upper()
    if method not in ("POST", "PUT", "PATCH"):
        raise ValueError("'method' must be one of: POST, PUT, PATCH.")

    resp = requests.request(
        method,
        url,
        json=payload,
        timeout=30,
    )
    return {
        "url": url,
        "method": method,
        "status_code": resp.status_code,
        "response": resp.text[:2000],
    }


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "slack_send": _slack_send,
    "discord_send": _discord_send,
    "pagerduty_trigger": _pagerduty_trigger,
    "email_send": _email_send,
    "webhook_fire": _webhook_fire,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "notification_mcp",
        "version": "1.0.0",
    }


@app.get("/tools")
async def list_tools(_: None = Depends(_check_auth)) -> List[Dict]:
    return [_build_descriptor(name) for name in _TOOL_SCHEMAS]


@app.get("/tool/{name}")
async def get_tool(name: str, _: None = Depends(_check_auth)) -> Dict:
    if name not in _TOOL_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return _build_descriptor(name)


@app.post("/call")
async def call_tool(request: ToolCallRequest, _: None = Depends(_check_auth)) -> ToolResult:
    name = request.tool_name
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")

    _call_counts[name] = _call_counts.get(name, 0) + 1
    t0 = time.time()
    try:
        result = handler(request.args)
        elapsed = round((time.time() - t0) * 1000, 2)
        log_json("INFO", "notification_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "notification_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "notification_mcp_tool_error", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=f"Internal error: {exc}", elapsed_ms=elapsed)


@app.get("/metrics")
async def get_metrics(_: None = Depends(_check_auth)) -> Dict:
    uptime_s = round(time.time() - _SERVER_START, 1)
    total_calls = sum(_call_counts.values())
    total_errors = sum(_call_errors.values())
    per_tool = {
        name: {
            "calls": _call_counts.get(name, 0),
            "errors": _call_errors.get(name, 0),
        }
        for name in _TOOL_SCHEMAS
    }
    return {
        "uptime_seconds": uptime_s,
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_calls, 1), 4),
        "tools": per_tool,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from core.config_manager import config as _cfg
    port = int(os.getenv("NOTIFICATION_MCP_PORT", _cfg.get_mcp_server_port("notification", default=8015)))
    uvicorn.run("tools.notification_mcp:app", host="0.0.0.0", port=port, reload=False)
