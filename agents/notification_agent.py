"""Notification agent — sends alerts via Slack, Discord, PagerDuty, email, webhooks.

Routes to the notification MCP server (port 8015) or sends directly via
HTTP when the MCP server is not available.
"""
from __future__ import annotations

import time
import requests
from typing import Dict, List, Optional


class NotificationAgentAdapter:
    """Pipeline adapter for notification dispatch.

    Supports multiple channels: Slack, Discord, PagerDuty, email, and
    generic webhooks.
    """

    name = "notification"

    def __init__(self, notification_mcp_url: Optional[str] = None):
        self._mcp_url = notification_mcp_url or "http://127.0.0.1:8015"
        self._history: List[Dict] = []
        self._max_history = 500

    def run(self, input_data: dict) -> dict:
        """Send a notification.

        Args:
            input_data: Dict with keys:
                - channel (str): "slack", "discord", "pagerduty", "email", "webhook".
                - message (str): Message content.
                - severity (str, optional): "info", "warning", "critical". Default "info".
                - metadata (dict, optional): Channel-specific metadata.
                    For slack: {channel: "#channel-name"}
                    For discord: {webhook_url: "https://..."}
                    For pagerduty: {routing_key: "..."}
                    For email: {to: "...", subject: "..."}
                    For webhook: {url: "...", method: "POST"}

        Returns:
            Dict with status, delivery_id, timestamp.
        """
        channel = input_data.get("channel", "")
        message = input_data.get("message", "")
        severity = input_data.get("severity", "info")
        metadata = input_data.get("metadata", {})

        if not channel:
            return {"error": "'channel' is required"}
        if not message:
            return {"error": "'message' is required"}

        # Try to send via MCP server first
        result = self._send_via_mcp(channel, message, severity, metadata)

        # Record in history
        entry = {
            "channel": channel,
            "message": message[:200],
            "severity": severity,
            "timestamp": time.time(),
            "status": result.get("status", "unknown"),
        }
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return result

    def _send_via_mcp(
        self, channel: str, message: str, severity: str, metadata: dict
    ) -> dict:
        """Send notification via the notification MCP server."""
        tool_map = {
            "slack": "slack_send",
            "discord": "discord_send",
            "pagerduty": "pagerduty_trigger",
            "email": "email_send",
            "webhook": "webhook_fire",
        }

        tool_name = tool_map.get(channel)
        if not tool_name:
            return {"error": f"Unknown channel: {channel}", "status": "failed"}

        # Build tool-specific args
        args = self._build_args(channel, message, severity, metadata)

        try:
            response = requests.post(
                f"{self._mcp_url}/call",
                json={"tool_name": tool_name, "args": args},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "sent",
                    "channel": channel,
                    "tool": tool_name,
                    "result": data.get("result"),
                    "timestamp": time.time(),
                }
            return {
                "status": "failed",
                "channel": channel,
                "error": f"MCP server returned {response.status_code}",
            }
        except Exception as exc:
            return {
                "status": "failed",
                "channel": channel,
                "error": f"MCP server unreachable: {exc}",
            }

    def _build_args(
        self, channel: str, message: str, severity: str, metadata: dict
    ) -> dict:
        """Build channel-specific args for the MCP tool call."""
        if channel == "slack":
            return {
                "channel": metadata.get("channel", "#general"),
                "text": f"[{severity.upper()}] {message}",
            }
        elif channel == "discord":
            return {
                "webhook_url": metadata.get("webhook_url", ""),
                "content": f"[{severity.upper()}] {message}",
            }
        elif channel == "pagerduty":
            return {
                "routing_key": metadata.get("routing_key", ""),
                "summary": message,
                "severity": severity,
            }
        elif channel == "email":
            return {
                "to": metadata.get("to", ""),
                "subject": metadata.get("subject", f"[AURA {severity.upper()}] Notification"),
                "body": message,
            }
        elif channel == "webhook":
            return {
                "url": metadata.get("url", ""),
                "payload": {"message": message, "severity": severity, **metadata},
                "method": metadata.get("method", "POST"),
            }
        return {"message": message}

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent notification history."""
        return self._history[-limit:]
