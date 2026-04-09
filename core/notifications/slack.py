"""Slack webhook integration for notifications."""

from __future__ import annotations

from typing import Dict, Any, Optional
import asyncio
import json

from core.logging_utils import log_json
from .manager import NotificationEvent, NotificationChannel


class SlackNotifier:
    """Slack webhook notifier."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self._enabled = bool(webhook_url)
    
    def configure(self, webhook_url: str) -> None:
        """Configure the webhook URL.
        
        Args:
            webhook_url: Slack incoming webhook URL
        """
        self.webhook_url = webhook_url
        self._enabled = True
        log_json("INFO", "slack_configured")
    
    async def send(self, event: NotificationEvent) -> bool:
        """Send notification to Slack.
        
        Args:
            event: Notification event
            
        Returns:
            True if sent successfully
        """
        if not self._enabled or not self.webhook_url:
            log_json("DEBUG", "slack_skipped", {"reason": "not_configured"})
            return False
        
        payload = self._format_payload(event)
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    success = response.status == 200
                    
                    if success:
                        log_json("INFO", "slack_notification_sent", {
                            "event_type": event.type,
                            "title": event.title,
                        })
                    else:
                        log_json("ERROR", "slack_notification_failed", {
                            "status": response.status,
                            "response": await response.text(),
                        })
                    
                    return success
                    
        except ImportError:
            # Fallback to requests if aiohttp not available
            try:
                import requests
                
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                success = response.status_code == 200
                
                if success:
                    log_json("INFO", "slack_notification_sent", {
                        "event_type": event.type,
                    })
                
                return success
                
            except ImportError:
                log_json("ERROR", "slack_no_http_client")
                return False
                
        except Exception as e:
            log_json("ERROR", "slack_notification_error", {"error": str(e)})
            return False
    
    def _format_payload(self, event: NotificationEvent) -> Dict[str, Any]:
        """Format event for Slack webhook.
        
        Args:
            event: Notification event
            
        Returns:
            Slack payload
        """
        # Color based on priority
        colors = {
            "low": "#36a64f",      # Green
            "normal": "#2196F3",   # Blue
            "high": "#ff9800",     # Orange
            "urgent": "#f44336",   # Red
        }
        
        emoji = {
            "pr_opened": "📝",
            "pr_merged": "✅",
            "pr_closed": "❌",
            "goal_completed": "🎯",
            "goal_failed": "⚠️",
            "error": "🚨",
            "info": "ℹ️",
        }
        
        icon = emoji.get(event.type, "📢")
        color = colors.get(event.priority, "#2196F3")
        
        # Build attachment fields from metadata
        fields = []
        for key, value in event.metadata.items():
            if isinstance(value, (str, int, float)):
                fields.append({
                    "title": key.replace("_", " ").title(),
                    "value": str(value),
                    "short": len(str(value)) < 50,
                })
        
        return {
            "text": f"{icon} *{event.title}*",
            "attachments": [
                {
                    "color": color,
                    "text": event.message,
                    "fields": fields,
                    "footer": "AURA Notifications",
                    "ts": int(asyncio.get_event_loop().time()),
                }
            ]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get notifier status."""
        return {
            "channel": "slack",
            "enabled": self._enabled,
            "configured": bool(self.webhook_url),
        }


# Global instance
_notifier: Optional[SlackNotifier] = None


def get_slack_notifier() -> SlackNotifier:
    """Get global Slack notifier."""
    global _notifier
    if _notifier is None:
        _notifier = SlackNotifier()
    return _notifier


async def handle_slack_notification(event: NotificationEvent) -> None:
    """Handler for Slack notifications.
    
    Args:
        event: Notification event
    """
    notifier = get_slack_notifier()
    await notifier.send(event)
