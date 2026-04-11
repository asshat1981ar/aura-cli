"""Discord webhook integration for notifications."""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import asyncio

from core.logging_utils import log_json
from .manager import NotificationEvent, NotificationChannel


class DiscordNotifier:
    """Discord webhook notifier with rich embed support."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self._enabled = bool(webhook_url)
        self._username = "AURA"
        self._avatar_url: Optional[str] = None

    def configure(
        self,
        webhook_url: str,
        username: str = "AURA",
        avatar_url: Optional[str] = None,
    ) -> None:
        """Configure the webhook.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username
            avatar_url: Bot avatar URL
        """
        self.webhook_url = webhook_url
        self._username = username
        self._avatar_url = avatar_url
        self._enabled = True
        log_json("INFO", "discord_configured")

    async def send(self, event: NotificationEvent) -> bool:
        """Send notification to Discord.

        Args:
            event: Notification event

        Returns:
            True if sent successfully
        """
        if not self._enabled or not self.webhook_url:
            log_json("DEBUG", "discord_skipped", {"reason": "not_configured"})
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
                    success = response.status in (200, 204)

                    if success:
                        log_json(
                            "INFO",
                            "discord_notification_sent",
                            {
                                "event_type": event.type,
                                "title": event.title,
                            },
                        )
                    else:
                        log_json(
                            "ERROR",
                            "discord_notification_failed",
                            {
                                "status": response.status,
                                "response": await response.text(),
                            },
                        )

                    return success

        except ImportError:
            # Fallback to requests
            try:
                import requests

                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                success = response.status_code in (200, 204)

                if success:
                    log_json(
                        "INFO",
                        "discord_notification_sent",
                        {
                            "event_type": event.type,
                        },
                    )

                return success

            except ImportError:
                log_json("ERROR", "discord_no_http_client")
                return False

        except Exception as e:
            log_json("ERROR", "discord_notification_error", {"error": str(e)})
            return False

    def _format_payload(self, event: NotificationEvent) -> Dict[str, Any]:
        """Format event for Discord webhook with rich embeds.

        Args:
            event: Notification event

        Returns:
            Discord payload
        """
        # Color based on priority (Discord uses integer colors)
        colors = {
            "low": 0x36A64F,  # Green
            "normal": 0x2196F3,  # Blue
            "high": 0xFF9800,  # Orange
            "urgent": 0xF44336,  # Red
        }

        # Emoji mapping
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
        color = colors.get(event.priority, 0x2196F3)

        # Build embed fields
        fields: List[Dict[str, Any]] = []
        for key, value in event.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                fields.append(
                    {
                        "name": key.replace("_", " ").title(),
                        "value": str(value)[:1024],  # Discord limit
                        "inline": len(str(value)) < 50,
                    }
                )

        # Build rich embed
        embed = {
            "title": f"{icon} {event.title}",
            "description": event.message[:2048],  # Discord limit
            "color": color,
            "fields": fields[:25],  # Discord limit
            "timestamp": asyncio.get_event_loop().time(),
            "footer": {
                "text": "AURA Notifications",
            },
        }

        # Add URL if available in metadata
        if "url" in event.metadata:
            embed["url"] = event.metadata["url"]

        if "pr_url" in event.metadata:
            embed["url"] = event.metadata["pr_url"]

        payload: Dict[str, Any] = {
            "username": self._username,
            "embeds": [embed],
        }

        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        return payload

    def send_pr_notification(
        self,
        action: str,
        pr_number: int,
        pr_title: str,
        repo: str,
        author: str,
        url: str,
        additions: int = 0,
        deletions: int = 0,
    ) -> asyncio.Coroutine:
        """Convenience method for PR notifications.

        Args:
            action: PR action (opened, merged, closed)
            pr_number: PR number
            pr_title: PR title
            repo: Repository name
            author: PR author
            url: PR URL
            additions: Lines added
            deletions: Lines deleted

        Returns:
            Coroutine for sending
        """
        emoji_map = {
            "opened": "📝",
            "merged": "✅",
            "closed": "❌",
            "synchronize": "🔄",
        }

        event = NotificationEvent(
            type=f"pr_{action}",
            title=f"{emoji_map.get(action, '')} PR #{pr_number} {action.title()}",
            message=f"**{pr_title}** by {author}",
            metadata={
                "repository": repo,
                "author": author,
                "url": url,
                "additions": additions,
                "deletions": deletions,
                "files_changed": f"+{additions}/-{deletions}",
            },
            priority="normal" if action == "opened" else "low",
            channels=[NotificationChannel.DISCORD],
        )

        return self.send(event)

    def get_status(self) -> Dict[str, Any]:
        """Get notifier status."""
        return {
            "channel": "discord",
            "enabled": self._enabled,
            "configured": bool(self.webhook_url),
            "username": self._username,
        }


# Global instance
_notifier: Optional[DiscordNotifier] = None


def get_discord_notifier() -> DiscordNotifier:
    """Get global Discord notifier."""
    global _notifier
    if _notifier is None:
        _notifier = DiscordNotifier()
    return _notifier


async def handle_discord_notification(event: NotificationEvent) -> None:
    """Handler for Discord notifications.

    Args:
        event: Notification event
    """
    notifier = get_discord_notifier()
    await notifier.send(event)
