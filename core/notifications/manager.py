"""Notification manager for routing events to channels."""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

from core.logging_utils import log_json


class NotificationChannel(Enum):
    """Available notification channels."""

    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class NotificationEvent:
    """Notification event data."""

    type: str
    title: str
    message: str
    metadata: Dict[str, Any]
    priority: str = "normal"  # low, normal, high, urgent
    channels: List[NotificationChannel] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = []


class NotificationManager:
    """Manage and route notifications to configured channels."""

    def __init__(self):
        self._handlers: Dict[NotificationChannel, Callable] = {}
        self._rules: List[Dict[str, Any]] = []
        self._enabled = True

    def register_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[NotificationEvent], asyncio.Coroutine],
    ) -> None:
        """Register a handler for a notification channel.

        Args:
            channel: The channel to handle
            handler: Async handler function
        """
        self._handlers[channel] = handler
        log_json(
            "INFO",
            "notification_handler_registered",
            {
                "channel": channel.value,
            },
        )

    def add_rule(
        self,
        event_type: str,
        channels: List[NotificationChannel],
        condition: Optional[Callable[[NotificationEvent], bool]] = None,
    ) -> None:
        """Add a routing rule for events.

        Args:
            event_type: Type of event to match
            channels: Channels to send to
            condition: Optional condition function
        """
        self._rules.append(
            {
                "event_type": event_type,
                "channels": channels,
                "condition": condition,
            }
        )
        log_json(
            "INFO",
            "notification_rule_added",
            {
                "event_type": event_type,
                "channels": [c.value for c in channels],
            },
        )

    async def send(self, event: NotificationEvent) -> Dict[str, Any]:
        """Send a notification event.

        Args:
            event: The notification event

        Returns:
            Results by channel
        """
        if not self._enabled:
            return {"status": "disabled"}

        # Determine channels from rules if not specified
        if not event.channels:
            event.channels = self._get_channels_for_event(event)

        results = {}

        for channel in event.channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    await handler(event)
                    results[channel.value] = "sent"
                except Exception as e:
                    log_json(
                        "ERROR",
                        "notification_send_failed",
                        {
                            "channel": channel.value,
                            "error": str(e),
                        },
                    )
                    results[channel.value] = f"failed: {e}"
            else:
                results[channel.value] = "no_handler"

        return results

    def _get_channels_for_event(self, event: NotificationEvent) -> List[NotificationChannel]:
        """Determine channels for an event based on rules."""
        channels = []

        for rule in self._rules:
            if rule["event_type"] == event.type or rule["event_type"] == "*":
                condition = rule.get("condition")
                if condition is None or condition(event):
                    channels.extend(rule["channels"])

        # Remove duplicates while preserving order
        seen = set()
        return [c for c in channels if not (c in seen or seen.add(c))]

    def enable(self) -> None:
        """Enable notifications."""
        self._enabled = True
        log_json("INFO", "notifications_enabled")

    def disable(self) -> None:
        """Disable notifications."""
        self._enabled = False
        log_json("INFO", "notifications_disabled")

    def get_status(self) -> Dict[str, Any]:
        """Get notification system status."""
        return {
            "enabled": self._enabled,
            "handlers": [c.value for c in self._handlers.keys()],
            "rules_count": len(self._rules),
        }


# Global manager instance
_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get global notification manager."""
    global _manager
    if _manager is None:
        _manager = NotificationManager()
    return _manager


# Convenience functions
async def notify(
    type: str,
    title: str,
    message: str,
    metadata: Dict[str, Any] = None,
    priority: str = "normal",
    channels: List[NotificationChannel] = None,
) -> Dict[str, Any]:
    """Send a notification.

    Args:
        type: Event type
        title: Notification title
        message: Notification message
        metadata: Additional data
        priority: Priority level
        channels: Specific channels (or None for rules-based)

    Returns:
        Send results
    """
    event = NotificationEvent(
        type=type,
        title=title,
        message=message,
        metadata=metadata or {},
        priority=priority,
        channels=channels,
    )

    return await get_notification_manager().send(event)


def setup_default_rules() -> None:
    """Set up default notification rules."""
    manager = get_notification_manager()

    # High priority events go to all channels
    manager.add_rule(
        event_type="pr_merged",
        channels=[NotificationChannel.SLACK, NotificationChannel.DISCORD],
    )

    manager.add_rule(
        event_type="goal_completed",
        channels=[NotificationChannel.SLACK],
    )

    # Error events
    manager.add_rule(
        event_type="error",
        channels=[NotificationChannel.SLACK, NotificationChannel.DISCORD],
        condition=lambda e: e.priority in ("high", "urgent"),
    )
