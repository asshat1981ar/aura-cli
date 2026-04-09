"""Notification system for external integrations.

Supports Slack, Discord, and webhook notifications.
"""

from .manager import NotificationManager, NotificationChannel
from .slack import SlackNotifier
from .discord import DiscordNotifier

__all__ = [
    "NotificationManager",
    "NotificationChannel",
    "SlackNotifier",
    "DiscordNotifier",
]
