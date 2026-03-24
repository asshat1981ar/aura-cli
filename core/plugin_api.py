"""Stable versioned plugin API surface for AURA plugins.

Third-party plugins should import from this module exclusively.
Internal implementation details are subject to change.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

# Re-export stable public API
from core.events import AuraHooks, EventBus, aura_plugin
from core.plugin_loader import discover_plugins, get_registered_skills, register_skill

__all__ = [
    # Hooks / event names
    "AuraHooks",
    # Event bus
    "EventBus",
    # Decorator for quick registration
    "aura_plugin",
    # Plugin registry helpers
    "register_skill",
    "get_registered_skills",
    "discover_plugins",
    # Version constant
    "PLUGIN_API_VERSION",
]

PLUGIN_API_VERSION: str = "1.0"


class SkillPlugin:
    """Base class for all AURA skill plugins.

    Subclass this and implement ``run()`` to create a plugin skill.

    Entry-point group: ``aura.skills``

    Example ``pyproject.toml`` registration::

        [project.entry-points."aura.skills"]
        my_skill = "my_package.my_module:MySkill"
    """

    #: Human-readable name (must be unique across plugins)
    name: str = ""

    #: Short description shown in ``aura skills --list``
    description: str = ""

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the skill.

        Args:
            input_data: Arbitrary key/value pairs supplied by the caller.

        Returns:
            A dict with at minimum a ``"status"`` key.
            On error, return ``{"error": "<message>"}`` — never raise.
        """
        raise NotImplementedError


def on_event(event_name: str) -> Callable:
    """Alias for :func:`aura_plugin` — subscribe a coroutine to an event.

    Usage::

        from core.plugin_api import on_event, AuraHooks

        @on_event(AuraHooks.POST_VERIFY)
        async def my_handler(result=None, **kwargs):
            ...
    """
    return aura_plugin(event_name)
