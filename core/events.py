import asyncio
from typing import Any, Callable, Dict, List, Awaitable

class AuraHooks:
    ON_CYCLE_START = "on_cycle_start"
    PRE_APPLY_CHANGES = "pre_apply_changes"
    POST_VERIFY = "post_verify"
    ON_AGENT_ERROR = "on_agent_error"

EventHandler = Callable[..., Awaitable[Any]]

class EventBus:
    _subscribers: Dict[str, List[EventHandler]] = {}

    @classmethod
    def subscribe(cls, event_name: str, handler: EventHandler) -> None:
        if event_name not in cls._subscribers:
            cls._subscribers[event_name] = []
        if handler not in cls._subscribers[event_name]:
            cls._subscribers[event_name].append(handler)

    @classmethod
    def unsubscribe(cls, event_name: str, handler: EventHandler) -> None:
        if event_name in cls._subscribers and handler in cls._subscribers[event_name]:
            cls._subscribers[event_name].remove(handler)

    @classmethod
    async def publish(cls, event_name: str, **kwargs: Any) -> List[Any]:
        handlers = cls._subscribers.get(event_name, [])
        if not handlers:
            return []
        results = await asyncio.gather(*(handler(**kwargs) for handler in handlers), return_exceptions=True)
        return results

    @classmethod
    def clear(cls) -> None:
        cls._subscribers.clear()

def aura_plugin(event_name: str):
    """Decorator to easily register a plugin function to an event."""
    def decorator(func: EventHandler) -> EventHandler:
        EventBus.subscribe(event_name, func)
        return func
    return decorator
