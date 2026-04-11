"""
Cache adapter factory — returns the best available adapter.

Usage::

    from memory.cache_adapter_factory import create_cache_adapter
    adapter = create_cache_adapter()   # MomentoAdapter or LocalCacheAdapter
    adapter.cache_set("aura-working-memory", "key", "value")

    # Explicit type selection (e.g. for L0 Redis acceleration):
    from memory.cache_adapter_factory import get_adapter
    redis_adapter = get_adapter("redis")

Selection logic (``create_cache_adapter``):
    1. If ``MOMENTO_API_KEY`` env var is set → :class:`MomentoAdapter` (cloud L1)
    2. Otherwise → :class:`LocalCacheAdapter` (in-process L1, always available)

``get_adapter(adapter_type)`` supports:
    - ``"local"``   → :class:`~memory.local_cache_adapter.LocalCacheAdapter`
    - ``"momento"`` → :class:`~memory.momento_adapter.MomentoAdapter`
    - ``"redis"``   → :class:`~memory.redis_cache_adapter.RedisCacheAdapter`

This lets code call ``create_cache_adapter()`` once and never worry about
which backend is active.  Both adapters share the same interface, so
callers are unaffected by the switch.
"""

from __future__ import annotations

import os

_ADAPTER_TYPES = ("local", "momento", "redis")


def create_cache_adapter():
    """Return the appropriate cache adapter based on environment.

    Returns:
        :class:`~memory.momento_adapter.MomentoAdapter` when
        ``MOMENTO_API_KEY`` is set, otherwise
        :class:`~memory.local_cache_adapter.LocalCacheAdapter`.
    """
    if os.getenv("MOMENTO_API_KEY"):
        from memory.momento_adapter import MomentoAdapter

        return MomentoAdapter()

    from memory.local_cache_adapter import LocalCacheAdapter

    return LocalCacheAdapter()


def get_adapter(adapter_type: str):
    """Return a cache adapter instance for the given *adapter_type*.

    Args:
        adapter_type: One of ``"local"``, ``"momento"``, or ``"redis"``.

    Returns:
        An instance of the requested adapter class.

    Raises:
        ValueError: If *adapter_type* is not recognised.
    """
    adapter_type = adapter_type.lower()
    if adapter_type == "local":
        from memory.local_cache_adapter import LocalCacheAdapter

        return LocalCacheAdapter()
    if adapter_type == "momento":
        from memory.momento_adapter import MomentoAdapter

        return MomentoAdapter()
    if adapter_type == "redis":
        from memory.redis_cache_adapter import RedisCacheAdapter

        return RedisCacheAdapter()
    raise ValueError(f"Unknown adapter type {adapter_type!r}. Valid options: {_ADAPTER_TYPES}")
