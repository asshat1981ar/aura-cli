"""
Cache adapter factory — returns the best available adapter.

Usage::

    from memory.cache_adapter_factory import create_cache_adapter
    adapter = create_cache_adapter()   # MomentoAdapter or LocalCacheAdapter
    adapter.cache_set("aura-working-memory", "key", "value")

Selection logic:
    1. If ``MOMENTO_API_KEY`` env var is set → :class:`MomentoAdapter` (cloud L1)
    2. Otherwise → :class:`LocalCacheAdapter` (in-process L1, always available)

This lets code call ``create_cache_adapter()`` once and never worry about
which backend is active.  Both adapters share the same interface, so
callers are unaffected by the switch.
"""
from __future__ import annotations

import os


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
