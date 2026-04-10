"""Conftest for tests/unit/.

All tests collected under this directory are automatically tagged with
``pytest.mark.unit`` so they can be selected with ``-m unit``.
"""

from __future__ import annotations

import pytest


# Auto-apply the `unit` marker to every test in this subdirectory.
def pytest_collection_modifyitems(items):
    for item in items:
        if str(item.fspath).find("/tests/unit/") != -1:
            item.add_marker(pytest.mark.unit)
