# R2: Shim — canonical TaskManager lives in core/task_manager.py.
# This file is kept for backward-compatibility; import from core.task_manager instead.
from core.task_manager import *  # noqa: F401, F403
from core.task_manager import Task, TaskManager  # noqa: F401 (explicit re-export)
