from typing import Any

def process_task(task: Any):
    # process individual task
    """
    Normalize and validate a task object before it is enqueued or executed.

    This helper returns a dictionary representation of the task with at least
    a ``status`` field. Callers can rely on the returned value having a
    consistent shape suitable for use in a task queue.

    :param task: The raw task object to process. Can be a dict or any other type.
    :raises ValueError: If ``task`` is None.
    :return: A normalized task dictionary.
    """

    if task is None:
        raise ValueError("task must not be None")

    # If the task is already a mapping, make a shallow copy and ensure defaults.
    if isinstance(task, dict):
        normalized = dict(task)
        normalized.setdefault("status", "pending")
        return normalized

    # For non-dict tasks, wrap the value in a basic task structure.
    return {
        "id": None,
        "payload": task,
        "status": "pending",
    }
