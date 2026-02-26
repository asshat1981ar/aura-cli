import json
from collections import deque
from pathlib import Path
from core.logging_utils import log_json

class TaskQueue:
    """
    Manages a persistent queue of tasks. Tasks are stored in a JSON file and
    loaded/saved automatically, ensuring work continuity across sessions.
    """

    def __init__(self, queue_path="memory/task_queue.json"):
        """
        Initializes the TaskQueue.

        Args:
            queue_path (str): The path to the JSON file where the queue is persisted.
                              Defaults to "memory/task_queue.json".
        """
        self.queue_path = Path(queue_path)
        self.queue = self._load_queue()

    def add_task(self, task):
        """
        Adds a new task to the end of the queue.

        Args:
            task: The task to be added.
        """
        self.queue.append(task)
        self._save_queue()
        log_json("INFO", "task_added", details={"task": task, "queue_size": len(self.queue)})

    def get_next_task(self):
        """
        Retrieves and removes the next task from the front of the queue.

        Returns:
            The next task in the queue, or None if the queue is empty.
        """
        if self.queue:
            task = self.queue.popleft()
            self._save_queue()
            log_json("INFO", "task_retrieved", details={"task": task, "queue_size": len(self.queue)})
            return task
        log_json("INFO", "no_tasks_in_queue")
        return None

    def has_tasks(self):
        """
        Checks if there are any tasks currently in the queue.

        Returns:
            bool: True if the queue contains tasks, False otherwise.
        """
        return len(self.queue) > 0

    def _save_queue(self):
        """
        Persists the current state of the task queue to the configured JSON file.
        Ensures the parent directory exists before writing.
        """
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.queue_path, 'w') as f:
            json.dump(list(self.queue), f, indent=4)
        log_json("INFO", "task_queue_saved", details={"path": str(self.queue_path), "queue_size": len(self.queue)})

    def _load_queue(self):
        """
        Loads the task queue from the configured JSON file.
        If the file does not exist or is corrupted, it initializes an empty queue.

        Returns:
            collections.deque: The loaded task queue.
        """
        if self.queue_path.exists():
            with open(self.queue_path, 'r') as f:
                try:
                    loaded_queue = deque(json.load(f))
                    log_json("INFO", "task_queue_loaded", details={"path": str(self.queue_path), "queue_size": len(loaded_queue)})
                    return loaded_queue
                except json.JSONDecodeError:
                    log_json("WARN", "task_queue_corrupted", details={"path": str(self.queue_path), "message": "Starting with empty queue."})
                    return deque()
        log_json("INFO", "task_queue_file_not_found", details={"path": str(self.queue_path), "message": "Initializing empty queue."})
        return deque()