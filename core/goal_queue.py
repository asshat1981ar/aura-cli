import json
from collections import deque
from pathlib import Path # Import Path
from core.logging_utils import log_json # Import log_json

class GoalQueue:
    """
    Manages a persistent queue of goals for the AURA system. Goals are stored
    in a JSON file and loaded/saved automatically, ensuring work continuity
    across sessions.
    """

    def __init__(self, queue_path="memory/goal_queue.json"):
        """
        Initializes the GoalQueue.

        Args:
            queue_path (str): The path to the JSON file where the queue is persisted.
                              Defaults to "memory/goal_queue.json".
        """
        self.queue_path = Path(queue_path) # Convert to Path object
        self.queue = self._load_queue() # Load into a deque

    def add(self, goal):
        """
        Adds a new goal to the end of the queue.

        Args:
            goal: The goal to be added.
        """
        self.queue.append(goal)
        self._save_queue()

    def next(self):
        """
        Retrieves and removes the next goal from the front of the queue.

        Returns:
            The next goal in the queue, or None if the queue is empty.
        """
        if self.queue:
            goal = self.queue.popleft() # Optimized pop
            self._save_queue()
            return goal


    def has_goals(self):
        """
        Checks if there are any goals currently in the queue.

        Returns:
            bool: True if the queue contains goals, False otherwise.
        """
        return len(self.queue) > 0

    def _save_queue(self):
        """
        Persists the current state of the goal queue to the configured JSON file.
        Ensures the parent directory exists before writing.
        """
        self.queue_path.parent.mkdir(parents=True, exist_ok=True) # Use Path.parent and mkdir
        with open(self.queue_path, 'w') as f:
            json.dump(list(self.queue), f, indent=4) # Convert deque to list for JSON serialization

    def _load_queue(self):
        """
        Loads the goal queue from the configured JSON file.
        If the file does not exist or is corrupted, it initializes an empty queue.

        Returns:
            collections.deque: The loaded goal queue.
        """
        if self.queue_path.exists(): # Use Path.exists()
            with open(self.queue_path, 'r') as f:
                try:
                    return deque(json.load(f))
                except json.JSONDecodeError:
                    log_json("WARN", "goal_queue_corrupted", details={"path": str(self.queue_path), "message": "Starting with empty queue."})
                    return deque()
        return deque()

