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
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = self._load_queue() # Load into a deque

    def add(self, goal):
        """
        Adds a new goal to the end of the queue and persists immediately.

        For adding multiple goals at once, prefer :meth:`batch_add` to avoid
        N separate JSON flushes (each ~0.33ms on Termux/Android).

        Args:
            goal: The goal to be added.
        """
        self.queue.append(goal)
        self._save_queue()

    def batch_add(self, goals):
        """Add multiple goals with a single JSON flush.

        Significantly faster than calling :meth:`add` in a loop: N goals require
        only 1 disk write instead of N.  Benchmark: 30 goals → 0.33ms vs 9.9ms
        (30x speedup).

        Args:
            goals: Iterable of goal objects to enqueue.
        """
        for goal in goals:
            self.queue.append(goal)
        self._save_queue()

    def prepend_batch(self, goals):
        """Add multiple goals to the front of the queue while preserving order."""
        for goal in reversed(list(goals)):
            self.queue.appendleft(goal)
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


    def contains(self, goal) -> bool:
        """Return ``True`` if *goal* is already present in the queue.

        Uses an exact-equality check.  Prefer :meth:`add_if_absent` when you
        want to enqueue a goal only when it is not already pending.

        Args:
            goal: The goal object to search for.

        Returns:
            ``True`` when the goal is already in the queue, ``False`` otherwise.
        """
        return goal in self.queue

    def add_if_absent(self, goal) -> bool:
        """Add *goal* to the queue only if an identical goal is not already pending.

        This prevents the same remediation or follow-up goal from being queued
        multiple times across successive cycles, avoiding redundant work.

        Args:
            goal: The goal to conditionally enqueue.

        Returns:
            ``True`` when the goal was added, ``False`` when it was skipped
            because a duplicate was found.
        """
        if self.contains(goal):
            log_json("DEBUG", "goal_queue_duplicate_skipped",
                     details={"goal_preview": str(goal)[:120]})
            return False
        self.add(goal)
        return True

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
        Uses a compact encoding because this path is on the hot add/pop loop.
        """
        self.queue_path.write_text(
            json.dumps(list(self.queue), separators=(",", ":")),
            encoding="utf-8",
        )

    def _load_queue(self):
        """
        Loads the goal queue from the configured JSON file.
        If the file does not exist or is corrupted, it initializes an empty queue.

        Returns:
            collections.deque: The loaded goal queue.
        """
        if self.queue_path.exists(): # Use Path.exists()
            with self.queue_path.open('r', encoding='utf-8') as f:
                try:
                    return deque(json.load(f))
                except json.JSONDecodeError:
                    log_json("WARN", "goal_queue_corrupted", details={"path": str(self.queue_path), "message": "Starting with empty queue."})
                    return deque()
        return deque()
