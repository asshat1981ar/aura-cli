import json
import time
from collections import deque
from pathlib import Path # Import Path
from core.logging_utils import log_json # Import log_json

class GoalQueue:
    """
    Manages a persistent queue of goals for the AURA system. Goals are stored
    in a JSON file and loaded/saved automatically, ensuring work continuity
    across sessions.

    An in-flight tracker prevents silent goal loss on crash or interrupt: goals
    moved out of the queue by :meth:`next` are kept in ``_in_flight`` until
    :meth:`complete` or :meth:`fail` is called.  On startup, call
    :meth:`recover` to push any stale in-flight entries back to the front of
    the queue.
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
        data = self._load_data()
        self.queue = deque(data.get("queue", []))
        # _in_flight: dict mapping goal (str) → unix timestamp (float)
        self._in_flight: dict[str, float] = data.get("in_flight", {})

    # ------------------------------------------------------------------
    # Public queue operations
    # ------------------------------------------------------------------

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
        Retrieves the next goal from the front of the queue and moves it to
        the in-flight tracker instead of deleting it.

        The goal remains in ``_in_flight`` until the caller invokes
        :meth:`complete` (success) or :meth:`fail` (failure/retry).

        Returns:
            The next goal in the queue, or None if the queue is empty.
        """
        if self.queue:
            goal = self.queue.popleft()
            self._in_flight[goal] = time.time()
            self._save_queue()
            return goal

    def has_goals(self):
        """
        Checks if there are any goals currently in the queue.

        Returns:
            bool: True if the queue contains goals, False otherwise.
        """
        return len(self.queue) > 0

    def clear(self):
        """
        Clears all goals from the queue and persists.
        """
        self.queue.clear()
        self._save_queue()

    # ------------------------------------------------------------------
    # In-flight lifecycle methods
    # ------------------------------------------------------------------

    def complete(self, goal):
        """Remove a goal from the in-flight tracker after successful execution.

        Args:
            goal: The goal that finished successfully.
        """
        if goal in self._in_flight:
            del self._in_flight[goal]
            self._save_queue()
            log_json("INFO", "goal_completed", details={"goal": str(goal)[:120]})
        else:
            log_json("WARN", "goal_complete_not_inflight", details={"goal": str(goal)[:120]})

    def fail(self, goal):
        """Move a goal from in-flight back to the front of the queue for retry.

        Args:
            goal: The goal that failed and should be retried.
        """
        if goal in self._in_flight:
            del self._in_flight[goal]
        self.queue.appendleft(goal)
        self._save_queue()
        log_json("INFO", "goal_requeued_after_failure", details={"goal": str(goal)[:120]})

    def recover(self):
        """Restore any in-flight goals to the front of the queue.

        Call this once on startup after a crash or unclean shutdown to prevent
        silent goal loss.  Goals are prepended in their original insertion order
        (oldest first) so that work resumes naturally.

        Returns:
            int: Number of goals recovered.
        """
        if not self._in_flight:
            return 0

        # Sort by timestamp so oldest in-flight goal ends up first
        recovered = sorted(self._in_flight.keys(), key=lambda g: self._in_flight[g])
        count = len(recovered)
        self._in_flight = {}
        for goal in reversed(recovered):
            self.queue.appendleft(goal)
        self._save_queue()
        log_json("INFO", "goal_queue_recovered", details={"count": count})
        return count

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_queue(self):
        """
        Persists the current state of the goal queue and in-flight tracker to
        the configured JSON file.  Uses a compact encoding because this path is
        on the hot add/pop loop.
        """
        self.queue_path.write_text(
            json.dumps(
                {"queue": list(self.queue), "in_flight": self._in_flight},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    def _load_data(self) -> dict:
        """
        Loads queue state from the configured JSON file.

        Supports both the legacy format (plain JSON array) and the new format
        (dict with ``queue`` and ``in_flight`` keys) so that existing deployments
        are not broken on upgrade.

        Returns:
            dict with keys ``queue`` (list) and ``in_flight`` (dict).
        """
        if self.queue_path.exists():
            with self.queue_path.open("r", encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                except json.JSONDecodeError:
                    log_json(
                        "WARN",
                        "goal_queue_corrupted",
                        details={"path": str(self.queue_path), "message": "Starting with empty queue."},
                    )
                    return {"queue": [], "in_flight": {}}

            # Legacy format: plain list
            if isinstance(raw, list):
                return {"queue": raw, "in_flight": {}}

            # New format
            if isinstance(raw, dict):
                return {
                    "queue": raw.get("queue", []),
                    "in_flight": raw.get("in_flight", {}),
                }

        return {"queue": [], "in_flight": {}}

    # Keep private alias for any external callers that relied on _load_queue
    def _load_queue(self):
        """Backward-compatible shim — returns a deque of queued goals only."""
        return deque(self._load_data()["queue"])
