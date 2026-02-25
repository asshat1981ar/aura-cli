import json
import os
from pathlib import Path # Import Path
from core.logging_utils import log_json # Import log_json

class GoalArchive:
    """
    Manages a persistent archive of completed goals for the AURA system.
    Completed goals and their scores are stored in a JSON file, providing a
    record of past achievements and performance.
    """

    def __init__(self, archive_path="memory/goal_archive.json"):
        """
        Initializes the GoalArchive.

        Args:
            archive_path (str): The path to the JSON file where the archive is persisted.
                                Defaults to "memory/goal_archive.json".
        """
        self.archive_path = Path(archive_path) # Convert to Path object
        self.completed = self._load_archive()

    def record(self, goal, score):
        """
        Records a completed goal along with its score.

        Args:
            goal: The completed goal to record.
            score: The score associated with the completed goal.
        """
        self.completed.append((goal, score))
        self._save_archive()

    def _save_archive(self):
        """
        Persists the current state of the goal archive to the configured JSON file.
        Ensures the parent directory exists before writing.
        """
        self.archive_path.parent.mkdir(parents=True, exist_ok=True) # Use Path.parent and mkdir
        with open(self.archive_path, 'w') as f:
            json.dump(self.completed, f, indent=4)

    def _load_archive(self):
        """
        Loads the goal archive from the configured JSON file.
        If the file does not exist or is corrupted, it initializes an empty archive.

        Returns:
            list: The loaded list of completed goals and their scores.
        """
        if self.archive_path.exists(): # Use Path.exists()
            with open(self.archive_path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    log_json("WARN", "goal_archive_corrupted", details={"path": str(self.archive_path), "message": "Starting with empty archive."})
                    return []
        return []
        if self.archive_path.exists(): # Use Path.exists()
            with open(self.archive_path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {self.archive_path}. Starting with empty archive.")
                    return []
        return []
