import json
import tempfile
from collections import deque
from pathlib import Path

from core.goal_archive import GoalArchive
from core.goal_queue import GoalQueue
from core.runtime_state import validate_brain_schema
from memory.brain import Brain


def test_goal_queue_restores_from_backup_when_primary_is_corrupted():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "goal_queue.json"
        path.write_text("{broken")
        path.with_suffix(".json.bak").write_text(json.dumps(["goal-a", "goal-b"]))

        queue = GoalQueue(queue_path=str(path))

    assert list(queue.queue) == ["goal-a", "goal-b"]


def test_goal_archive_restores_from_backup_when_primary_is_corrupted():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "goal_archive.json"
        path.write_text("{broken")
        path.with_suffix(".json.bak").write_text(json.dumps([["done", 1.0]]))

        archive = GoalArchive(archive_path=str(path))

    assert archive.completed == [["done", 1.0]]


def test_validate_brain_schema_reports_expected_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "brain.db"
        brain = Brain(db_path=str(db_path))
        try:
            validation = validate_brain_schema(db_path)
        finally:
            brain.db.close()

    assert validation["ok"] is True
    assert validation["missing_tables"] == []
