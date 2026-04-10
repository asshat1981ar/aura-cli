"""Tests for core/task_queue.py."""

import json
import tempfile
from pathlib import Path

import pytest

from core.task_queue import TaskQueue


class TestTaskQueue:
    """Test TaskQueue functionality."""

    def test_init_creates_empty_queue(self):
        """Test initialization creates empty queue when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))
            assert len(queue.queue) == 0
            assert not queue.has_tasks()

    def test_init_loads_existing_queue(self):
        """Test initialization loads existing queue from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"

            # Pre-populate file
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps(["task1", "task2"]))

            queue = TaskQueue(str(queue_path))
            assert len(queue.queue) == 2
            assert queue.has_tasks()

    def test_init_handles_corrupted_file(self):
        """Test initialization handles corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text("invalid json")

            queue = TaskQueue(str(queue_path))
            assert len(queue.queue) == 0

    def test_add_task(self):
        """Test adding task to queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            queue.add_task("task_data")
            assert len(queue.queue) == 1
            assert queue.has_tasks()

    def test_add_task_persists(self):
        """Test that add_task persists to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            queue.add_task("persisted_task")

            # Load fresh instance
            queue2 = TaskQueue(str(queue_path))
            assert len(queue2.queue) == 1
            assert queue2.queue[0] == "persisted_task"

    def test_get_next_task_fifo_order(self):
        """Test FIFO order of task retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            queue.add_task("first")
            queue.add_task("second")
            queue.add_task("third")

            assert queue.get_next_task() == "first"
            assert queue.get_next_task() == "second"
            assert queue.get_next_task() == "third"
            assert queue.get_next_task() is None

    def test_get_next_task_empty_queue(self):
        """Test getting task from empty queue returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            assert queue.get_next_task() is None

    def test_get_next_task_persists(self):
        """Test that get_next_task persists state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            queue.add_task("task1")
            queue.add_task("task2")
            queue.get_next_task()

            # Load fresh instance
            queue2 = TaskQueue(str(queue_path))
            assert len(queue2.queue) == 1
            assert queue2.queue[0] == "task2"

    def test_has_tasks(self):
        """Test has_tasks method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            assert not queue.has_tasks()
            queue.add_task("task")
            assert queue.has_tasks()
            queue.get_next_task()
            assert not queue.has_tasks()

    def test_complex_task_data(self):
        """Test queue handles complex task data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            complex_task = {
                "id": 123,
                "action": "process",
                "data": ["item1", "item2"],
                "nested": {"key": "value"},
            }

            queue.add_task(complex_task)
            retrieved = queue.get_next_task()

            assert retrieved == complex_task

    def test_multiple_tasks_persistence(self):
        """Test persistence with multiple add/remove operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "test_queue.json"
            queue = TaskQueue(str(queue_path))

            # Add tasks
            for i in range(5):
                queue.add_task(f"task_{i}")

            # Remove some
            queue.get_next_task()
            queue.get_next_task()

            # Add more
            queue.add_task("new_task")

            # Verify state
            queue2 = TaskQueue(str(queue_path))
            assert len(queue2.queue) == 4
            assert list(queue2.queue) == ["task_2", "task_3", "task_4", "new_task"]
