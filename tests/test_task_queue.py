import unittest
import json
from collections import deque
from pathlib import Path
from unittest.mock import patch, mock_open

from core.task_queue import TaskQueue

class TestTaskQueue(unittest.TestCase):

    def setUp(self):
        self.test_queue_path = Path("memory/test_task_queue.json")
        # Ensure a clean state for each test
        if self.test_queue_path.exists():
            self.test_queue_path.unlink()
        self.task_queue = TaskQueue(queue_path=self.test_queue_path)

    def tearDown(self):
        # Clean up the test file after each test
        if self.test_queue_path.exists():
            self.test_queue_path.unlink()

    def test_add_task_and_get_next_task(self):
        task1 = {"id": 1, "name": "Task 1"}
        task2 = {"id": 2, "name": "Task 2"}

        self.task_queue.add_task(task1)
        self.task_queue.add_task(task2)

        self.assertTrue(self.task_queue.has_tasks())
        self.assertEqual(self.task_queue.get_next_task(), task1)
        self.assertEqual(self.task_queue.get_next_task(), task2)
        self.assertFalse(self.task_queue.has_tasks())
        self.assertIsNone(self.task_queue.get_next_task())

    def test_persistence(self):
        task1 = {"id": 1, "name": "Task 1"}
        task2 = {"id": 2, "name": "Task 2"}

        self.task_queue.add_task(task1)
        self.task_queue.add_task(task2)

        # Re-initialize the queue to simulate loading from disk
        new_task_queue = TaskQueue(queue_path=self.test_queue_path)
        self.assertTrue(new_task_queue.has_tasks())
        self.assertEqual(new_task_queue.get_next_task(), task1)
        self.assertEqual(new_task_queue.get_next_task(), task2)
        self.assertFalse(new_task_queue.has_tasks())

    def test_empty_queue(self):
        self.assertFalse(self.task_queue.has_tasks())
        self.assertIsNone(self.task_queue.get_next_task())

    def test_corrupted_queue_file(self):
        # Create a corrupted JSON file
        self.test_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.test_queue_path, 'w') as f:
            f.write("this is not json")

        with patch('core.task_queue.log_json') as mock_log_json:
            corrupted_task_queue = TaskQueue(queue_path=self.test_queue_path)
            self.assertFalse(corrupted_task_queue.has_tasks())
            mock_log_json.assert_called_with("WARN", "task_queue_corrupted", details={"path": str(self.test_queue_path), "message": "Starting with empty queue."})

    @patch('core.task_queue.log_json')
    def test_logging(self, mock_log_json):
        task = {"id": 1, "name": "Logged Task"}
        self.task_queue.add_task(task)
        mock_log_json.assert_any_call("INFO", "task_added", details={"task": task, "queue_size": 1})

        self.task_queue.get_next_task()
        mock_log_json.assert_any_call("INFO", "task_retrieved", details={"task": task, "queue_size": 0})

        # Test for no tasks in queue
        self.task_queue.get_next_task()
        mock_log_json.assert_any_call("INFO", "no_tasks_in_queue")

        # Test initial load with no file
        if self.test_queue_path.exists():
            self.test_queue_path.unlink() # Ensure file doesn't exist
        TaskQueue(queue_path=self.test_queue_path)
        mock_log_json.assert_any_call("INFO", "task_queue_file_not_found", details={"path": str(self.test_queue_path), "message": "Initializing empty queue."})

if __name__ == '__main__':
    unittest.main()