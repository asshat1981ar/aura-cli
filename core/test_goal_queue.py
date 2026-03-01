import unittest
import os
import json
from pathlib import Path
from core.goal_queue import GoalQueue

class TestGoalQueue(unittest.TestCase):

    def setUp(self):
        self.test_queue_path = Path(__file__).parent / "test_goal_queue.json"
        if self.test_queue_path.exists():
            os.remove(self.test_queue_path)
        # Instantiate GoalQueue with the test JSON file path
        self.goal_queue = GoalQueue(queue_path=self.test_queue_path)

    def tearDown(self):
        if self.test_queue_path.exists():
            os.remove(self.test_queue_path)

    def test_init_and_load_empty(self):
        self.assertFalse(self.goal_queue.has_goals())
        self.assertEqual(len(self.goal_queue.queue), 0)

    def test_add_goal(self):
        self.goal_queue.add("Implement feature A")
        self.assertTrue(self.goal_queue.has_goals())
        self.assertEqual(len(self.goal_queue.queue), 1)
        self.assertEqual(self.goal_queue.queue[0], "Implement feature A")

        # Verify in JSON file
        with open(self.test_queue_path, 'r') as f:
            content = json.load(f)
        self.assertEqual(content, ["Implement feature A"])

    def test_next_goal(self):
        self.goal_queue.add("Implement feature A")
        self.goal_queue.add("Fix bug B")
        
        next_goal = self.goal_queue.next()
        self.assertEqual(next_goal, "Implement feature A")
        self.assertEqual(len(self.goal_queue.queue), 1)
        self.assertEqual(self.goal_queue.queue[0], "Fix bug B")

        # Verify in JSON file
        with open(self.test_queue_path, 'r') as f:
            content = json.load(f)
        self.assertEqual(content, ["Fix bug B"])


    def test_next_with_empty_queue(self):
        self.assertIsNone(self.goal_queue.next())
        self.assertFalse(self.goal_queue.has_goals())

    def test_persistence(self):
        self.goal_queue.add("Goal 1 for persistence")
        self.goal_queue.add("Goal 2 for persistence")
        
        # Simulate reopening the queue
        new_queue = GoalQueue(queue_path=self.test_queue_path)

        self.assertTrue(new_queue.has_goals())
        self.assertEqual(len(new_queue.queue), 2)
        self.assertEqual(new_queue.queue[0], "Goal 1 for persistence")
        self.assertEqual(new_queue.queue[1], "Goal 2 for persistence")

        next_goal = new_queue.next()
        self.assertEqual(next_goal, "Goal 1 for persistence")

        # Simulate reopening again after a pop
        final_queue = GoalQueue(queue_path=self.test_queue_path)
        self.assertEqual(len(final_queue.queue), 1)
        self.assertEqual(final_queue.queue[0], "Goal 2 for persistence")

    def test_prepend_batch_preserves_order(self):
        self.goal_queue.add("Existing goal")
        self.goal_queue.prepend_batch(["First priority", "Second priority"])

        self.assertEqual(
            list(self.goal_queue.queue),
            ["First priority", "Second priority", "Existing goal"],
        )

        with open(self.test_queue_path, 'r') as f:
            content = json.load(f)
        self.assertEqual(content, ["First priority", "Second priority", "Existing goal"])

if __name__ == '__main__':
    unittest.main()
