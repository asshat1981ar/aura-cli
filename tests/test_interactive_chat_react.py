import unittest
import json
from unittest.mock import MagicMock
from pathlib import Path
import tempfile
import shutil

from aura_cli.dispatch.chat import interactive_chat

class TestInteractiveChatReAct(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_root = Path(self.test_dir)
        (self.project_root / "test_file.py").write_text("print('hello world')")
        
        self.model = MagicMock()
        self.vector_store = MagicMock()
        self.goal_queue = MagicMock()
        
        self.runtime = {
            "model_adapter": self.model,
            "vector_store": self.vector_store,
            "project_root": self.project_root,
            "goal_queue": self.goal_queue
        }
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_react_read_file(self):
        # Simulate LLM asking to read file, then replying
        self.model.respond.side_effect = [
            json.dumps({"action": "read_file", "path": "test_file.py"}),
            json.dumps({"action": "reply", "text": "The file contains print('hello world')"})
        ]
        
        history = []
        action = interactive_chat(self.runtime, "What is in test_file.py?", history)
        
        self.assertEqual(action, "continue")
        self.assertEqual(self.model.respond.call_count, 2)
        
        # Verify history structure
        # user -> assistant(action) -> system(content) -> assistant(reply)
        roles = [msg["role"] for msg in history]
        self.assertEqual(roles, ["user", "assistant", "system", "assistant"])
        self.assertIn("hello world", history[2]["content"])
        self.assertEqual(history[3]["content"], "The file contains print('hello world')")

    def test_react_search(self):
        # Simulate LLM asking to search, then replying
        self.vector_store.search.return_value = ["SearchResult: print('hello')"]
        
        self.model.respond.side_effect = [
            json.dumps({"action": "search", "query": "hello"}),
            json.dumps({"action": "reply", "text": "Found hello"})
        ]
        
        history = []
        action = interactive_chat(self.runtime, "Search for hello", history)
        
        self.assertEqual(action, "continue")
        self.assertEqual(self.model.respond.call_count, 2)
        
        roles = [msg["role"] for msg in history]
        self.assertEqual(roles, ["user", "assistant", "system", "assistant"])
        self.assertIn("SearchResult", history[2]["content"])

    def test_react_max_iterations(self):
        # Simulate LLM caught in a loop
        self.model.respond.return_value = json.dumps({"action": "search", "query": "loop"})
        
        history = []
        action = interactive_chat(self.runtime, "Trigger loop", history)
        
        self.assertEqual(action, "continue")
        self.assertEqual(self.model.respond.call_count, 5)

if __name__ == "__main__":
    unittest.main()
