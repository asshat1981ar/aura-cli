import unittest
from unittest.mock import MagicMock
from agents.code_search_agent import CodeSearchAgent

class TestCodeSearchAgent(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock()
        self.agent = CodeSearchAgent(vector_store=self.mock_vector_store)

    def test_query_no_vector_store(self):
        agent_no_store = CodeSearchAgent(vector_store=None)
        results = agent_no_store.query("def test")
        self.assertEqual(results, [])

    def test_query_success(self):
        self.mock_vector_store.search.return_value = [
            {"content": "def test(): pass", "score": 0.9, "metadata": {"file": "test.py"}},
            {"content": "class A:", "score": 0.4, "metadata": {"file": "a.py"}}
        ]
        
        results = self.agent.query("test", min_score=0.5)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["file"], "test.py")
        self.assertEqual(results[0]["content"], "def test(): pass")
        self.assertEqual(results[0]["score"], 0.9)

    def test_query_exception_handling(self):
        self.mock_vector_store.search.side_effect = Exception("DB connection lost")
        results = self.agent.query("test")
        self.assertEqual(results, [])

    def test_refine_query(self):
        refined = self.agent.refine_query("IndexError")
        self.assertIn("IndexError", refined)

if __name__ == '__main__':
    unittest.main()
