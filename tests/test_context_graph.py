import tempfile
import unittest
from pathlib import Path

from core.context_graph import ContextGraph


class TestContextGraphSqliteConnection(unittest.TestCase):
    def test_context_graph_initializes_and_can_write_edge(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "context_graph.db"
            graph = ContextGraph(db_path=db_path)
            graph.add_edge("core/task_handler.py", "tests/test_task_handler.py", "related_to", weight=1.0)

            rows = graph.goals_touching_file("core/task_handler.py")
            db_exists = db_path.exists()

        self.assertEqual(rows, [])
        self.assertTrue(db_exists)


if __name__ == "__main__":
    unittest.main()
