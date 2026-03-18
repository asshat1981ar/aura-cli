import unittest
import os
import tempfile
from pathlib import Path
from memory.brain import Brain

class TestBrainKV(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "brain.db")
        self.brain = Brain(self.db_path)

    def tearDown(self):
        self.brain.db.close()
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_kv_set_get_string(self):
        self.brain.set("test_key", "test_value")
        self.assertEqual(self.brain.get("test_key"), "test_value")

    def test_kv_set_get_dict(self):
        data = {"a": 1, "b": [1, 2, 3]}
        self.brain.set("test_dict", data)
        self.assertEqual(self.brain.get("test_dict"), data)

    def test_kv_get_default(self):
        self.assertEqual(self.brain.get("nonexistent", "default"), "default")

    def test_kv_overwrite(self):
        self.brain.set("key", "val1")
        self.brain.set("key", "val2")
        self.assertEqual(self.brain.get("key"), "val2")

    def test_kv_persistence(self):
        self.brain.set("persist_key", "persist_val")
        self.brain.db.close()
        
        # Re-open
        new_brain = Brain(self.db_path)
        self.assertEqual(new_brain.get("persist_key"), "persist_val")
        new_brain.db.close()

if __name__ == "__main__":
    unittest.main()
