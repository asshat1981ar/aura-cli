import os
import shutil
import tempfile
import unittest
import sys

# Add the project root to sys.path to allow importing from scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.generate_capability_index import parse_retros, generate_index, get_existing_ids


class TestGenerateCapabilityIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.retro_dir = os.path.join(self.test_dir, "retros")
        os.makedirs(self.retro_dir)
        self.index_file = os.path.join(self.test_dir, "index.yaml")

        # Create a mock retro file
        self.retro_content = """# Retro - S001
## Capability Deltas Proposed

- **AF-DELTA-1234**: Fix the things
- **AF-DELTA-5678**: Add more things
"""
        with open(os.path.join(self.retro_dir, "S001_retro.md"), "w") as f:
            f.write(self.retro_content)

        # Create a mock index file
        self.index_content = """# Index
proposed:
  - id: AF-DELTA-0000
    title: "Old thing"
    source: "S000 retro"
under_review: []
"""
        with open(self.index_file, "w") as f:
            f.write(self.index_content)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parse_retros(self):
        deltas = parse_retros(self.retro_dir)
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[0]["id"], "AF-DELTA-1234")
        self.assertEqual(deltas[0]["title"], "Fix the things")
        self.assertEqual(deltas[0]["source"], "S001 retro")

    def test_get_existing_ids(self):
        ids = get_existing_ids(self.index_content)
        self.assertIn("AF-DELTA-0000", ids)

    def test_generate_index_idempotent(self):
        # Run first time
        deltas = parse_retros(self.retro_dir)
        generate_index(self.index_file, deltas)

        with open(self.index_file, "r") as f:
            content1 = f.read()

        self.assertIn("AF-DELTA-1234", content1)
        self.assertIn("AF-DELTA-5678", content1)

        # Run second time
        generate_index(self.index_file, deltas)
        with open(self.index_file, "r") as f:
            content2 = f.read()

        self.assertEqual(content1, content2)

    def test_generate_index_with_existing_delta_in_under_review(self):
        # Move AF-DELTA-1234 to under_review manually
        self.index_content = """# Index
proposed: []
under_review:
  - id: AF-DELTA-1234
    title: "Fix the things"
"""
        with open(self.index_file, "w") as f:
            f.write(self.index_content)

        deltas = parse_retros(self.retro_dir)
        generate_index(self.index_file, deltas)

        with open(self.index_file, "r") as f:
            content = f.read()

        # AF-DELTA-1234 should NOT be in proposed: [] section
        # We check that it didn't get added twice or added back to proposed
        self.assertEqual(content.count("AF-DELTA-1234"), 1)
        self.assertIn("AF-DELTA-5678", content)


if __name__ == "__main__":
    unittest.main()
