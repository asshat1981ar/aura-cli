"""Tests for MemoryStore._rotate_log_if_needed in memory/store.py."""
import json
import os
import tempfile
import unittest

from memory.store import MemoryStore, _LOG_MAX_BYTES, _LOG_KEEP_ROTATIONS


class TestStoreRotation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(root=self.tmpdir)
        # Expose module-level constants through the instance for convenience
        self.store._LOG_MAX_BYTES = _LOG_MAX_BYTES
        self.store._LOG_KEEP_ROTATIONS = _LOG_KEEP_ROTATIONS

    def _log_path(self) -> str:
        return str(self.store.log_path)

    def _log_rotation_path(self, n: int) -> str:
        return self._log_path() + f".{n}"

    def _seed_large_file(self, size_bytes: int) -> None:
        """Pre-seed the decision log with *size_bytes* of content."""
        with open(self._log_path(), "w") as f:
            f.write("x" * size_bytes)

    # ── No rotation below threshold ──────────────────────────────────────────

    def test_no_rotation_below_max_bytes(self):
        entry = {"event": "test", "status": "ok"}
        self.store.append_log(entry)
        self.assertFalse(os.path.exists(self._log_rotation_path(1)),
                         "No rotation file should exist below threshold")

    # ── Rotation at threshold ─────────────────────────────────────────────────

    def test_rotation_triggered_at_max_bytes(self):
        max_bytes = self.store._LOG_MAX_BYTES
        self._seed_large_file(max_bytes + 1)
        self.store.append_log({"event": "after_threshold"})
        self.assertTrue(os.path.exists(self._log_rotation_path(1)),
                        "Rotation file .1 should exist after exceeding threshold")

    def test_rotation_clears_current_log(self):
        max_bytes = self.store._LOG_MAX_BYTES
        self._seed_large_file(max_bytes + 1)
        self.store.append_log({"event": "fresh"})
        with open(self._log_path()) as f:
            content = f.read()
        self.assertIn("fresh", content)
        self.assertLess(len(content), max_bytes,
                        "Current log should be small after rotation")

    # ── Multiple rotations ────────────────────────────────────────────────────

    def test_multiple_rotations_shift_files(self):
        max_bytes = self.store._LOG_MAX_BYTES
        for i in range(3):
            self._seed_large_file(max_bytes + 1)
            self.store.append_log({"cycle": i})
        # After 3 rotations, .1, .2, .3 should all exist
        for n in range(1, self.store._LOG_KEEP_ROTATIONS + 1):
            self.assertTrue(os.path.exists(self._log_rotation_path(n)),
                            f"Rotation file .{n} should exist")

    def test_oldest_rotation_pruned(self):
        """After exceeding _LOG_KEEP_ROTATIONS, the oldest file is dropped."""
        max_bytes = self.store._LOG_MAX_BYTES
        keep = self.store._LOG_KEEP_ROTATIONS
        # Trigger keep+1 rotations so the oldest falls off
        for i in range(keep + 1):
            self._seed_large_file(max_bytes + 1)
            self.store.append_log({"cycle": i})
        beyond = self._log_rotation_path(keep + 1)
        self.assertFalse(os.path.exists(beyond),
                         f"Rotation file .{keep + 1} should not exist (pruned)")

    # ── Content integrity ────────────────────────────────────────────────────

    def test_appended_entries_are_valid_json_lines(self):
        entries = [{"step": i, "value": f"data_{i}"} for i in range(5)]
        for entry in entries:
            self.store.append_log(entry)
        with open(self._log_path()) as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            self.assertEqual(parsed["step"], i)

    # ── read_log limit behaviour ──────────────────────────────────────────────

    def test_read_log_no_limit_returns_all(self):
        """Default limit=0 must return every entry."""
        for i in range(10):
            self.store.append_log({"n": i})
        result = self.store.read_log()
        self.assertEqual(len(result), 10)

    def test_read_log_limit_returns_last_n(self):
        """Positive limit returns only the last N entries."""
        for i in range(10):
            self.store.append_log({"n": i})
        result = self.store.read_log(limit=3)
        self.assertEqual(len(result), 3)
        self.assertEqual([e["n"] for e in result], [7, 8, 9])

    def test_read_log_limit_zero_returns_all(self):
        """Explicit limit=0 still returns all entries (no truncation)."""
        for i in range(5):
            self.store.append_log({"n": i})
        result = self.store.read_log(limit=0)
        self.assertEqual(len(result), 5)

    def test_read_log_empty_log(self):
        """read_log on a missing log file returns an empty list."""
        result = self.store.read_log()
        self.assertEqual(result, [])
