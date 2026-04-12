"""Thread-safety tests for memory/brain.py — covers issue #328."""

import threading
import tempfile
import unittest
from pathlib import Path


class TestBrainThreadSafety(unittest.TestCase):
    """Verify that Brain serialises concurrent DB writes via a threading.Lock."""

    def _make_brain(self, tmp_dir: str) -> object:
        from memory.brain import Brain

        return Brain(db_path=str(Path(tmp_dir) / "brain_test.db"))

    def test_lock_attribute_exists(self):
        """Brain.__init__ must create a _lock attribute that is a threading.Lock."""
        with tempfile.TemporaryDirectory() as tmp:
            brain = self._make_brain(tmp)
            self.assertTrue(
                hasattr(brain, "_lock"),
                "Brain must have a '_lock' attribute after __init__",
            )
            # Accept both Lock and RLock
            self.assertTrue(
                isinstance(brain._lock, type(threading.Lock())) or isinstance(brain._lock, type(threading.RLock())),
                "_lock must be a threading.Lock or threading.RLock instance",
            )

    def test_context_manager_method_exists(self):
        """Brain must expose a _db_lock() context-manager method."""
        with tempfile.TemporaryDirectory() as tmp:
            brain = self._make_brain(tmp)
            self.assertTrue(
                hasattr(brain, "_db_lock"),
                "Brain must have a '_db_lock' method",
            )
            # Must be usable as a context manager
            with brain._db_lock():
                pass  # no exception expected

    def test_concurrent_writes_from_two_threads(self):
        """Two threads writing simultaneously must not corrupt data or raise."""
        with tempfile.TemporaryDirectory() as tmp:
            brain = self._make_brain(tmp)
            errors: list[Exception] = []
            write_count = 50

            def writer(prefix: str) -> None:
                try:
                    for i in range(write_count):
                        brain.remember(f"{prefix}-entry-{i}")
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=writer, args=("thread-A",))
            t2 = threading.Thread(target=writer, args=("thread-B",))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            self.assertFalse(
                errors,
                f"Concurrent writes raised exceptions: {errors}",
            )

            # All writes must be durably stored
            total = brain.count_memories()
            self.assertEqual(
                total,
                write_count * 2,
                f"Expected {write_count * 2} entries, got {total}",
            )

    def test_concurrent_writes_and_reads(self):
        """Interleaved reads and writes from multiple threads must be consistent."""
        with tempfile.TemporaryDirectory() as tmp:
            brain = self._make_brain(tmp)
            # Pre-populate a few entries
            for i in range(10):
                brain.remember(f"seed-{i}")

            errors: list[Exception] = []

            def writer() -> None:
                try:
                    for i in range(30):
                        brain.remember(f"write-{i}")
                except Exception as exc:
                    errors.append(exc)

            def reader() -> None:
                try:
                    for _ in range(30):
                        brain.recall_recent(limit=20)
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=reader),
                threading.Thread(target=writer),
                threading.Thread(target=reader),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            self.assertFalse(
                errors,
                f"Concurrent read/write raised exceptions: {errors}",
            )


if __name__ == "__main__":
    unittest.main()
