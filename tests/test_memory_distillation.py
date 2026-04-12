"""Tests for memory.controller — distill() and tier_stats() methods."""

import unittest
from unittest.mock import patch

from memory.controller import MemoryController, MemoryTier


class TestDistillEmptySession(unittest.TestCase):
    """Distill with no session entries is a no-op."""

    def test_distill_empty_returns_zero(self):
        mc = MemoryController()
        result = mc.distill()
        self.assertEqual(result, 0)

    def test_distill_empty_no_project_entries_added(self):
        mc = MemoryController()
        mc.distill()
        self.assertEqual(len(mc.tiers[MemoryTier.PROJECT]), 0)


class TestDistillPromotesSessionToProject(unittest.TestCase):
    """Distill promotes session entries to the project tier."""

    def test_promotes_entries(self):
        mc = MemoryController()
        mc.store(MemoryTier.SESSION, "fact_a", {"source": "test"})
        mc.store(MemoryTier.SESSION, "fact_b", {"source": "test"})

        distilled = mc.distill()
        self.assertEqual(distilled, 2)

        # Both should now be in PROJECT tier (in-memory)
        project_contents = [e.content for e in mc.tiers[MemoryTier.PROJECT]]
        self.assertIn("fact_a", project_contents)
        self.assertIn("fact_b", project_contents)


class TestDistillDeduplication(unittest.TestCase):
    """Distill deduplicates entries within the session batch."""

    def test_deduplicates_within_session(self):
        mc = MemoryController()
        mc.store(MemoryTier.SESSION, "duplicate_fact")
        mc.store(MemoryTier.SESSION, "duplicate_fact")
        mc.store(MemoryTier.SESSION, "unique_fact")

        distilled = mc.distill()
        # 'duplicate_fact' should only be promoted once
        self.assertEqual(distilled, 2)


class TestDistillSkipsExistingProject(unittest.TestCase):
    """Distill skips entries already present in the project tier."""

    def test_skips_existing(self):
        mc = MemoryController()
        # Pre-populate project tier
        mc.store(MemoryTier.PROJECT, "already_known")

        # Add to session
        mc.store(MemoryTier.SESSION, "already_known")
        mc.store(MemoryTier.SESSION, "new_fact")

        distilled = mc.distill()
        self.assertEqual(distilled, 1)  # Only 'new_fact' should be distilled


class TestDistillMaxEntries(unittest.TestCase):
    """Distill respects max_entries parameter."""

    def test_respects_max_entries(self):
        mc = MemoryController()
        for i in range(10):
            mc.store(MemoryTier.SESSION, f"fact_{i}")

        distilled = mc.distill(max_entries=5)
        # Should only review the last 5 session entries
        self.assertLessEqual(distilled, 5)


class TestTierStats(unittest.TestCase):
    """Test tier_stats() method."""

    def test_empty_stats(self):
        mc = MemoryController()
        stats = mc.tier_stats()
        self.assertEqual(stats, {"working": 0, "session": 0, "project": 0})

    def test_stats_after_stores(self):
        mc = MemoryController()
        mc.store(MemoryTier.WORKING, "w1")
        mc.store(MemoryTier.SESSION, "s1")
        mc.store(MemoryTier.SESSION, "s2")
        mc.store(MemoryTier.PROJECT, "p1")
        mc.store(MemoryTier.PROJECT, "p2")
        mc.store(MemoryTier.PROJECT, "p3")

        stats = mc.tier_stats()
        self.assertEqual(stats["working"], 1)
        self.assertEqual(stats["session"], 2)
        self.assertEqual(stats["project"], 3)

    def test_stats_after_distill(self):
        mc = MemoryController()
        mc.store(MemoryTier.SESSION, "distill_me")

        before = mc.tier_stats()
        self.assertEqual(before["session"], 1)
        self.assertEqual(before["project"], 0)

        mc.distill()

        after = mc.tier_stats()
        # Session still has the original entry (distill does not remove)
        self.assertEqual(after["session"], 1)
        # Project now has the distilled entry
        self.assertEqual(after["project"], 1)


if __name__ == "__main__":
    unittest.main()
