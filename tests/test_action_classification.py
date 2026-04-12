"""Tests for core.sanitizer — action classification (inbound/outbound)."""

import unittest

from core.sanitizer import (
    ActionCategory,
    classify_action,
    is_safe_action,
    _INBOUND_ACTIONS,
    _OUTBOUND_ACTIONS,
)


class TestClassifyAction(unittest.TestCase):
    """Test classify_action for known and unknown actions."""

    def test_inbound_actions(self):
        for action in ("read", "search", "analyze", "list", "query", "inspect",
                       "grep", "cat", "ls", "find", "stat",
                       "git_log", "git_status", "git_diff",
                       "memory_search", "memory_stats",
                       "health_check", "doctor", "diag",
                       "config_list", "mcp_tools"):
            with self.subTest(action=action):
                self.assertEqual(classify_action(action), ActionCategory.INBOUND,
                                 f"{action} should be INBOUND")

    def test_outbound_actions(self):
        for action in ("write", "apply", "commit", "push", "deploy", "send",
                       "delete", "rm", "mv",
                       "git_commit", "git_push",
                       "file_write", "file_delete",
                       "goal_add", "goal_run",
                       "credential_migrate"):
            with self.subTest(action=action):
                self.assertEqual(classify_action(action), ActionCategory.OUTBOUND,
                                 f"{action} should be OUTBOUND")

    def test_unknown_defaults_to_outbound(self):
        self.assertEqual(classify_action("hack_the_planet"), ActionCategory.OUTBOUND)
        self.assertEqual(classify_action(""), ActionCategory.OUTBOUND)
        self.assertEqual(classify_action("some_new_action"), ActionCategory.OUTBOUND)

    def test_case_insensitivity(self):
        self.assertEqual(classify_action("READ"), ActionCategory.INBOUND)
        self.assertEqual(classify_action("Read"), ActionCategory.INBOUND)
        self.assertEqual(classify_action("WRITE"), ActionCategory.OUTBOUND)
        self.assertEqual(classify_action("Write"), ActionCategory.OUTBOUND)

    def test_whitespace_tolerance(self):
        self.assertEqual(classify_action("  read  "), ActionCategory.INBOUND)
        self.assertEqual(classify_action("  write  "), ActionCategory.OUTBOUND)


class TestIsSafeAction(unittest.TestCase):
    """Test is_safe_action convenience function."""

    def test_safe_for_inbound(self):
        self.assertTrue(is_safe_action("read"))
        self.assertTrue(is_safe_action("ls"))
        self.assertTrue(is_safe_action("git_status"))
        self.assertTrue(is_safe_action("doctor"))

    def test_unsafe_for_outbound(self):
        self.assertFalse(is_safe_action("write"))
        self.assertFalse(is_safe_action("git_push"))
        self.assertFalse(is_safe_action("delete"))

    def test_unsafe_for_unknown(self):
        self.assertFalse(is_safe_action("unknown_thing"))


class TestActionSets(unittest.TestCase):
    """Verify that the inbound and outbound sets are disjoint."""

    def test_no_overlap(self):
        overlap = _INBOUND_ACTIONS & _OUTBOUND_ACTIONS
        self.assertEqual(overlap, frozenset(),
                         f"Inbound and outbound should be disjoint, overlap: {overlap}")


if __name__ == "__main__":
    unittest.main()
