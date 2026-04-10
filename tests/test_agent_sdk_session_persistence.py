# tests/test_agent_sdk_session_persistence.py
"""Tests for session persistence and cost tracking."""

import tempfile
import unittest
from pathlib import Path


class TestSessionStore(unittest.TestCase):
    """Test SQLite session persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_sessions.db"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_session(self):
        from core.agent_sdk.session_persistence import SessionStore

        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("sdk-123", "Fix bug", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        self.assertIsInstance(pk, int)
        self.assertGreater(pk, 0)

    def test_get_session(self):
        from core.agent_sdk.session_persistence import SessionStore

        store = SessionStore(db_path=self.db_path)
        store.create_session("sdk-456", "Add feature", "feature", "feature", "claude-sonnet-4-6")
        session = store.get_session("sdk-456")
        self.assertIsNotNone(session)
        self.assertEqual(session["goal"], "Add feature")
        self.assertEqual(session["status"], "active")

    def test_update_status(self):
        from core.agent_sdk.session_persistence import SessionStore

        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("sdk-789", "Refactor", "refactor", "refactor", "claude-sonnet-4-6")
        store.update_status(pk, "completed")
        session = store.get_session("sdk-789")
        self.assertEqual(session["status"], "completed")

    def test_record_event(self):
        from core.agent_sdk.session_persistence import SessionStore

        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("sdk-e1", "Test", "default", "feature", "claude-sonnet-4-6")
        store.record_event(pk, "analyze_goal", "analyze_goal", "claude-sonnet-4-6", 100, 50, True, None)
        session = store.get_session("sdk-e1")
        self.assertGreater(session["total_input_tokens"], 0)

    def test_list_sessions(self):
        from core.agent_sdk.session_persistence import SessionStore

        store = SessionStore(db_path=self.db_path)
        store.create_session("s1", "Goal 1", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        store.create_session("s2", "Goal 2", "feature", "feature", "claude-sonnet-4-6")
        sessions = store.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_list_sessions_by_status(self):
        from core.agent_sdk.session_persistence import SessionStore

        store = SessionStore(db_path=self.db_path)
        pk1 = store.create_session("s1", "G1", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        store.create_session("s2", "G2", "feature", "feature", "claude-sonnet-4-6")
        store.update_status(pk1, "completed")
        active = store.list_sessions(status="active")
        self.assertEqual(len(active), 1)

    def test_get_resumable(self):
        from core.agent_sdk.session_persistence import SessionStore

        store = SessionStore(db_path=self.db_path)
        pk = store.create_session("s1", "Paused goal", "bug_fix", "bug_fix", "claude-sonnet-4-6")
        store.update_status(pk, "paused")
        resumable = store.get_resumable()
        self.assertEqual(len(resumable), 1)
        self.assertEqual(resumable[0]["session_id"], "s1")


class TestCostComputation(unittest.TestCase):
    """Test cost calculation utility."""

    def test_compute_cost_sonnet(self):
        from core.agent_sdk.session_persistence import compute_cost

        cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 18.0)  # 3 + 15

    def test_compute_cost_haiku(self):
        from core.agent_sdk.session_persistence import compute_cost

        cost = compute_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 6.0)  # 1 + 5

    def test_compute_cost_small_usage(self):
        from core.agent_sdk.session_persistence import compute_cost

        cost = compute_cost("claude-sonnet-4-6", 1000, 500)
        expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
        self.assertAlmostEqual(cost, expected)


if __name__ == "__main__":
    unittest.main()
