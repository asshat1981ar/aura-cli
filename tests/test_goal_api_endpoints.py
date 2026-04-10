"""Tests for the FastAPI goal management REST endpoints in aura_cli/api_server.py.

Covers POST /api/goals, GET /api/goals, GET /api/goals/{id},
DELETE /api/goals/{id}, and POST /api/goals/{id}/prioritize.
"""

from __future__ import annotations

import json
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _make_client():
    from aura_cli.api_server import app

    return TestClient(app, raise_server_exceptions=True)


class TestGoalAPIModels(unittest.TestCase):
    def test_goal_create_requires_description(self):
        from aura_cli.api_server import GoalCreate
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            GoalCreate(description="", priority=1, max_cycles=5)

    def test_goal_create_defaults(self):
        from aura_cli.api_server import GoalCreate

        g = GoalCreate(description="test goal")
        self.assertEqual(g.priority, 1)
        self.assertEqual(g.max_cycles, 10)

    def test_goal_response_fields(self):
        from aura_cli.api_server import GoalResponse

        gr = GoalResponse(
            id="goal-q-0",
            description="test",
            status="pending",
            priority=1,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        self.assertEqual(gr.status, "pending")
        self.assertEqual(gr.progress, 0)

    def test_goal_detail_has_history(self):
        from aura_cli.api_server import GoalDetailResponse

        g = GoalDetailResponse(
            id="goal-q-0",
            description="test",
            status="pending",
            priority=1,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            history=[],
        )
        self.assertIsInstance(g.history, list)


class TestGoalListEndpoint(unittest.TestCase):
    def _client_with_empty_queue(self):
        client = _make_client()
        return client

    def test_get_goals_returns_list(self):
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": [], "in_flight": {}}), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_get_goals_with_queued_items(self):
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": ["goal A", "goal B"], "in_flight": {}}), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals")
        self.assertEqual(resp.status_code, 200)
        goals = resp.json()
        statuses = [g["status"] for g in goals]
        self.assertIn("pending", statuses)

    def test_get_goals_status_filter(self):
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": ["goal A"], "in_flight": {"running goal": 1000.0}}), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals?status=pending")
        self.assertEqual(resp.status_code, 200)
        for g in resp.json():
            self.assertEqual(g["status"], "pending")

    def test_get_goals_running_filter(self):
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": [], "in_flight": {"running goal": 1000.0}}), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals?status=running")
        self.assertEqual(resp.status_code, 200)
        for g in resp.json():
            self.assertEqual(g["status"], "running")


class TestGoalCreateEndpoint(unittest.TestCase):
    def test_post_goal_enqueues_and_returns_201(self):
        client = _make_client()
        mock_gq = MagicMock()
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals", json={"description": "New test goal"})
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["description"], "New test goal")
        self.assertEqual(body["status"], "pending")
        mock_gq.add.assert_called_once_with("New test goal")

    def test_post_goal_with_priority(self):
        client = _make_client()
        mock_gq = MagicMock()
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals", json={"description": "Priority goal", "priority": 3})
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["priority"], 3)

    def test_post_goal_empty_description_rejected(self):
        client = _make_client()
        resp = client.post("/api/goals", json={"description": "  "})
        self.assertEqual(resp.status_code, 422)

    def test_post_goal_missing_description_rejected(self):
        client = _make_client()
        resp = client.post("/api/goals", json={})
        self.assertEqual(resp.status_code, 422)

    def test_post_goal_queue_error_returns_500(self):
        client = _make_client()
        mock_gq = MagicMock()
        mock_gq.add.side_effect = OSError("disk full")
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals", json={"description": "New goal"})
        self.assertEqual(resp.status_code, 500)


class TestGoalDetailEndpoint(unittest.TestCase):
    def test_get_queued_goal_by_id(self):
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": ["my goal"], "in_flight": {}}), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals/goal-q-0")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["description"], "my goal")
        self.assertEqual(body["status"], "pending")
        self.assertIn("history", body)

    def test_get_inflight_goal_by_id(self):
        import time

        ts = time.time()
        desc = "inflight goal"
        goal_id = f"goal-f-{hash(desc) & 0xFFFFFFFF}"
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": [], "in_flight": {desc: ts}}), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get(f"/api/goals/{goal_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "running")

    def test_get_goal_not_found_returns_404(self):
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": [], "in_flight": {}}), patch("aura_cli.api_server.get_goal_archive", return_value=[]):
            resp = client.get("/api/goals/goal-q-999")
        self.assertEqual(resp.status_code, 404)

    def test_get_archived_goal_includes_history(self):
        desc = "archived goal"
        goal_id = f"goal-a-{hash(desc) & 0xFFFFFFFF}"
        archive = [{"goal": desc, "status": "completed", "timestamp": "2024-01-01T00:00:00", "cycles": 3, "history": [{"phase": "plan", "outcome": "ok"}]}]
        client = _make_client()
        with patch("aura_cli.api_server.get_goal_queue_data", return_value={"queue": [], "in_flight": {}}), patch("aura_cli.api_server.get_goal_archive", return_value=archive):
            resp = client.get(f"/api/goals/{goal_id}")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "completed")
        self.assertIsInstance(body["history"], list)


class TestGoalDeleteEndpoint(unittest.TestCase):
    def test_delete_pending_goal_removes_from_queue(self):
        client = _make_client()
        mock_gq = MagicMock()
        mock_gq.queue = deque(["goal A"])
        mock_gq.in_flight_keys.return_value = []
        mock_gq.cancel.return_value = "goal A"
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.delete("/api/goals/goal-q-0")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        mock_gq.cancel.assert_called_once_with(0)

    def test_delete_inflight_goal_returns_409(self):
        client = _make_client()
        desc = "running goal"
        mock_gq = MagicMock()
        mock_gq.queue = deque([])
        mock_gq.in_flight_keys.return_value = [desc]
        inflight_id = f"goal-f-{hash(desc) & 0xFFFFFFFF}"
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.delete(f"/api/goals/{inflight_id}")
        self.assertEqual(resp.status_code, 409)

    def test_delete_nonexistent_goal_returns_404(self):
        client = _make_client()
        mock_gq = MagicMock()
        mock_gq.queue = deque(["other goal"])
        mock_gq.in_flight_keys.return_value = []
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.delete("/api/goals/goal-q-999")
        self.assertEqual(resp.status_code, 404)

    def test_delete_unavailable_queue_returns_503(self):
        client = _make_client()
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", False):
            resp = client.delete("/api/goals/goal-q-0")
        self.assertEqual(resp.status_code, 503)


class TestGoalPrioritizeEndpoint(unittest.TestCase):
    def test_prioritize_moves_goal_to_front(self):
        client = _make_client()
        mock_gq = MagicMock()
        mock_gq.queue = deque(["goal A", "goal B", "goal C"])
        mock_gq.in_flight_keys.return_value = []
        mock_gq.promote.return_value = "goal B"
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals/goal-q-1/prioritize")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["position"], 0)
        mock_gq.promote.assert_called_once_with(1)

    def test_prioritize_already_front_returns_ok(self):
        client = _make_client()
        mock_gq = MagicMock()
        mock_gq.queue = deque(["goal A", "goal B"])
        mock_gq.in_flight_keys.return_value = []
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals/goal-q-0/prioritize")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["position"], 0)

    def test_prioritize_inflight_returns_409(self):
        client = _make_client()
        desc = "running goal"
        inflight_id = f"goal-f-{hash(desc) & 0xFFFFFFFF}"
        mock_gq = MagicMock()
        mock_gq.queue = deque([])
        mock_gq.in_flight_keys.return_value = [desc]
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post(f"/api/goals/{inflight_id}/prioritize")
        self.assertEqual(resp.status_code, 409)

    def test_prioritize_nonexistent_goal_returns_404(self):
        client = _make_client()
        mock_gq = MagicMock()
        mock_gq.queue = deque(["goal A"])
        mock_gq.in_flight_keys.return_value = []
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", True), patch("aura_cli.api_server.GoalQueue", return_value=mock_gq):
            resp = client.post("/api/goals/goal-q-999/prioritize")
        self.assertEqual(resp.status_code, 404)

    def test_prioritize_unavailable_queue_returns_503(self):
        client = _make_client()
        with patch("aura_cli.api_server.GOAL_QUEUE_AVAILABLE", False):
            resp = client.post("/api/goals/goal-q-0/prioritize")
        self.assertEqual(resp.status_code, 503)
