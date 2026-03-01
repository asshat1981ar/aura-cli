"""Tests for tools/aura_control_mcp.py."""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures — patch heavy singletons before importing the app
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons between tests."""
    import tools.aura_control_mcp as mod
    mod._goal_queue = None
    mod._goal_archive = None
    mod._brain = None
    mod._memory_cache.clear()
    yield
    mod._goal_queue = None
    mod._goal_archive = None
    mod._brain = None
    mod._memory_cache.clear()


@pytest.fixture()
def mock_queue():
    q = MagicMock()
    q.queue = deque(["Goal A", "Goal B", "Goal C"])
    q._save_queue = MagicMock()
    return q


@pytest.fixture()
def mock_archive():
    a = MagicMock()
    a._load_archive = MagicMock(return_value=[
        {"goal": "Old goal", "score": 0.9},
        {"goal": "Another goal", "score": 0.7},
    ])
    return a


@pytest.fixture()
def mock_brain():
    b = MagicMock()
    _memories = ["memory about Python", "memory about tests"]
    b.recall_all = MagicMock(return_value=_memories)
    b.recall_with_budget = MagicMock(return_value=_memories)
    b.count_memories = MagicMock(return_value=len(_memories))
    b.recall_weaknesses = MagicMock(return_value=["sometimes forgets context"])
    b.remember = MagicMock()
    return b


@pytest.fixture()
def client(mock_queue, mock_archive, mock_brain):
    import tools.aura_control_mcp as mod
    mod._goal_queue = mock_queue
    mod._goal_archive = mock_archive
    mod._brain = mock_brain
    from tools.aura_control_mcp import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tool_count"] == 11


# ---------------------------------------------------------------------------
# /tools
# ---------------------------------------------------------------------------

class TestListTools:
    def test_returns_all_tools(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        tools = resp.json()
        names = {t["name"] for t in tools}
        expected = {
            "goal_add", "goal_list", "goal_remove", "goal_clear", "goal_archive_list",
            "memory_search", "memory_add", "memory_weaknesses",
            "file_read", "file_list", "project_status",
        }
        assert expected == names

    def test_tool_has_schema(self, client):
        resp = client.get("/tools")
        for tool in resp.json():
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


class TestGetTool:
    def test_get_existing_tool(self, client):
        resp = client.get("/tool/goal_add")
        assert resp.status_code == 200
        assert resp.json()["name"] == "goal_add"

    def test_get_missing_tool(self, client):
        resp = client.get("/tool/nonexistent_tool")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /call — goal management
# ---------------------------------------------------------------------------

class TestGoalAdd:
    def test_add_goal(self, client, mock_queue):
        resp = client.post("/call", json={"tool_name": "goal_add", "args": {"text": "Fix the bug"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert data["result"]["added"] == "Fix the bug"
        mock_queue.add.assert_called_once_with("Fix the bug")

    def test_add_empty_goal_returns_error(self, client):
        resp = client.post("/call", json={"tool_name": "goal_add", "args": {"text": ""}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None

    def test_add_missing_text_returns_error(self, client):
        resp = client.post("/call", json={"tool_name": "goal_add", "args": {}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None


class TestGoalList:
    def test_list_goals(self, client, mock_queue):
        resp = client.post("/call", json={"tool_name": "goal_list", "args": {}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["count"] == 3
        assert "Goal A" in result["goals"]


class TestGoalRemove:
    def test_remove_valid_index(self, client, mock_queue):
        resp = client.post("/call", json={"tool_name": "goal_remove", "args": {"index": 1}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert data["result"]["removed"] == "Goal B"

    def test_remove_out_of_range(self, client):
        resp = client.post("/call", json={"tool_name": "goal_remove", "args": {"index": 99}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None

    def test_remove_missing_index(self, client):
        resp = client.post("/call", json={"tool_name": "goal_remove", "args": {}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None


class TestGoalClear:
    def test_clear_queue(self, client, mock_queue):
        resp = client.post("/call", json={"tool_name": "goal_clear", "args": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert data["result"]["cleared"] == 3


class TestGoalArchiveList:
    def test_list_archive(self, client):
        resp = client.post("/call", json={"tool_name": "goal_archive_list", "args": {}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["count"] == 2

    def test_list_archive_with_limit(self, client):
        resp = client.post("/call", json={"tool_name": "goal_archive_list", "args": {"limit": 1}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["count"] <= 1


# ---------------------------------------------------------------------------
# /call — memory tools
# ---------------------------------------------------------------------------

class TestMemorySearch:
    def test_search_with_match(self, client, mock_brain):
        resp = client.post("/call", json={"tool_name": "memory_search", "args": {"query": "Python"}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["total_matched"] >= 1
        assert any("Python" in m for m in result["matches"])

    def test_search_no_match(self, client):
        resp = client.post("/call", json={"tool_name": "memory_search", "args": {"query": "zzz_no_match_xyz"}})
        assert resp.status_code == 200
        assert resp.json()["result"]["total_matched"] == 0

    def test_search_missing_query(self, client):
        resp = client.post("/call", json={"tool_name": "memory_search", "args": {}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None


class TestMemoryAdd:
    def test_add_memory(self, client, mock_brain):
        resp = client.post("/call", json={"tool_name": "memory_add", "args": {"text": "New insight"}})
        assert resp.status_code == 200
        assert resp.json()["error"] is None
        mock_brain.remember.assert_called_once_with("New insight")

    def test_add_empty_memory(self, client):
        resp = client.post("/call", json={"tool_name": "memory_add", "args": {"text": ""}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None


class TestMemoryWeaknesses:
    def test_list_weaknesses(self, client, mock_brain):
        resp = client.post("/call", json={"tool_name": "memory_weaknesses", "args": {}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["count"] == 1
        assert "sometimes forgets context" in result["weaknesses"]


# ---------------------------------------------------------------------------
# /call — file tools
# ---------------------------------------------------------------------------

class TestFileRead:
    def test_read_existing_file(self, client, tmp_path):
        import tools.aura_control_mcp as mod
        original_root = mod._ROOT
        mod._ROOT = tmp_path
        (tmp_path / "test.txt").write_text("hello world")
        try:
            resp = client.post("/call", json={"tool_name": "file_read", "args": {"path": "test.txt"}})
            assert resp.status_code == 200
            result = resp.json()["result"]
            assert result["content"] == "hello world"
        finally:
            mod._ROOT = original_root

    def test_read_missing_file(self, client, tmp_path):
        import tools.aura_control_mcp as mod
        original_root = mod._ROOT
        mod._ROOT = tmp_path
        try:
            resp = client.post("/call", json={"tool_name": "file_read", "args": {"path": "missing.txt"}})
            assert resp.status_code == 200
            assert resp.json()["error"] is not None
        finally:
            mod._ROOT = original_root

    def test_path_traversal_blocked(self, client):
        resp = client.post("/call", json={"tool_name": "file_read", "args": {"path": "../../etc/passwd"}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None


class TestFileList:
    def test_list_directory(self, client, tmp_path):
        import tools.aura_control_mcp as mod
        original_root = mod._ROOT
        mod._ROOT = tmp_path
        (tmp_path / "a.py").write_text("")
        (tmp_path / "subdir").mkdir()
        try:
            resp = client.post("/call", json={"tool_name": "file_list", "args": {"path": "."}})
            assert resp.status_code == 200
            result = resp.json()["result"]
            names = {e["name"] for e in result["entries"]}
            assert "a.py" in names
            assert "subdir" in names
        finally:
            mod._ROOT = original_root


# ---------------------------------------------------------------------------
# /call — project_status
# ---------------------------------------------------------------------------

class TestProjectStatus:
    def test_project_status(self, client):
        resp = client.post("/call", json={"tool_name": "project_status", "args": {}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "queue_size" in result
        assert "memory_count" in result
        assert "server_time" in result


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_tool(self, client):
        resp = client.post("/call", json={"tool_name": "does_not_exist", "args": {}})
        assert resp.status_code == 404

    def test_elapsed_ms_present(self, client):
        resp = client.post("/call", json={"tool_name": "goal_list", "args": {}})
        assert resp.status_code == 200
        assert resp.json()["elapsed_ms"] >= 0


class TestMetrics:
    def test_metrics_endpoint_exists(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_structure(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        assert "uptime_seconds" in data
        assert "total_calls" in data
        assert "total_errors" in data
        assert "error_rate" in data
        assert "queue_size" in data
        assert "memory_count" in data
        assert "tools" in data

    def test_metrics_increments_on_call(self, client):
        # Reset-ish: make a known call then check counts rose
        before = client.get("/metrics").json()["total_calls"]
        client.post("/call", json={"tool_name": "goal_list", "args": {}})
        after = client.get("/metrics").json()["total_calls"]
        assert after == before + 1

    def test_metrics_error_rate_type(self, client):
        data = client.get("/metrics").json()
        assert isinstance(data["error_rate"], float)
        assert 0.0 <= data["error_rate"] <= 1.0

    def test_metrics_per_tool_keys(self, client):
        data = client.get("/metrics").json()
        # Each tool entry has calls + errors
        for name, stats in data["tools"].items():
            assert "calls" in stats
            assert "errors" in stats
