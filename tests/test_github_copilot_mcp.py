"""Tests for tools/github_copilot_mcp.py — mocks GitHub + Model so no credentials needed."""
from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    import tools.github_copilot_mcp as mod
    mod._github = None
    mod._model = None
    yield
    mod._github = None
    mod._model = None


def _make_github_mock():
    gh = MagicMock()
    gh.BASE_URL = "https://api.github.com"
    gh.headers = {"Authorization": "token test"}
    gh.get_repo.return_value = {
        "stargazers_count": 42, "forks_count": 5, "open_issues_count": 3,
        "language": "Python", "license": {"spdx_id": "MIT"},
        "has_wiki": True, "archived": False,
    }
    gh.get_issue_details.return_value = {
        "title": "Fix null pointer in auth module",
        "body": "When user is None the login() function crashes with AttributeError.",
        "labels": [{"name": "bug"}, {"name": "priority:high"}],
    }
    gh.get_pull_request_details.return_value = {
        "title": "Add null check in login()",
        "body": "Fixes #42",
        "head": {"ref": "fix/null-login"},
        "base": {"ref": "main"},
    }
    gh._make_request.return_value = []
    gh.search_code.return_value = {
        "results": [
            {"path": "src/auth.py", "name": "auth.py", "url": "https://github.com/x/y/blob/main/src/auth.py"},
            {"path": "tests/test_auth.py", "name": "test_auth.py", "url": "https://github.com/x/y/blob/main/tests/test_auth.py"},
        ],
        "total_count": 2,
    }
    return gh


def _make_model_mock():
    m = MagicMock()
    m.respond.return_value = "AI response: here is the analysis."
    return m


@pytest.fixture()
def client():
    import tools.github_copilot_mcp as mod
    mod._github = _make_github_mock()
    mod._model = _make_model_mock()
    from tools.github_copilot_mcp import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["tool_count"] == 10
        assert data["server"] == "github_copilot_mcp"


# ---------------------------------------------------------------------------
# /tools  /tool/{name}
# ---------------------------------------------------------------------------

class TestToolList:
    def test_lists_all_10_tools(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()}
        assert names == {
            "issue_analyze", "pr_review", "pr_describe", "code_explain",
            "code_fix", "test_generate", "commit_message", "repo_health",
            "issue_to_plan", "find_related_code",
        }

    def test_each_tool_has_input_schema(self, client):
        for tool in client.get("/tools").json():
            assert "inputSchema" in tool
            assert "description" in tool

    def test_get_single_tool(self, client):
        resp = client.get("/tool/pr_review")
        assert resp.status_code == 200
        assert resp.json()["name"] == "pr_review"

    def test_get_missing_tool_404(self, client):
        resp = client.get("/tool/does_not_exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /call — each tool
# ---------------------------------------------------------------------------

class TestIssueAnalyze:
    def test_returns_analysis(self, client):
        resp = client.post("/call", json={
            "tool_name": "issue_analyze",
            "args": {"repo": "owner/repo", "issue_number": 42},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert "analysis" in data["result"]
        assert data["result"]["issue_number"] == 42

    def test_missing_required_arg(self, client):
        resp = client.post("/call", json={"tool_name": "issue_analyze", "args": {"repo": "owner/repo"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is not None


class TestPRReview:
    def test_returns_review_and_verdict(self, client):
        import tools.github_copilot_mcp as mod
        # patch requests.get for diff fetch
        with patch("tools.github_copilot_mcp._req") as mock_req:
            mock_req_inst = MagicMock()
            mock_req_inst.ok = True
            mock_req_inst.text = "diff --git a/auth.py b/auth.py\n+if user is None: return\n"
            mock_req.get.return_value = mock_req_inst
            resp = client.post("/call", json={
                "tool_name": "pr_review",
                "args": {"repo": "owner/repo", "pr_number": 1},
            })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "review" in result
        assert "verdict" in result
        assert result["verdict"] in ("APPROVE", "REQUEST_CHANGES", "COMMENT")


class TestPRDescribe:
    def test_returns_generated_description(self, client):
        with patch("tools.github_copilot_mcp._req"):
            resp = client.post("/call", json={
                "tool_name": "pr_describe",
                "args": {"repo": "owner/repo", "pr_number": 1},
            })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "generated_description" in result


class TestCodeExplain:
    def test_explains_code(self, client):
        resp = client.post("/call", json={
            "tool_name": "code_explain",
            "args": {"code": "def add(a, b): return a + b", "language": "python"},
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "explanation" in result
        assert result["language"] == "python"

    def test_missing_code_returns_error(self, client):
        resp = client.post("/call", json={"tool_name": "code_explain", "args": {}})
        assert resp.status_code == 200
        assert resp.json()["error"] is not None


class TestCodeFix:
    def test_returns_fix(self, client):
        resp = client.post("/call", json={
            "tool_name": "code_fix",
            "args": {
                "code": "print(user.name)",
                "error": "AttributeError: 'NoneType' object has no attribute 'name'",
            },
        })
        assert resp.status_code == 200
        assert resp.json()["result"]["fix"] is not None


class TestTestGenerate:
    def test_generates_tests(self, client):
        resp = client.post("/call", json={
            "tool_name": "test_generate",
            "args": {"code": "def multiply(a, b): return a * b", "framework": "pytest"},
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["framework"] == "pytest"
        assert "tests" in result


class TestCommitMessage:
    def test_generates_commit_msg(self, client):
        import tools.github_copilot_mcp as mod
        mod._model.respond.return_value = "## Commit Message\nfeat(auth): add null check for user login\n## Body\n- Prevents AttributeError when user is None"
        resp = client.post("/call", json={
            "tool_name": "commit_message",
            "args": {"diff": "+if user is None:\n+    return None"},
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "commit_message" in result
        assert len(result["commit_message"]) > 0


class TestRepoHealth:
    def test_returns_health_score(self, client):
        import tools.github_copilot_mcp as mod
        mod._github._make_request.return_value = []
        resp = client.post("/call", json={
            "tool_name": "repo_health",
            "args": {"repo": "owner/repo"},
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "health_score" in result
        assert 0 <= result["health_score"] <= 100
        assert "ai_summary" in result


class TestIssueToplan:
    def test_returns_plan(self, client):
        resp = client.post("/call", json={
            "tool_name": "issue_to_plan",
            "args": {"repo": "owner/repo", "issue_number": 42},
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "plan" in result
        assert result["issue_number"] == 42


class TestFindRelatedCode:
    def test_returns_ranked_results(self, client):
        resp = client.post("/call", json={
            "tool_name": "find_related_code",
            "args": {"repo": "owner/repo", "query": "authentication login"},
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert "results" in result
        assert "ai_ranking" in result
        assert result["total_found"] == 2

    def test_no_results_returns_gracefully(self, client):
        import tools.github_copilot_mcp as mod
        mod._github.search_code.return_value = {"results": [], "total_count": 0}
        resp = client.post("/call", json={
            "tool_name": "find_related_code",
            "args": {"repo": "owner/repo", "query": "zzz_nonexistent"},
        })
        assert resp.status_code == 200
        assert resp.json()["result"]["results"] == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_tool_404(self, client):
        resp = client.post("/call", json={"tool_name": "fake_tool", "args": {}})
        assert resp.status_code == 404

    def test_elapsed_ms_in_response(self, client):
        resp = client.post("/call", json={
            "tool_name": "code_explain",
            "args": {"code": "x = 1"},
        })
        assert resp.json()["elapsed_ms"] >= 0

    def test_model_error_surfaced_gracefully(self, client):
        import tools.github_copilot_mcp as mod
        mod._model.respond.side_effect = RuntimeError("Model unavailable")
        resp = client.post("/call", json={
            "tool_name": "code_explain",
            "args": {"code": "x = 1"},
        })
        assert resp.status_code == 200
        # Should degrade gracefully — either error field or model error in result
        data = resp.json()
        assert data["result"] is not None or data["error"] is not None
