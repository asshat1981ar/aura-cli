"""Tests for PR Review Agent."""

import pytest
from agents.pr_review import PRReviewAgent, ReviewComment, Severity


class TestPRReviewAgent:
    """Test PR Review Agent functionality."""

    @pytest.fixture
    def agent(self):
        """Create PR review agent fixture."""
        return PRReviewAgent()

    @pytest.fixture
    def sample_pr_data(self):
        """Sample PR data for testing."""
        return {
            "pr_number": 42,
            "pr_title": "Add new feature",
            "diff_content": "diff --git a/file.py b/file.py",
            "files": [
                {
                    "filename": "src/feature.py",
                    "content": 'def hello():\n    print("Hello")\n    return 42\n',
                    "patch": "@@ -0,0 +1,3 @@\n+def hello():\n+    print(\"Hello\")\n+    return 42",
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_review_pr_basic(self, agent, sample_pr_data):
        """Test basic PR review."""
        result = await agent.review_pr(
            pr_number=sample_pr_data["pr_number"],
            pr_title=sample_pr_data["pr_title"],
            diff_content=sample_pr_data["diff_content"],
            files=sample_pr_data["files"],
        )

        assert result.pr_number == 42
        assert result.pr_title == "Add new feature"
        assert isinstance(result.comments, list)
        assert "Summary" in result.summary

    @pytest.mark.asyncio
    async def test_detects_print_statement(self, agent):
        """Test detection of print statements."""
        files = [{
            "filename": "src/module.py",  # Not a test file
            "content": 'def test():\n    print("debug")\n',
            "patch": "@@ -0,0 +1,2 @@\n+def test():\n+    print(\"debug\")",
        }]

        result = await agent.review_pr(
            pr_number=1,
            pr_title="Test PR",
            diff_content="",
            files=files,
        )

        print_comments = [c for c in result.comments if c.rule_id == "PRINT_STATEMENT"]
        assert len(print_comments) == 1
        assert print_comments[0].severity == Severity.WARNING

    @pytest.mark.asyncio
    async def test_detects_todo_without_ticket(self, agent):
        """Test detection of TODOs without ticket references."""
        files = [{
            "filename": "src/test.py",
            "content": '# TODO fix this later\ndef test():\n    pass\n',
            "patch": "@@ -0,0 +1,3 @@\n+# TODO fix this later\n+def test():\n+    pass",
        }]

        result = await agent.review_pr(
            pr_number=1,
            pr_title="Test PR",
            diff_content="",
            files=files,
        )

        todo_comments = [c for c in result.comments if c.rule_id == "TODO_WITHOUT_TICKET"]
        assert len(todo_comments) == 1

    @pytest.mark.asyncio
    async def test_detects_debug_breakpoint(self, agent):
        """Test detection of debug breakpoints."""
        files = [{
            "filename": "src/test.py",
            "content": 'def test():\n    breakpoint()\n    pass\n',
            "patch": "@@ -0,0 +1,3 @@\n+def test():\n+    breakpoint()\n+    pass",
        }]

        result = await agent.review_pr(
            pr_number=1,
            pr_title="Test PR",
            diff_content="",
            files=files,
        )

        debug_comments = [c for c in result.comments if c.rule_id == "DEBUG_BREAKPOINT"]
        assert len(debug_comments) == 1
        assert debug_comments[0].severity == Severity.ERROR

    @pytest.mark.asyncio
    async def test_approves_clean_pr(self, agent):
        """Test that clean PRs get approved."""
        files = [{
            "filename": "src/clean.py",
            "content": 'def hello():\n    """Say hello."""\n    return "Hello"\n',
            "patch": "@@ -0,0 +1,3 @@\n+def hello():\n+    \"\"\"Say hello.\"\"\"\n+    return \"Hello\"",
        }]

        result = await agent.review_pr(
            pr_number=1,
            pr_title="Clean PR",
            diff_content="",
            files=files,
        )

        assert result.approved is True
        assert result.stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_rejects_pr_with_errors(self, agent):
        """Test that PRs with errors are not approved."""
        files = [{
            "filename": "src/bad.py",
            "content": 'def test():\n    breakpoint()\n    password = "secret123"\n',
            "patch": "@@ -0,0 +1,3 @@\n+def test():\n+    breakpoint()\n+    password = \"secret123\"",
        }]

        result = await agent.review_pr(
            pr_number=1,
            pr_title="Bad PR",
            diff_content="",
            files=files,
        )

        assert result.approved is False
        assert result.stats["errors"] > 0

    def test_format_for_github(self, agent):
        """Test GitHub API formatting."""
        from agents.pr_review import ReviewResult

        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            summary="Test summary",
            comments=[
                ReviewComment(
                    path="src/test.py",
                    line=5,
                    message="Test issue",
                    severity=Severity.WARNING,
                    suggestion="Fix it",
                )
            ],
            approved=True,
        )

        github_format = agent.format_for_github(result)

        assert "body" in github_format
        assert "event" in github_format
        assert github_format["event"] == "APPROVE"
        assert len(github_format["comments"]) == 1
        assert github_format["comments"][0]["path"] == "src/test.py"
        assert github_format["comments"][0]["line"] == 5


class TestReviewComment:
    """Test ReviewComment dataclass."""

    def test_comment_creation(self):
        """Test creating a review comment."""
        comment = ReviewComment(
            path="src/test.py",
            line=10,
            message="Test message",
            severity=Severity.ERROR,
            suggestion="Fix this",
            rule_id="TEST_RULE",
        )

        assert comment.path == "src/test.py"
        assert comment.line == 10
        assert comment.message == "Test message"
        assert comment.severity == Severity.ERROR
        assert comment.suggestion == "Fix this"
        assert comment.rule_id == "TEST_RULE"


class TestSeverity:
    """Test Severity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
