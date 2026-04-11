"""Tests for PR Fix Agent."""

import pytest
from agents.pr_review import PRFixAgent, FixResult, FixStatus, ReviewComment, Severity


class TestPRFixAgent:
    """Test PR Fix Agent functionality."""

    @pytest.fixture
    def agent(self):
        """Create PR fix agent fixture."""
        return PRFixAgent()

    @pytest.fixture
    def sample_comment(self):
        """Create sample review comment."""
        return ReviewComment(
            path="src/test.py",
            line=2,
            message="Test issue",
            severity=Severity.WARNING,
            rule_id="PRINT_STATEMENT",
        )

    @pytest.mark.asyncio
    async def test_fix_print_statement(self, agent, sample_comment):
        """Test fixing print statements."""
        content = 'def test():\n    print("debug")\n    return 42\n'
        lines = content.split("\n")

        result = agent._fix_print_statement(sample_comment, lines)

        assert result.status == FixStatus.SUCCESS
        assert "log_json" in result.fixed_code
        assert "print" not in result.fixed_code

    @pytest.mark.asyncio
    async def test_fix_todo(self, agent):
        """Test fixing TODOs."""
        comment = ReviewComment(
            path="src/test.py",
            line=1,
            message="TODO without ticket",
            severity=Severity.WARNING,
            rule_id="TODO_WITHOUT_TICKET",
        )
        content = "# TODO fix this\ndef test():\n    pass\n"
        lines = content.split("\n")

        result = agent._fix_todo(comment, lines)

        assert result.status == FixStatus.SUCCESS
        assert "TODO(TICKET-XXX)" in result.fixed_code

    @pytest.mark.asyncio
    async def test_fix_breakpoint(self, agent):
        """Test removing breakpoints."""
        comment = ReviewComment(
            path="src/test.py",
            line=2,
            message="Debug breakpoint",
            severity=Severity.ERROR,
            rule_id="DEBUG_BREAKPOINT",
        )
        content = "def test():\n    breakpoint()\n    pass\n"
        lines = content.split("\n")

        result = agent._fix_breakpoint(comment, lines)

        assert result.status == FixStatus.SUCCESS
        assert result.fixed_code is not None
        assert "breakpoint()" not in result.fixed_code

    @pytest.mark.asyncio
    async def test_fix_bare_except(self, agent):
        """Test fixing bare except clauses."""
        comment = ReviewComment(
            path="src/test.py",
            line=3,
            message="Bare except",
            severity=Severity.ERROR,
            rule_id="BARE_EXCEPT",
        )
        content = "try:\n    do_something()\nexcept:\n    pass\n"
        lines = content.split("\n")

        result = agent._fix_bare_except(comment, lines)

        assert result.status == FixStatus.SUCCESS
        assert "except Exception:" in result.fixed_code
        assert "except:" not in result.fixed_code.replace("except Exception:", "")

    @pytest.mark.asyncio
    async def test_skip_long_line(self, agent):
        """Test that long lines are skipped."""
        comment = ReviewComment(
            path="src/test.py",
            line=1,
            message="Long line",
            severity=Severity.INFO,
            rule_id="LONG_LINE",
        )
        content = 'x = "' + "a" * 150 + '"\n'
        lines = content.split("\n")

        result = agent._fix_long_line(comment, lines)

        assert result.status == FixStatus.SKIPPED
        assert result.fixed_code is None

    @pytest.mark.asyncio
    async def test_fix_multiple_issues(self, agent):
        """Test fixing multiple issues in one file."""
        content = """def test():
    print("debug")
    # TODO fix this
    return 42
"""
        comments = [
            ReviewComment(
                path="src/test.py",
                line=2,
                message="Print statement",
                severity=Severity.WARNING,
                rule_id="PRINT_STATEMENT",
            ),
            ReviewComment(
                path="src/test.py",
                line=3,
                message="TODO without ticket",
                severity=Severity.WARNING,
                rule_id="TODO_WITHOUT_TICKET",
            ),
        ]

        results = await agent.fix_issues("src/test.py", content, comments)

        assert len(results) == 2
        # At least one fix should succeed
        assert any(r.status == FixStatus.SUCCESS for r in results)

    def test_can_fix(self, agent):
        """Test can_fix method."""
        fixable = ReviewComment(
            path="src/test.py",
            line=1,
            message="Print",
            severity=Severity.WARNING,
            rule_id="PRINT_STATEMENT",
        )
        not_fixable = ReviewComment(
            path="src/test.py",
            line=1,
            message="Long line",
            severity=Severity.INFO,
            rule_id="LONG_LINE",
        )
        unknown = ReviewComment(
            path="src/test.py",
            line=1,
            message="Unknown",
            severity=Severity.WARNING,
            rule_id="UNKNOWN_RULE",
        )

        assert agent.can_fix(fixable) is True
        assert agent.can_fix(not_fixable) is False  # Requires manual review
        assert agent.can_fix(unknown) is False

    def test_apply_fixes(self, agent):
        """Test applying fixes to content."""
        original = 'def test():\n    print("debug")\n'
        results = [
            FixResult(
                comment=ReviewComment(
                    path="src/test.py",
                    line=2,
                    message="Print",
                    severity=Severity.WARNING,
                    rule_id="PRINT_STATEMENT",
                ),
                status=FixStatus.SUCCESS,
                original_code='    print("debug")',
                fixed_code='def test():\n    log_json("DEBUG", "msg", {})\n',
                message="Fixed",
                line_number=2,
            ),
        ]

        fixed, summary = agent.apply_fixes(original, results)

        assert "log_json" in fixed
        assert "print" not in fixed
        assert summary["applied"] == 1

    @pytest.mark.asyncio
    async def test_no_fixer_available(self, agent):
        """Test handling of comments without fixers."""
        comment = ReviewComment(
            path="src/test.py",
            line=1,
            message="Unknown issue",
            severity=Severity.WARNING,
            rule_id="UNKNOWN_RULE",
        )
        content = "def test():\n    pass\n"

        results = await agent.fix_issues("src/test.py", content, [comment])

        assert len(results) == 1
        assert results[0].status == FixStatus.SKIPPED
        assert "No fixer available" in results[0].message
