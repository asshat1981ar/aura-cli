"""PR Review Agent implementation.

Provides automated code review for GitHub pull requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

from core.logging_utils import log_json


class Severity(Enum):
    """Review comment severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ReviewComment:
    """A single review comment."""
    path: str
    line: int
    message: str
    severity: Severity = Severity.INFO
    suggestion: Optional[str] = None
    rule_id: Optional[str] = None


@dataclass
class ReviewResult:
    """Complete PR review result."""
    pr_number: int
    pr_title: str
    summary: str
    comments: List[ReviewComment] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False


class PRReviewAgent:
    """Agent for reviewing pull requests."""

    # Patterns to check
    PATTERNS = {
        "TODO_WITHOUT_TICKET": {
            "pattern": r"#\s*TODO[^:]*$",
            "message": "TODO without ticket reference",
            "severity": Severity.WARNING,
            "suggestion": "Add ticket reference: TODO(PROJ-123): description",
        },
        "PRINT_STATEMENT": {
            "pattern": r"^\s*print\s*\(",
            "message": "Use logging instead of print()",
            "severity": Severity.WARNING,
            "suggestion": "Replace with log_json() or logger.debug()",
        },
        "BARE_EXCEPT": {
            "pattern": r"except\s*:",
            "message": "Bare except clause",
            "severity": Severity.ERROR,
            "suggestion": "Use 'except Exception:' or specific exception type",
        },
        "DEBUG_BREAKPOINT": {
            "pattern": r"(import\s+pdb|pdb\.set_trace\(\)|breakpoint\s*\(?)",
            "message": "Debug breakpoint left in code",
            "severity": Severity.ERROR,
            "suggestion": "Remove breakpoint before committing",
        },
        "HARDcoded_SECRET": {
            "pattern": r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']',
            "message": "Possible hardcoded secret",
            "severity": Severity.ERROR,
            "suggestion": "Use environment variables or secret manager",
        },
        "LONG_LINE": {
            "pattern": r".{101,}",
            "message": "Line exceeds 100 characters",
            "severity": Severity.INFO,
            "suggestion": "Break into multiple lines",
        },
    }

    def __init__(self):
        self.stats = {
            "files_changed": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "issues_found": 0,
            "warnings": 0,
            "errors": 0,
        }

    async def review_pr(
        self,
        pr_number: int,
        pr_title: str,
        diff_content: str,
        files: List[Dict[str, Any]],
    ) -> ReviewResult:
        """Review a pull request.

        Args:
            pr_number: PR number
            pr_title: PR title
            diff_content: The diff content
            files: List of changed files with content

        Returns:
            ReviewResult with comments and summary
        """
        log_json("INFO", "pr_review_started", {
            "pr_number": pr_number,
            "title": pr_title,
            "files_count": len(files),
        })

        comments: List[ReviewComment] = []
        self.stats["files_changed"] = len(files)

        # Analyze each file
        for file_info in files:
            file_comments = self._analyze_file(file_info)
            comments.extend(file_comments)

        # Calculate stats
        self.stats["issues_found"] = len(comments)
        self.stats["warnings"] = sum(1 for c in comments if c.severity == Severity.WARNING)
        self.stats["errors"] = sum(1 for c in comments if c.severity == Severity.ERROR)

        # Generate summary
        summary = self._generate_summary(pr_title, comments)

        # Determine approval
        approved = self.stats["errors"] == 0

        result = ReviewResult(
            pr_number=pr_number,
            pr_title=pr_title,
            summary=summary,
            comments=comments,
            stats=self.stats.copy(),
            approved=approved,
        )

        log_json("INFO", "pr_review_completed", {
            "pr_number": pr_number,
            "comments_count": len(comments),
            "approved": approved,
        })

        return result

    def _analyze_file(self, file_info: Dict[str, Any]) -> List[ReviewComment]:
        """Analyze a single file for issues."""
        comments = []
        path = file_info.get("filename", "")
        content = file_info.get("content", "")
        patch = file_info.get("patch", "")

        # Skip non-Python files for pattern checks
        if not path.endswith(".py"):
            return comments

        # Skip test files for certain checks
        is_test = "test" in path.lower()

        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Check each pattern
            for rule_id, rule in self.PATTERNS.items():
                # Skip certain checks for test files
                if is_test and rule_id in ["PRINT_STATEMENT"]:
                    continue

                if re.search(rule["pattern"], line):
                    # Try to map to changed lines from patch
                    if self._is_line_in_patch(line_num, patch):
                        comment = ReviewComment(
                            path=path,
                            line=line_num,
                            message=rule["message"],
                            severity=rule["severity"],
                            suggestion=rule.get("suggestion"),
                            rule_id=rule_id,
                        )
                        comments.append(comment)

        return comments

    def _is_line_in_patch(self, line_num: int, patch: str) -> bool:
        """Check if a line number is within the changed lines of a patch."""
        if not patch:
            return True  # Assume included if no patch info

        # Simple parser for unified diff format
        # @@ -start,count +start,count @@
        for line in patch.split("\n"):
            if line.startswith("@@"):
                # Extract new file line numbers
                match = re.search(r"\+(\d+),?(\d*)", line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    if start <= line_num < start + count:
                        return True
        return False

    def _generate_summary(self, pr_title: str, comments: List[ReviewComment]) -> str:
        """Generate a review summary."""
        lines = [
            f"## AURA Code Review",
            f"",
            f"**PR:** {pr_title}",
            f"",
            f"### Summary",
            f"- Files changed: {self.stats['files_changed']}",
            f"- Issues found: {self.stats['issues_found']}",
            f"  - Errors: {self.stats['errors']}",
            f"  - Warnings: {self.stats['warnings']}",
            f"",
        ]

        if comments:
            lines.extend([
                "### Issues by Category",
                "",
            ])

            # Group by rule
            by_rule: Dict[str, List[ReviewComment]] = {}
            for c in comments:
                rule = c.rule_id or "OTHER"
                by_rule.setdefault(rule, []).append(c)

            for rule_id, rule_comments in sorted(by_rule.items()):
                severity = rule_comments[0].severity.value.upper()
                lines.append(f"- **{rule_id}** ({severity}): {len(rule_comments)} occurrence(s)")

            lines.append("")

            if self.stats["errors"] > 0:
                lines.append("⚠️ **Please address the errors before merging.**")
            else:
                lines.append("✅ **No critical issues found.**")
        else:
            lines.append("✅ **No issues found. Great job!**")

        lines.append("")
        lines.append("---")
        lines.append("*This review was generated automatically by AURA.*")

        return "\n".join(lines)

    def format_for_github(self, result: ReviewResult) -> Dict[str, Any]:
        """Format review result for GitHub API."""
        body = result.summary

        # Format comments
        comments = []
        for c in result.comments:
            comment = {
                "path": c.path,
                "line": c.line,
                "body": f"**{c.severity.value.upper()}:** {c.message}",
            }
            if c.suggestion:
                comment["body"] += f"\n\n💡 **Suggestion:** {c.suggestion}"
            comments.append(comment)

        return {
            "body": body,
            "event": "APPROVE" if result.approved else "COMMENT",
            "comments": comments,
        }
