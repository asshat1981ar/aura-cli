"""PR Fix Agent for automatic code fixing.

Automatically fixes issues detected by PRReviewAgent.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.logging_utils import log_json
from .review_agent import ReviewComment


class FixStatus(Enum):
    """Status of a fix attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class FixResult:
    """Result of a fix operation."""

    comment: ReviewComment
    status: FixStatus
    original_code: str
    fixed_code: Optional[str]
    message: str
    line_number: int


class PRFixAgent:
    """Agent for automatically fixing code issues."""

    def __init__(self):
        self._fixers = {
            "PRINT_STATEMENT": self._fix_print_statement,
            "TODO_WITHOUT_TICKET": self._fix_todo,
            "DEBUG_BREAKPOINT": self._fix_breakpoint,
            "BARE_EXCEPT": self._fix_bare_except,
            "LONG_LINE": self._fix_long_line,
        }

    async def fix_issues(
        self,
        file_path: str,
        content: str,
        comments: List[ReviewComment],
    ) -> List[FixResult]:
        """Fix issues in a file.

        Args:
            file_path: Path to the file
            content: File content
            comments: Review comments to fix

        Returns:
            List of fix results
        """
        results = []
        lines = content.split("\n")

        for comment in comments:
            fixer = self._fixers.get(comment.rule_id)
            if fixer:
                try:
                    result = fixer(comment, lines)
                    results.append(result)

                    if result.status == FixStatus.SUCCESS and result.fixed_code:
                        # Update lines for subsequent fixes
                        lines = result.fixed_code.split("\n")

                except Exception as e:
                    log_json(
                        "ERROR",
                        "fix_failed",
                        {
                            "rule": comment.rule_id,
                            "error": str(e),
                        },
                    )
                    results.append(
                        FixResult(
                            comment=comment,
                            status=FixStatus.FAILED,
                            original_code="",
                            fixed_code=None,
                            message=f"Fix failed: {e}",
                            line_number=comment.line,
                        )
                    )
            else:
                results.append(
                    FixResult(
                        comment=comment,
                        status=FixStatus.SKIPPED,
                        original_code="",
                        fixed_code=None,
                        message=f"No fixer available for {comment.rule_id}",
                        line_number=comment.line,
                    )
                )

        return results

    def _fix_print_statement(self, comment: ReviewComment, lines: List[str]) -> FixResult:
        """Fix print statements by replacing with logging.

        Args:
            comment: The review comment
            lines: File lines

        Returns:
            Fix result
        """
        line_idx = comment.line - 1
        if line_idx >= len(lines):
            return FixResult(
                comment=comment,
                status=FixStatus.FAILED,
                original_code="",
                fixed_code=None,
                message="Line number out of range",
                line_number=comment.line,
            )

        original_line = lines[line_idx]

        # Pattern: print(...) -> log_json("DEBUG", "message", {...})
        # This is a simple transformation - real implementation would be more sophisticated
        match = re.match(r"^(\s*)print\s*\((.*)\)\s*$", original_line)
        if match:
            indent = match.group(1)
            args = match.group(2)

            # Try to extract the message
            if '"' in args or "'" in args:
                # String argument - extract it
                fixed_line = f'{indent}from core.logging_utils import log_json\n{indent}log_json("DEBUG", "debug_output", {{"value": {args}}})'
            else:
                # Variable or expression
                fixed_line = f'{indent}from core.logging_utils import log_json\n{indent}log_json("DEBUG", "debug_output", {{"value": str({args})}})'

            lines[line_idx] = fixed_line

            return FixResult(
                comment=comment,
                status=FixStatus.SUCCESS,
                original_code=original_line,
                fixed_code="\n".join(lines),
                message="Replaced print with log_json",
                line_number=comment.line,
            )

        return FixResult(
            comment=comment,
            status=FixStatus.FAILED,
            original_code=original_line,
            fixed_code=None,
            message="Could not parse print statement",
            line_number=comment.line,
        )

    def _fix_todo(self, comment: ReviewComment, lines: List[str]) -> FixResult:
        """Fix TODOs by adding a placeholder ticket reference.

        Args:
            comment: The review comment
            lines: File lines

        Returns:
            Fix result
        """
        line_idx = comment.line - 1
        original_line = lines[line_idx]

        # Replace TODO with TODO(TICKET-XXX)
        fixed_line = re.sub(
            r"#\s*TODO\b",
            "# TODO(TICKET-XXX)",
            original_line,
            flags=re.IGNORECASE,
        )

        if fixed_line != original_line:
            lines[line_idx] = fixed_line
            return FixResult(
                comment=comment,
                status=FixStatus.SUCCESS,
                original_code=original_line,
                fixed_code="\n".join(lines),
                message="Added placeholder ticket reference to TODO",
                line_number=comment.line,
            )

        return FixResult(
            comment=comment,
            status=FixStatus.FAILED,
            original_code=original_line,
            fixed_code=None,
            message="Could not modify TODO",
            line_number=comment.line,
        )

    def _fix_breakpoint(self, comment: ReviewComment, lines: List[str]) -> FixResult:
        """Remove debug breakpoints.

        Args:
            comment: The review comment
            lines: File lines

        Returns:
            Fix result
        """
        line_idx = comment.line - 1
        original_line = lines[line_idx]

        # Remove breakpoint lines
        if re.match(r"^\s*(breakpoint\s*\(\s*\)|import\s+pdb|pdb\.set_trace)", original_line):
            lines.pop(line_idx)
            return FixResult(
                comment=comment,
                status=FixStatus.SUCCESS,
                original_code=original_line,
                fixed_code="\n".join(lines),
                message="Removed debug breakpoint",
                line_number=comment.line,
            )

        return FixResult(
            comment=comment,
            status=FixStatus.FAILED,
            original_code=original_line,
            fixed_code=None,
            message="Could not remove breakpoint",
            line_number=comment.line,
        )

    def _fix_bare_except(self, comment: ReviewComment, lines: List[str]) -> FixResult:
        """Fix bare except clauses.

        Args:
            comment: The review comment
            lines: File lines

        Returns:
            Fix result
        """
        line_idx = comment.line - 1
        original_line = lines[line_idx]

        # Replace bare except with except Exception
        fixed_line = re.sub(
            r"except\s*:",
            "except Exception:",
            original_line,
        )

        if fixed_line != original_line:
            lines[line_idx] = fixed_line
            return FixResult(
                comment=comment,
                status=FixStatus.SUCCESS,
                original_code=original_line,
                fixed_code="\n".join(lines),
                message="Replaced bare except with except Exception",
                line_number=comment.line,
            )

        return FixResult(
            comment=comment,
            status=FixStatus.FAILED,
            original_code=original_line,
            fixed_code=None,
            message="Could not fix bare except",
            line_number=comment.line,
        )

    def _fix_long_line(self, comment: ReviewComment, lines: List[str]) -> FixResult:
        """Fix long lines by suggesting breaks.

        Args:
            comment: The review comment
            lines: File lines

        Returns:
            Fix result
        """
        line_idx = comment.line - 1
        original_line = lines[line_idx]

        # This is tricky to auto-fix - mark as manual
        return FixResult(
            comment=comment,
            status=FixStatus.SKIPPED,
            original_code=original_line,
            fixed_code=None,
            message="Long lines require manual review - consider breaking the line",
            line_number=comment.line,
        )

    def apply_fixes(self, original_content: str, results: List[FixResult]) -> Tuple[str, Dict[str, Any]]:
        """Apply successful fixes to content.

        Args:
            original_content: Original file content
            results: Fix results

        Returns:
            Tuple of (fixed content, summary)
        """
        successful_fixes = [r for r in results if r.status == FixStatus.SUCCESS]

        if not successful_fixes:
            return original_content, {
                "applied": 0,
                "failed": len(results),
                "skipped": 0,
            }

        # Apply fixes in reverse order (line numbers change)
        lines = original_content.split("\n")
        applied = 0

        for result in sorted(successful_fixes, key=lambda r: r.line_number, reverse=True):
            if result.fixed_code:
                lines = result.fixed_code.split("\n")
                applied += 1

        summary = {
            "applied": applied,
            "failed": len([r for r in results if r.status == FixStatus.FAILED]),
            "skipped": len([r for r in results if r.status == FixStatus.SKIPPED]),
            "total": len(results),
        }

        return "\n".join(lines), summary

    def can_fix(self, comment: ReviewComment) -> bool:
        """Check if a comment can be automatically fixed.

        Args:
            comment: Review comment

        Returns:
            True if fixable
        """
        return comment.rule_id in self._fixers and comment.rule_id != "LONG_LINE"
