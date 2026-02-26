"""Human-in-the-loop gate that pauses autonomous operation when risk is too high.

Checks verify/skill results after each cycle and optionally prompts the user
for approval before proceeding with risky changes.

Environment variables
---------------------
AURA_AUTO_APPROVE=1   Skip interactive prompt — always approve (useful in CI
                      pipelines where the run has been explicitly sanctioned).
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

from core.logging_utils import log_json


class HumanGate:
    """Gate that pauses autonomous operation when risk is too high."""

    # Minimum drop in coverage (percentage points) that triggers a block.
    COVERAGE_DROP_THRESHOLD: float = 5.0

    def __init__(self, coverage_baseline: float | None = None):
        """Initialise the gate.

        Args:
            coverage_baseline: Known-good test-coverage percentage (0–100).
                When provided, drops beyond :attr:`COVERAGE_DROP_THRESHOLD`
                will be flagged as risky.  Pass ``None`` to skip coverage checks.
        """
        self.coverage_baseline = coverage_baseline

    # ── Public API ───────────────────────────────────────────────────────────

    def should_block(
        self, verify_result: dict, skill_results: dict
    ) -> Tuple[bool, str]:
        """Decide whether the current change-set is too risky to auto-apply.

        Args:
            verify_result: Output dict from the ``"verify"`` phase
                (``{"status": ..., "failures": [...], ...}``).
            skill_results: Aggregated skill-dispatch output, potentially
                containing ``"security_scanner"`` and
                ``"test_coverage_analyzer"`` sub-dicts.

        Returns:
            A 2-tuple ``(blocked: bool, reason: str)``.  When *blocked* is
            ``False`` the reason string is empty.
        """
        # ── Security: block on any critical findings ─────────────────────────
        sec = skill_results.get("security_scanner", {})
        critical_count = sec.get("critical_count", 0)
        if isinstance(critical_count, int) and critical_count > 0:
            reason = (
                f"security_scanner reported {critical_count} critical finding(s)"
            )
            log_json("WARN", "human_gate_blocked", details={"reason": reason})
            return True, reason

        # ── Coverage: block on significant drop from baseline ────────────────
        if self.coverage_baseline is not None:
            cov = skill_results.get("test_coverage_analyzer", {})
            current_pct = cov.get("coverage_pct")
            if isinstance(current_pct, (int, float)):
                drop = self.coverage_baseline - float(current_pct)
                if drop > self.COVERAGE_DROP_THRESHOLD:
                    reason = (
                        f"test coverage dropped {drop:.1f}pp "
                        f"(baseline={self.coverage_baseline:.1f}%, "
                        f"current={current_pct:.1f}%)"
                    )
                    log_json("WARN", "human_gate_blocked", details={"reason": reason})
                    return True, reason

        return False, ""

    def request_approval(self, reason: str, change_summary: dict) -> bool:
        """Ask the operator whether to proceed with a risky change.

        Behaviour depends on the runtime context:

        * If ``AURA_AUTO_APPROVE=1`` — automatically approves and returns
          ``True`` (useful for explicitly-sanctioned CI pipelines).
        * If stdin is not a TTY (non-interactive / CI) — automatically denies
          and returns ``False``.
        * Otherwise — prompts the user with ``y/n``.

        Args:
            reason: Human-readable explanation of why the gate triggered.
            change_summary: Dict summary of the proposed changes shown to the
                user for context (e.g. ``{"files_changed": [...], ...}``).

        Returns:
            ``True`` if the operator approves proceeding, ``False`` to deny.
        """
        if os.environ.get("AURA_AUTO_APPROVE", "").strip() == "1":
            log_json("INFO", "human_gate_auto_approved", details={"reason": reason})
            return True

        if not sys.stdin.isatty():
            log_json("WARN", "human_gate_denied_non_interactive",
                     details={"reason": reason})
            return False

        # Interactive prompt
        print(f"\n[AURA HumanGate] Blocked: {reason}")
        print(f"Change summary: {change_summary}")
        try:
            answer = input("Approve and proceed? [y/N] ").strip().lower()
        except EOFError:
            answer = ""

        approved = answer in ("y", "yes")
        log_json(
            "INFO" if approved else "WARN",
            "human_gate_user_decision",
            details={"approved": approved, "reason": reason},
        )
        return approved
