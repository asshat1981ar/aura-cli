"""Verification, failure routing, and error analysis methods for the orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json
from core.schema import RoutingDecision


class VerifyMixin:
    """Mixin providing verification, failure routing, and error analysis methods."""

    def _analyze_error(self, error: str, context: Optional[dict] = None) -> Optional[str]:
        """Optionally use SelfCorrectionAgent to analyze and suggest fixes."""
        if not self.self_correction_agent:
            return None
        try:
            return self.self_correction_agent.analyze_error(error, context or {})
        except Exception as exc:
            log_json("WARN", "self_correction_analysis_failed", details={"error": str(exc)})
            return None

    def _run_root_cause_analysis(
        self,
        failures: List[str],
        logs: str,
        context: Optional[dict] = None,
        *,
        history: Optional[List[dict]] = None,
    ) -> Optional[dict]:
        """Optionally produce a structured RCA report for a failed phase."""
        if not self.root_cause_analysis_agent:
            return None
        try:
            return self.root_cause_analysis_agent.run(
                {
                    "failures": failures,
                    "logs": logs,
                    "context": context or {},
                    "history": history or [],
                }
            )
        except Exception as exc:
            log_json("WARN", "root_cause_analysis_failed", details={"error": str(exc)})
            return None

    def _run_investigation(
        self,
        *,
        goal: str,
        verification: Dict[str, Any],
        context: Optional[dict] = None,
        route: str = "act",
        analysis_suggestion: str | None = None,
        root_cause_analysis: Optional[dict] = None,
        previous_test_count: int | None = None,
        current_test_count: int | None = None,
    ) -> Optional[dict]:
        """Optionally produce a structured investigation report for a failed phase."""
        if not self.investigation_agent:
            return None
        try:
            return self.investigation_agent.run(
                {
                    "goal": goal,
                    "verification": verification,
                    "context": context or {},
                    "route": route,
                    "analysis_suggestion": analysis_suggestion,
                    "root_cause_analysis": root_cause_analysis or {},
                    "history": self._failure_history(),
                    "previous_test_count": previous_test_count,
                    "current_test_count": current_test_count,
                }
            )
        except Exception as exc:
            log_json("WARN", "investigation_failed", details={"error": str(exc)})
            return None

    def _failure_history(self, limit: int = 5) -> List[dict]:
        """Return recent cycle summaries to help classify repeated failures."""
        if not self.memory_controller or not self.memory_controller.persistent_store:
            return []
        try:
            return list(self.memory_controller.read_log()[-limit:])
        except (OSError, AttributeError):
            return []

    def _restore_applied_changes(self, snapshots: List[Dict]) -> None:
        """Restore only the files mutated by the current loop attempt.

        This avoids touching unrelated user changes elsewhere in the repo.
        """
        restored: list[str] = []
        failed: list[Dict[str, str]] = []

        for snapshot in reversed(snapshots):
            target = Path(snapshot["target"])
            try:
                if snapshot.get("existed"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(snapshot.get("content") or "", encoding="utf-8")
                    if snapshot.get("mode") is not None:
                        os.chmod(target, int(snapshot["mode"]))
                else:
                    if target.exists():
                        target.unlink()
                restored.append(snapshot["file"])
            except Exception as exc:
                failed.append({"file": snapshot["file"], "error": str(exc)})

        if restored:
            log_json("INFO", "verify_fail_restore_ok", details={"files": restored})
        if failed:
            log_json("WARN", "verify_fail_restore_failed", details={"failures": failed})

    def _route_failure(self, verification: Dict) -> str:
        """Classify a verification failure and return the recommended re-entry point.

        Inspects the ``"failures"`` list and ``"logs"`` string in *verification*
        for well-known signal words to determine how the orchestrator should
        respond.

        Args:
            verification: The dict returned by the ``"verify"`` phase.  The
                keys ``"failures"`` (list) and ``"logs"`` (str) are inspected;
                all other keys are ignored.

        Returns:
            A :class:`~core.schema.RoutingDecision` value:

            * :attr:`~core.schema.RoutingDecision.ACT`  — recoverable code-level error; retry the act phase.
            * :attr:`~core.schema.RoutingDecision.PLAN` — structural or design error; re-plan from scratch.
            * :attr:`~core.schema.RoutingDecision.SKIP` — external / environment issue that cannot be
              self-fixed (e.g. missing dependency, network error).
        """
        failures = " ".join(str(f) for f in verification.get("failures", []))
        logs = str(verification.get("logs", ""))
        combined = (failures + " " + logs).lower()

        structural_signals = [
            "architecture",
            "circular",
            "api_breaking",
            "breaking_change",
            "design",
            "interface",
            "contract",
        ]
        external_signals = [
            "dependency",
            "network",
            "env",
            "environment",
            "permission",
            "not found",
            "no module",
            "import error",
        ]

        if any(s in combined for s in structural_signals):
            return RoutingDecision.PLAN
        if any(s in combined for s in external_signals):
            return RoutingDecision.SKIP
        return RoutingDecision.ACT  # default: code-level fix is worth retrying

    def _normalize_verification_result(self, verification: Dict) -> Dict:
        """Accept both legacy ``passed`` and canonical ``status`` verification payloads."""
        if not isinstance(verification, dict):
            return {"status": "fail", "failures": ["invalid verification payload"], "logs": str(verification)}
        if verification.get("status") in ("pass", "fail", "skip"):
            return verification
        if "passed" in verification:
            normalized = dict(verification)
            normalized["status"] = "pass" if bool(verification.get("passed")) else "fail"
            normalized.setdefault("failures", [])
            normalized.setdefault("logs", "")
            return normalized
        return verification
