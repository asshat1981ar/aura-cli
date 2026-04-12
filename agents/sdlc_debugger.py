"""Debugger agent that performs SDLC-wide failure analysis."""

from __future__ import annotations

from typing import List

from core.swarm_models import DebugReport, SDLCFinding, SDLCLens, SwarmTask, TaskResult
from core.logging_utils import log_json


class SDLCDebuggerAgent:
    """Analyzes failed tasks across requirements, design, code, test, and ops lenses."""

    name = "sdlc_debugger"

    async def plan(self, task: SwarmTask, result: TaskResult) -> List[str]:
        """Return a high-level analysis plan for the failed task."""
        return [
            "Classify the failure signal.",
            "Map the failure to SDLC lenses.",
            "Build a prioritized recovery plan.",
            "Emit a debug task with concrete acceptance criteria.",
        ]

    async def execute(self, task: SwarmTask, result: TaskResult) -> DebugReport:
        """Generate a structured root-cause report."""
        message = " ".join(part for part in [result.summary, result.error_message or "", str(result.output)] if part).lower()

        findings: List[SDLCFinding] = []

        if any(token in message for token in ("requirement", "story", "acceptance criteria", "unclear")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.REQUIREMENTS,
                    severity="major",
                    observation="The failure suggests ambiguity between requested behavior and implemented behavior.",
                    probable_cause="Forge story or acceptance criteria were underspecified.",
                    recommended_action="Refine the story into verifiable acceptance criteria before re-coding.",
                )
            )

        if any(token in message for token in ("design", "architecture", "dependency", "coupling", "layer")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.DESIGN,
                    severity="major",
                    observation="The failure points to a design or layering mismatch.",
                    probable_cause="Responsibilities were assigned without an explicit execution contract.",
                    recommended_action="Introduce a plan artifact and explicit task boundaries between agents.",
                )
            )

        if any(token in message for token in ("exception", "traceback", "typeerror", "attributeerror", "syntax")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.IMPLEMENTATION,
                    severity="major",
                    observation="The failure has a direct implementation signal.",
                    probable_cause="Code was generated without satisfying runtime or type constraints.",
                    recommended_action="Patch the implementation and add targeted regression tests.",
                )
            )

        if any(token in message for token in ("timeout", "connection", "mcp", "port", "network", "integration")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.INTEGRATION,
                    severity="major",
                    observation="The failure implicates orchestration or external tool integration.",
                    probable_cause="MCP routing, port mapping, or tool contract alignment is incomplete.",
                    recommended_action="Verify MCP server availability on ports 8001-8007 and validate tool schemas.",
                )
            )

        if any(token in message for token in ("assert", "pytest", "test", "coverage", "fixture")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.TESTING,
                    severity="major",
                    observation="The failure was surfaced by the validation layer.",
                    probable_cause="Tests were absent, stale, or not aligned to acceptance criteria.",
                    recommended_action="Create a regression test first, then apply the fix until green.",
                )
            )

        if any(token in message for token in ("secret", "token", "injection", "auth", "permission")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.SECURITY,
                    severity="major",
                    observation="The failure intersects a security-sensitive path.",
                    probable_cause="Unsafe prompt handling, auth gaps, or credential misuse.",
                    recommended_action="Add explicit secret redaction and narrow tool permissions before retrying.",
                )
            )

        if any(token in message for token in ("slow", "latency", "memory", "cpu", "performance")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.PERFORMANCE,
                    severity="minor",
                    observation="The task degraded due to runtime performance pressure.",
                    probable_cause="Work was too coarse-grained or repeated unnecessary computation.",
                    recommended_action="Split the task further and cache expensive intermediate artifacts.",
                )
            )

        if any(token in message for token in ("deploy", "runtime", "env", "configuration", "container")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.OPERATIONS,
                    severity="major",
                    observation="The failure depends on environment or operational configuration.",
                    probable_cause="Runtime assumptions differ from the active environment.",
                    recommended_action="Codify runtime prerequisites and validate them before task execution.",
                )
            )

        if any(token in message for token in ("manual", "confusing", "discoverability", "developer")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.DX,
                    severity="minor",
                    observation="The task friction impacts developer experience and recovery speed.",
                    probable_cause="Insufficient documentation or poor task context transfer.",
                    recommended_action="Emit richer context in handoffs and update operator-facing docs.",
                )
            )

        if any(token in message for token in ("pr", "merge", "release", "branch", "handoff")):
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.DELIVERY,
                    severity="minor",
                    observation="The failure blocks delivery workflow progression.",
                    probable_cause="Quality gates and release triggers are not synchronized.",
                    recommended_action="Keep PR creation gated on tester success and explicit cycle completion.",
                )
            )

        if not findings:
            findings.append(
                SDLCFinding(
                    lens=SDLCLens.IMPLEMENTATION,
                    severity="major",
                    observation="A failure occurred without a specialized classifier match.",
                    probable_cause="The worker produced a generic failure without enough telemetry.",
                    recommended_action="Capture richer stderr/stdout and rerun with debug tracing enabled.",
                )
            )

        recovery_plan = self._build_recovery_plan(findings, task)
        debug_report = DebugReport(
            task_id=task.task_id,
            failure_summary=result.summary,
            findings=findings,
            recovery_plan=recovery_plan,
            should_retry=True,
        )

        # Superpowers: SDLC-wide failure classification + GitHub Issue emission
        await self._emit_github_issue(task, debug_report)

        return debug_report

    async def _emit_github_issue(self, task: SwarmTask, report: DebugReport) -> None:
        """Emit a GitHub issue for the SDLC-wide failure."""
        # This assumes a GitHub MCP is reachable on port 8007
        # In a real run, this would call 'create_issue'
        issue_body = f"SDLC Debugger RCA for {task.task_id}\n\n"
        issue_body += f"Summary: {report.failure_summary}\n\n"
        issue_body += "Findings:\n"
        for f in report.findings:
            issue_body += f"- [{f.lens}] {f.observation} (Cause: {f.probable_cause})\n"

        issue_body += "\nRecovery Plan:\n"
        for step in report.recovery_plan:
            issue_body += f"- {step}\n"

        # Logging for the AURA operator
        log_json("INFO", "sdlc_debugger_emit_issue", details={"task_id": task.task_id})
        # Mocking the MCP call for now since we lack the token
        # requests.post("http://localhost:8007/call", json={"tool_name": "create_issue", ...})

    async def reflect(self, report: DebugReport) -> List[str]:
        """Return lessons that the coordinator can inject into subsequent cycles."""
        return [finding.recommended_action for finding in report.findings]

    def _build_recovery_plan(self, findings: List[SDLCFinding], task: SwarmTask) -> List[str]:
        ordered = [
            f"Re-state acceptance criteria for task {task.task_id}.",
            "Write or update a failing regression test that proves the defect.",
        ]
        ordered.extend(finding.recommended_action for finding in findings)
        ordered.append("Re-run tester validation before allowing PR creation.")
        seen: set[str] = set()
        deduped: list[str] = []
        for item in ordered:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped
