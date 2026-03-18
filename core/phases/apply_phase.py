from __future__ import annotations

from core.phases.base import Phase, PhaseContext, PhaseResult


class ApplyPhase(Phase):
    name = "apply"

    def run(self, context: PhaseContext) -> PhaseResult:
        change_set = context.input_data.get("change_set", {})
        dry_run = bool(context.input_data.get("dry_run"))
        return PhaseResult(payload=self.orchestrator._apply_change_set(change_set, dry_run=dry_run))
