from __future__ import annotations

from core.phases.base import Phase, PhaseContext, PhaseResult


class SandboxPhase(Phase):
    name = "sandbox"

    def run(self, context: PhaseContext) -> PhaseResult:
        return PhaseResult(payload=self.orchestrator._run_phase(self.name, context.input_data))
