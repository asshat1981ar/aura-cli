from __future__ import annotations

from core.phases.base import Phase, PhaseContext, PhaseResult


class IngestPhase(Phase):
    name = "ingest"

    def run(self, context: PhaseContext) -> PhaseResult:
        return PhaseResult(payload=self.orchestrator._run_phase(self.name, context.input_data))
