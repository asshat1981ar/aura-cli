"""Phase registry for the modular orchestrator wrappers."""

from __future__ import annotations

from core.phases.act_phase import ActPhase
from core.phases.apply_phase import ApplyPhase
from core.phases.base import Phase, PhaseContext, PhaseResult
from core.phases.critique_phase import CritiquePhase
from core.phases.ingest_phase import IngestPhase
from core.phases.plan_phase import PlanPhase
from core.phases.reflect_phase import ReflectPhase
from core.phases.sandbox_phase import SandboxPhase
from core.phases.verify_phase import VerifyPhase


def build_phase_registry(orchestrator) -> dict[str, Phase]:
    return {
        "ingest": IngestPhase(orchestrator),
        "plan": PlanPhase(orchestrator),
        "critique": CritiquePhase(orchestrator),
        "act": ActPhase(orchestrator),
        "sandbox": SandboxPhase(orchestrator),
        "apply": ApplyPhase(orchestrator),
        "verify": VerifyPhase(orchestrator),
        "reflect": ReflectPhase(orchestrator),
    }


__all__ = [
    "Phase",
    "PhaseContext",
    "PhaseResult",
    "build_phase_registry",
]
