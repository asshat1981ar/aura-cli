"""Phase interface shared by modular orchestrator phase wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PhaseContext:
    input_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseResult:
    payload: Any
    status: str = "pass"


class Phase(ABC):
    name: str

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    @abstractmethod
    def run(self, context: PhaseContext) -> PhaseResult:
        raise NotImplementedError
