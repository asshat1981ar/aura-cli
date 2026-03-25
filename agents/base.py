from abc import ABC, abstractmethod
from typing import Dict


class Agent(ABC):
    name: str
    # Declare semantic capabilities in subclasses to enable rich registry resolution.
    # First entry is the PRIMARY capability used for tie-breaking.
    # Leave empty to fall back to FALLBACK_CAPABILITIES in agents/registry.py.
    capabilities: list[str] = []

    @abstractmethod
    def run(self, input_data: Dict) -> Dict:
        """Return JSON-serializable phase output."""
        raise NotImplementedError
