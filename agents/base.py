from abc import ABC, abstractmethod
from typing import Dict


class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, input_data: Dict) -> Dict:
        """Return JSON-serializable phase output."""
        raise NotImplementedError
