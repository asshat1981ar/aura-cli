from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IInteractionService(ABC):
    @abstractmethod
    def notify_ui(self, event: str, *args: Any, **kwargs: Any) -> None:
        pass


class IOrchestrationService(ABC):
    @abstractmethod
    def run_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass


class IMemoryService(ABC):
    @abstractmethod
    def store(
        self,
        key: str,
        value: Any,
        tier: str = "project",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def retrieve(self, key: str, tier: str = "project", limit: int = 100) -> List[Any]:
        pass


class IInferenceService(ABC):
    @abstractmethod
    def call_model(self, prompt: str, model_id: Optional[str] = None) -> str:
        pass


class IToolExecutionService(ABC):
    @abstractmethod
    def execute_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        pass


class ISafetyService(ABC):
    @abstractmethod
    def validate_action(self, action_name: str, params: Dict[str, Any]) -> bool:
        pass


class IPersonalizationService(ABC):
    @abstractmethod
    def resolve_user_id(self) -> str:
        pass

    @abstractmethod
    def get_user_preference(self, key: str) -> Optional[Any]:
        pass
