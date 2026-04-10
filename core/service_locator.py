from typing import Dict, Type, Any


class ServiceLocator:
    _services: Dict[Type, Any] = {}

    @classmethod
    def register(cls, interface: Type, implementation: Any) -> None:
        cls._services[interface] = implementation

    @classmethod
    def get(cls, interface: Type) -> Any:
        return cls._services.get(interface)
