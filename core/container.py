"""Dependency Injection Container for AURA CLI.

Provides a simple DI container for managing singletons and factory registrations.
This enables better testability and decoupling of components.

Example:
    >>> from core.container import Container
    >>> Container.register_singleton(DatabaseInterface, database_instance)
    >>> db = Container.resolve(DatabaseInterface)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type, TypeVar, Optional

T = TypeVar("T")


class Container:
    """Simple dependency injection container.
    
    Supports singleton and factory registrations. All registrations
    are stored at the class level for global access.
    """
    
    _singletons: Dict[Type[Any], Any] = {}
    _factories: Dict[Type[Any], Callable[[], Any]] = {}
    
    @classmethod
    def register_singleton(cls, interface: Type[T], instance: T) -> None:
        """Register a singleton instance for an interface.
        
        Args:
            interface: The type/interface being registered.
            instance: The singleton instance to return on resolve.
        """
        cls._singletons[interface] = instance
    
    @classmethod
    def register_factory(cls, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for an interface.
        
        The factory is called once on first resolve and the result
        is cached as a singleton.
        
        Args:
            interface: The type/interface being registered.
            factory: Function that creates the instance.
        """
        cls._factories[interface] = factory
    
    @classmethod
    def resolve(cls, interface: Type[T]) -> T:
        """Resolve an interface to its implementation.
        
        Args:
            interface: The type/interface to resolve.
            
        Returns:
            The registered instance for the interface.
            
        Raises:
            KeyError: If no registration exists for the interface.
        """
        # Check for existing singleton first
        if interface in cls._singletons:
            return cls._singletons[interface]
        
        # Check for factory and create singleton
        if interface in cls._factories:
            instance = cls._factories[interface]()
            cls._singletons[interface] = instance
            return instance
        
        raise KeyError(f"No registration for interface: {interface.__name__}")
    
    @classmethod
    def try_resolve(cls, interface: Type[T]) -> Optional[T]:
        """Try to resolve an interface, returning None if not registered.
        
        Args:
            interface: The type/interface to resolve.
            
        Returns:
            The registered instance or None.
        """
        try:
            return cls.resolve(interface)
        except KeyError:
            return None
    
    @classmethod
    def is_registered(cls, interface: Type[Any]) -> bool:
        """Check if an interface has a registration.
        
        Args:
            interface: The type/interface to check.
            
        Returns:
            True if registered, False otherwise.
        """
        return interface in cls._singletons or interface in cls._factories
    
    @classmethod
    def unregister(cls, interface: Type[Any]) -> bool:
        """Remove a registration for an interface.
        
        Args:
            interface: The type/interface to unregister.
            
        Returns:
            True if a registration was removed, False otherwise.
        """
        had_registration = False
        
        if interface in cls._singletons:
            del cls._singletons[interface]
            had_registration = True
        
        if interface in cls._factories:
            del cls._factories[interface]
            had_registration = True
        
        return had_registration
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations.
        
        Useful for testing to ensure a clean state between tests.
        """
        cls._singletons.clear()
        cls._factories.clear()
