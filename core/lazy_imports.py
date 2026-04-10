"""Lazy module importing to improve startup time.

This module provides utilities for deferring expensive imports until they are
actually needed, significantly reducing CLI cold start times.

Usage:
    # Instead of: from expensive_module import ExpensiveClass
    # Use:
    from core.lazy_imports import lazy_import
    ExpensiveClass = lazy_import("expensive_module", "ExpensiveClass")
    
    # Or for modules:
    from core.lazy_imports import LazyModule
    expensive_module = LazyModule("expensive_module")
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")


class LazyImport(Generic[T]):
    """Lazy importer for a specific attribute from a module.
    
    The module is not imported until the attribute is accessed.
    
    Example:
        Brain = LazyImport("memory.brain", "Brain")
        # Module not imported yet
        brain = Brain()  # Module imported here
    """
    
    def __init__(self, module_path: str, attr_name: str) -> None:
        """Initialize lazy import.
        
        Args:
            module_path: Full module path (e.g., "memory.brain")
            attr_name: Attribute name to import from module
        """
        self._module_path = module_path
        self._attr_name = attr_name
        self._module: Any = None
        self._attr: T | None = None
        self._loaded = False
    
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Allow lazy class instantiation: Obj()"""
        return self._get_attr()(*args, **kwargs)
    
    def _get_attr(self) -> T:
        """Load module and return attribute."""
        if not self._loaded:
            self._module = importlib.import_module(self._module_path)
            self._attr = getattr(self._module, self._attr_name)
            self._loaded = True
        return cast(T, self._attr)
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the actual class/module."""
        attr = self._get_attr()
        return getattr(attr, name)
    
    def __instancecheck__(self, instance: object) -> bool:
        """Support isinstance() checks."""
        return isinstance(instance, self._get_attr())
    
    def __subclasscheck__(self, subclass: type) -> bool:
        """Support issubclass() checks."""
        return issubclass(subclass, self._get_attr())


class LazyModule:
    """Lazy module importer.
    
    The module is not imported until an attribute is accessed.
    
    Example:
        numpy = LazyModule("numpy")
        # Module not imported yet
        arr = numpy.array([1, 2, 3])  # Module imported here
    """
    
    def __init__(self, module_path: str) -> None:
        """Initialize lazy module.
        
        Args:
            module_path: Full module path (e.g., "numpy.linalg")
        """
        self._module_path = module_path
        self._module: Any = None
        self._loaded = False
    
    def _get_module(self) -> Any:
        """Import and return module if not already loaded."""
        if not self._loaded:
            self._module = importlib.import_module(self._module_path)
            self._loaded = True
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the actual module."""
        return getattr(self._get_module(), name)
    
    def __dir__(self) -> list[str]:
        """Support dir() for introspection."""
        return dir(self._get_module())
    
    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self._loaded else "not loaded"
        return f"<LazyModule '{self._module_path}' ({status})>"


def lazy_import(module_path: str, attr_name: str | None = None) -> Any:
    """Create a lazy import for a module or attribute.
    
    This is the main convenience function for lazy importing.
    
    Args:
        module_path: Full module path (e.g., "memory.brain")
        attr_name: Optional attribute name. If None, returns LazyModule,
                  otherwise returns LazyImport for the attribute.
    
    Returns:
        LazyModule or LazyImport instance
    
    Examples:
        # Lazy module import
        numpy = lazy_import("numpy")
        arr = numpy.array([1, 2, 3])  # Imports numpy here
        
        # Lazy attribute import  
        Brain = lazy_import("memory.brain", "Brain")
        brain = Brain()  # Imports memory.brain here
    """
    if attr_name is None:
        return LazyModule(module_path)
    return LazyImport(module_path, attr_name)


# Pre-defined lazy imports for common heavy AURA modules
# Use these instead of direct imports for better startup performance

# Memory components
LazyBrain = LazyImport("memory.brain", "Brain")
LazyMemoryStore = LazyImport("memory.store", "MemoryStore")
LazyVectorStore = LazyImport("memory.vector_store_v2", "VectorStoreV2")

# Agents
LazyDebuggerAgent = LazyImport("agents.debugger", "DebuggerAgent")
LazyPlannerAgent = LazyImport("agents.planner", "PlannerAgent")
LazyRouterAgent = LazyImport("agents.router", "RouterAgent")
LazyScaffolderAgent = LazyImport("agents.scaffolder", "ScaffolderAgent")

# Core components
LazyLoopOrchestrator = LazyImport("core.orchestrator", "LoopOrchestrator")
LazyBeadsBridge = LazyImport("core.beads_bridge", "BeadsBridge")
LazyConfigManager = LazyImport("core.config_manager", "ConfigManager")
LazyGitTools = LazyImport("core.git_tools", "GitTools")
LazyGoalArchive = LazyImport("core.goal_archive", "GoalArchive")
LazyGoalQueue = LazyImport("core.goal_queue", "GoalQueue")
LazyModelAdapter = LazyImport("core.model_adapter", "ModelAdapter")
LazyPolicy = LazyImport("core.policy", "Policy")

# Container for lazy module access
agents_registry = LazyModule("agents.registry")


def clear_lazy_cache() -> None:
    """Clear all lazy import caches. Useful for testing."""
    # Force re-import on next access by clearing module cache
    modules_to_clear = [
        "memory.brain",
        "memory.store", 
        "memory.vector_store_v2",
        "agents.debugger",
        "agents.planner",
        "agents.router",
        "agents.scaffolder",
        "core.orchestrator",
        "core.beads_bridge",
        "core.config_manager",
        "core.git_tools",
        "core.goal_archive",
        "core.goal_queue",
        "core.model_adapter",
        "core.policy",
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]


__all__ = [
    "LazyImport",
    "LazyModule",
    "lazy_import",
    # Pre-defined lazy imports
    "LazyBrain",
    "LazyMemoryStore",
    "LazyVectorStore",
    "LazyDebuggerAgent",
    "LazyPlannerAgent",
    "LazyRouterAgent",
    "LazyScaffolderAgent",
    "LazyLoopOrchestrator",
    "LazyBeadsBridge",
    "LazyConfigManager",
    "LazyGitTools",
    "LazyGoalArchive",
    "LazyGoalQueue",
    "LazyModelAdapter",
    "LazyPolicy",
    "agents_registry",
    "clear_lazy_cache",
]
