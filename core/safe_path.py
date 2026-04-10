"""Path traversal protection for AURA CLI.

Provides safe path resolution that prevents directory traversal attacks
by ensuring resolved paths remain within allowed base directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from core.exceptions import AuraCLIError


class PathTraversalError(AuraCLIError):
    """Raised when a path traversal attempt is detected."""
    
    def __init__(self, message: str = "Path traversal detected"):
        super().__init__(code="AURA-002", message=message)


class SafePath:
    """Path operations with traversal protection.
    
    All path operations are constrained to remain within a base directory,
    preventing directory traversal attacks.
    
    Example:
        >>> safe = SafePath("/home/user/project")
        >>> safe.resolve("src/main.py")  # OK
        Path("/home/user/project/src/main.py")
        >>> safe.resolve("../../../etc/passwd")  # Raises PathTraversalError
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize SafePath with a base directory.
        
        Args:
            base_dir: The base directory that all paths must be within.
        """
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.exists():
            raise ValueError(f"Base directory does not exist: {self.base_dir}")
        if not self.base_dir.is_dir():
            raise ValueError(f"Base path is not a directory: {self.base_dir}")
    
    def resolve(self, user_path: Union[str, Path]) -> Path:
        """Resolve user path, preventing traversal attacks.
        
        Args:
            user_path: The user-provided path to resolve.
            
        Returns:
            Resolved absolute path within base directory.
            
        Raises:
            PathTraversalError: If path attempts to escape base directory.
            ValueError: If path is empty or invalid.
        """
        if not user_path:
            raise ValueError("Path cannot be empty")
        
        # Normalize and resolve the path
        full_path = (self.base_dir / user_path).resolve()
        
        # Ensure path is within base directory
        try:
            full_path.relative_to(self.base_dir)
        except ValueError:
            raise PathTraversalError(
                f"Path '{user_path}' attempts to escape base directory"
            )
        
        # Check for symlink traversal
        if full_path.exists():
            real_path = full_path.resolve()
            try:
                real_path.relative_to(self.base_dir)
            except ValueError:
                raise PathTraversalError(
                    f"Symlink '{user_path}' points outside base directory"
                )
        
        return full_path
    
    def is_safe(self, user_path: Union[str, Path]) -> bool:
        """Check if a path is safe without raising exceptions.
        
        Args:
            user_path: The path to check.
            
        Returns:
            True if path is safe, False otherwise.
        """
        try:
            self.resolve(user_path)
            return True
        except (PathTraversalError, ValueError):
            return False
    
    def safe_join(self, *paths: Union[str, Path]) -> Path:
        """Safely join multiple path components.
        
        Args:
            *paths: Path components to join.
            
        Returns:
            Resolved path within base directory.
        """
        joined = self.base_dir.joinpath(*paths)
        return self.resolve(joined.relative_to(self.base_dir))


def create_safe_path(base_dir: Union[str, Path]) -> SafePath:
    """Factory function to create a SafePath instance.
    
    Args:
        base_dir: Base directory for path operations.
        
    Returns:
        Configured SafePath instance.
    """
    return SafePath(base_dir)
