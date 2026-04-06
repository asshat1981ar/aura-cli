from typing import Set, Dict
import ast
from pathlib import Path

class CircularDependencyChecker:
    """
    A class to check for circular dependencies in Python files.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def check_for_circular_dependencies(self) -> Dict[str, Set[str]]:
        """
        Checks all Python files in the given directory for circular dependencies.
        Returns a dictionary where keys are filenames and values are sets of
        modules involved in circular dependencies.
        """  
        dependencies = {}
        for py_file in self.directory.glob('**/*.py'):
            with open(py_file, 'r', encoding='utf-8') as file:
                nodes = self._extract_imports(file)
                cycle = self._detect_cycle(nodes)
                if cycle:
                    dependencies[py_file.name] = cycle
        return dependencies

    def _extract_imports(self, file) -> Dict[str, Set[str]]:
        """
        Extracts import statements and returns a mapping of module names to
        sets of their dependencies.
        """
        tree = ast.parse(file.read())
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.setdefault(alias.name, set()).add('')
            elif isinstance(node, ast.ImportFrom):
                imports.setdefault(node.module, set()).add(node.level)
        return imports

    def _detect_cycle(self, graph: Dict[str, Set[str]]) -> Set[str]:
        """
        Uses a depth-first search to detect cycles in a directed graph.
        """  
        visited = set()
        stack = set()
        cyclic_nodes = set()

        def visit(node: str) -> bool:
            if node in stack:
                cyclic_nodes.add(node)
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for neighbor in graph.get(node, []):
                if visit(neighbor):
                    cyclic_nodes.add(node)
                    return True
            stack.remove(node)
            return False

        for node in graph:
            visit(node)
        return cyclic_nodes
