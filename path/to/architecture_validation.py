import ast
import os
from pathlib import Path
from typing import Set, Dict


class CircularDependencyDetector:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def _get_imports(self, root: Path) -> Dict[str, Set[str]]:
        imports = {}
        for file in root.rglob('*.py'):
            with file.open() as f:
                node = ast.parse(f.read(), filename=file.name)
                imports[file.stem] = {n.fstring for n in ast.walk(node) if isinstance(n, ast.ImportFrom)}
        return imports

    def check_for_circular_dependencies(self) -> Set[str]:
        imports = self._get_imports(self.base_path)
        visited = set()
        stack = set()  
        circular_refs = set()
        
        def visit(node):
            if node in stack:
                circular_refs.add(node)
                return
            if node in visited:
                return
            visited.add(node)
            stack.add(node)
            for neighbour in imports.get(node, []):
                visit(neighbour)
            stack.remove(node)

        for module in imports:
            visit(module)

        return circular_refs


if __name__ == '__main__':
    project_path = Path('/path/to/your/project')  # Update with actual project path
    detector = CircularDependencyDetector(project_path)
    cycles = detector.check_for_circular_dependencies()
    if cycles:
        print(f'Circular dependencies found: {cycles}')
    else:
        print('No circular dependencies found.')