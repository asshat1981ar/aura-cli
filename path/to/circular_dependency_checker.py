import ast
from pathlib import Path
from typing import List, Dict, Set

class CircularDependencyDetector:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _get_imports(self, filepath: Path) -> List[str]:
        with open(filepath, 'r') as file:
            node = ast.parse(file.read(), filename=str(filepath))
        imports = [n.names[0].name for n in ast.walk(node) if isinstance(n, ast.Import) or isinstance(n, ast.ImportFrom)]
        return imports

    def find_circular_dependencies(self) -> Dict[str, Set[str]]:
        modules = {f.stem: f for f in self.directory.glob('**/*.py')}
        dependencies = {mod: set(self._get_imports(path)) for mod, path in modules.items()}
        circular_deps = {mod: deps for mod, deps in dependencies.items() if mod in deps}
        return circular_deps

    def report(self) -> None:
        circular_deps = self.find_circular_dependencies()
        if circular_deps:
            for module, deps in circular_deps.items():
                print(f'Circular dependency detected in {module}: {deps}')
        else:
            print('No circular dependencies detected.')
