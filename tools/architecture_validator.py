import ast
from pathlib import Path
from typing import List, Dict, Set

class CircularDependencyError(Exception):
    pass

def find_circular_dependencies(root_dir: Path) -> List[List[Path]]:
    """Finds circular dependencies in a Python project.

    Args:
        root_dir: The root directory of the Python project.

    Returns:
        A list of circular dependencies, where each circular dependency is a list of paths.
        Returns an empty list if no circular dependencies are found.

    Raises:
        ValueError: If the root directory does not exist.
    """
    if not root_dir.exists():
        raise ValueError(f"Root directory {root_dir} does not exist.")

    module_dependencies: Dict[Path, Set[Path]] = {}
    all_python_files: List[Path] = list(root_dir.glob("**/*.py"))

    # Build dependency graph
    for file_path in all_python_files:
        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())

            module_dependencies[file_path] = set()

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_module_name = alias.name
                            imported_module_path = resolve_import_path(root_dir, file_path, imported_module_name)
                            if imported_module_path:
                                module_dependencies[file_path].add(imported_module_path)

                    elif isinstance(node, ast.ImportFrom):
                        module_name = node.module
                        if module_name is None:
                            continue  # Skip relative imports without a module name
                        imported_module_path = resolve_import_path(root_dir, file_path, module_name)
                        if imported_module_path:
                            module_dependencies[file_path].add(imported_module_path)
        except SyntaxError as e:
            print(f"Skipping {file_path} due to syntax error: {e}")
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    # Detect cycles
    cycles: List[List[Path]] = []
    visited: Set[Path] = set()
    recursion_stack: List[Path] = []

    def dfs(node: Path) -> None:
        visited.add(node)
        recursion_stack.append(node)

        for neighbor in module_dependencies.get(node, []):
            if neighbor in recursion_stack:
                cycle_start_index = recursion_stack.index(neighbor)
                cycle = recursion_stack[cycle_start_index:] + [neighbor]  # Close the cycle
                cycles.append(cycle)
            elif neighbor not in visited:
                dfs(neighbor)

        recursion_stack.pop()

    for node in module_dependencies:
        if node not in visited:
            dfs(node)

    return cycles

def resolve_import_path(root_dir: Path, current_file: Path, module_name: str) -> Path | None:
    """Resolves the path of an imported module.

    Args:
        root_dir: The root directory of the project.
        current_file: The file where the import statement is located.
        module_name: The name of the imported module.

    Returns:
        The path to the imported module, or None if it cannot be resolved.
    """
    # Try to find the module as a file relative to the root.
    module_path = (root_dir / module_name.replace(".", "/")).with_suffix(".py")
    if module_path.exists():
        return module_path

    # If not found, check if it's a package by looking for an __init__.py in the directory
    package_path = root_dir / module_name.replace(".", "/") / "__init__.py"
    if package_path.exists():
        return package_path.parent  # Return the parent directory (the package directory)

    return None

if __name__ == "__main__":
    # Example usage
    try:
        project_root = Path(".")  # Replace with the actual root directory of your project
        circular_dependencies = find_circular_dependencies(project_root)

        if circular_dependencies:
            print("Circular dependencies found:")
            for cycle in circular_dependencies:
                print(" -> ".join(str(p.relative_to(project_root)) for p in cycle))
        else:
            print("No circular dependencies found.")
    except ValueError as e:
        print(f"Error: {e}")
    except CircularDependencyError as e:
        print(f"Error: {e}")
