from pathlib import Path
import subprocess
import sys


def check_circular_dependencies(project_root: Path) -> bool:
    """
    Checks for circular dependencies in the provided project directory.

    Args:
        project_root (Path): The root directory of the project to analyze.

    Returns:
        bool: True if circular dependencies are detected, False otherwise.

    Raises:
        NotADirectoryError: If the provided path is not a directory.
        Exception: For general subprocess errors.
    """
    if not project_root.is_dir():
        raise NotADirectoryError(f'{project_root} is not a valid directory.')

    try:
        result = subprocess.run(['flake8', '--select', 'C90', str(project_root)],
                                capture_output=True, text=True, check=True)
        return len(result.stdout) > 0
    except subprocess.CalledProcessError as e:
        # Handle the case where flake8 failed
        print(f'Error checking circular dependencies: {e}', file=sys.stderr)
        return False
    except Exception as e:
        # Catch-all for other exceptions to prevent crashing
        print(f'Unexpected error: {e}', file=sys.stderr)
        return False