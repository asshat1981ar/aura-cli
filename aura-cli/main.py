import sys
import os
from pathlib import Path


def setup():
    """Prepend the repository root to sys.path so that root-level modules are importable.

    This file lives in the ``aura-cli/`` subdirectory; the importable Python
    package is ``aura_cli/`` (underscore) at the repository root.
    """
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


setup()

if __name__ == "__main__":
    try:
        from aura_cli.cli_main import main
        sys.exit(main())
    except ImportError as exc:
        print(f"Error: could not import aura_cli.cli_main: {exc}", file=sys.stderr)
        sys.exit(1)