"""Entry point shim for the aura-cli package directory.

Delegates to the canonical entry point in aura_cli.cli_main.
Provides better error diagnostics when the module cannot be loaded.
"""
import sys
import os

# Ensure the repository root is on sys.path so that aura_cli can be imported
# when this script is invoked directly (e.g. python aura-cli/main.py).
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

if __name__ == "__main__":
    try:
        from aura_cli.cli_main import main as _main
    except ImportError as exc:
        print(
            f"Error: could not import aura_cli.cli_main — {exc}\n"
            f"Make sure you are running from the repository root and that "
            f"the aura_cli package is present at: {os.path.join(_repo_root, 'aura_cli')}",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(_main())
