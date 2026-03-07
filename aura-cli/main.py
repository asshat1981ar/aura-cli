#!/usr/bin/env python3
"""AURA CLI entry point wrapper."""
import os
import sys


def setup():
    """Initialize the AURA CLI environment and add the repo root to sys.path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


if __name__ == "__main__":
    setup()
    try:
        from aura_cli.cli_main import main as cli_main  # noqa: E402
        raise SystemExit(cli_main())
    except ImportError as e:
        print(
            f"Error: {e}\n"
            "Please run AURA from the repository root: python3 main.py",
            file=sys.stderr,
        )
        sys.exit(1)