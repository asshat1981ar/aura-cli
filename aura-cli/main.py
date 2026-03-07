#!/usr/bin/env python3
"""Runnable entrypoint for the aura-cli subdirectory.

Prepends the repository root to sys.path so that the canonical CLI
entrypoint in aura_cli/cli_main.py can be imported regardless of the
working directory, then delegates to it.
"""
import os
import sys


def setup() -> None:
    """Prepend the repository root (parent of this file's directory) to sys.path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


if __name__ == "__main__":
    setup()
    try:
        from aura_cli.cli_main import main  # noqa: E402
    except ModuleNotFoundError as e:
        print(
            f"Error importing AURA CLI entrypoint: {e}\n"
            "Please run this script from the repository root.",
            file=sys.stderr,
        )
        sys.exit(1)

    exit_code = main()
    if isinstance(exit_code, int):
        sys.exit(exit_code)
