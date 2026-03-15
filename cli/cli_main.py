#!/usr/bin/env python3
"""
DEPRECATED: Legacy entry point.
Redirects to the new canonical entry point at aura_cli/cli_main.py.
"""
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Backward compat: surface helper commands expected in legacy tests
try:
    from cli.commands import _handle_doctor, _handle_clear  # noqa: F401
except Exception:
    _handle_doctor = None
    _handle_clear = None

from aura_cli.cli_main import main

if __name__ == "__main__":
    print("WARNING: You are using the deprecated 'cli/cli_main.py' entry point.", file=sys.stderr)
    print("Please use 'aura_cli/cli_main.py' or the 'aura' command instead.\n", file=sys.stderr)
    main()
