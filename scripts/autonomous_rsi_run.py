#!/usr/bin/env python3
"""Deprecated RSI entrypoint that delegates to the canonical audit runner."""

from __future__ import annotations

import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.run_rsi_evolution import main as run_rsi_evolution_main


def main(argv: list[str] | None = None) -> int:
    forwarded = list(argv if argv is not None else sys.argv[1:])
    if not forwarded:
        forwarded = ["--cycles", "10"]
    return run_rsi_evolution_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
