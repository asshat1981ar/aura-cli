#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aura_cli.contract_report import (
    build_cli_contract_report,
    cli_contract_report_exit_code,
    cli_contract_report_failure_message,
    render_cli_contract_report,
)


def build_report(*, include_dispatch: bool) -> dict:
    return build_cli_contract_report(include_dispatch=include_dispatch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the aggregated CLI contract report as JSON.")
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Skip dispatch-registry parity checks and emit only static parser/help/schema checks.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the report contains contract failures.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON instead of pretty-printed output.",
    )
    args = parser.parse_args(argv)

    report = build_report(include_dispatch=not args.no_dispatch)
    print(render_cli_contract_report(report, compact=args.compact), end="")

    exit_code = cli_contract_report_exit_code(report, check=args.check)
    if exit_code:
        print(cli_contract_report_failure_message(report), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
